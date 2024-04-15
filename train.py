import os
import argparse
import json

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import loggers

import numpy as np

from pathlib import Path
from src.models.models import HRTF_FIELD
from src.data.easycom import EasycomDataModule, EasycomDataset
from src.metrics import eusipco_metrics

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="local/conf.yml")
parser.add_argument("--data_h5")
parser.add_argument("--tag", default="tmp")
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")

def main(conf):
    exp_dir = conf["options"]["exp_dir"]
    seed=conf["training"]["seed"]
    
    pl.seed_everything(seed)

    dset_train = EasycomDataset(
        "train",
        conf["options"]["data_h5"],
        conf['model']['do_freqs_in'],
        conf["data"]["n_mics"],
        conf["data"]["n_rfft"],
        conf["training"]["train_idx"]
    )
    dset_train.prepare_data()
    
    dset_viz_hr = EasycomDataset(
        "viz_hr",
        conf["options"]["data_h5"],
        conf['model']['do_freqs_in'],
        conf["data"]["n_mics"],
        conf["data"]["n_rfft"],
        {  "az" :  [0, -1, 1], "el" : [9, 11, 2]}
    )
    dset_viz_hr.prepare_data()


    if conf['model']['do_freqs_in']:
        conf['model']['in_dim'] = 2

    # split train/val/test dataset
    n_obs = len(dset_train)
    n_train  = int((conf["training"]["train_val_split_perc"] / 100) * n_obs)
    n_val = n_obs - n_train
    subdset_train, subdset_val = torch.utils.data.random_split(dset_train, [n_train, n_val])
    print("Train/val split size", len(subdset_train), len(subdset_train))
    
    # # create viz dataset
    subdset_viz_train = torch.utils.data.random_split(subdset_train, [2, len(subdset_train)-2])[0]
    subdset_viz_val = torch.utils.data.random_split(subdset_val, [2, len(subdset_val)-2])[0]

    dm = EasycomDataModule(
        train_set = subdset_train,
        val_set = subdset_val,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
    )
    
    # Define model and optimizer
    conf["optim"]["optim_config_dict"]["half_lr"] = conf["training"]["half_lr"]
    acu_geo_params = {
        "mic_pos" : dset_train.mic_pos,
        "fs" : dset_train.fs,
        "n_rfft" : conf["data"]["n_rfft"]
    }
    
    model = HRTF_FIELD(exp_dir, 
                        **conf["model"], 
                        **acu_geo_params, 
                        **{"loss_config_dict" : conf["loss"]},
                        **conf["optim"])
    
    # for vizualization purpose 
    model.viz_hr_set = dset_viz_hr
    model.viz_lr_set = {
        "train" : subdset_viz_train,
        "val" : subdset_viz_val,
    }
    
    # Just after instantiating, save the args. Easy loading in the future.
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val/rmse_time", mode="min", save_top_k=3, verbose=True
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(monitor="val/rmse_time", mode="min", patience=10, verbose=True))

    tb_logger = loggers.TensorBoardLogger(exp_dir)
    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        logger=tb_logger,
        callbacks=callbacks,
        default_root_dir=exp_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=conf["training"]["precision"],
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=2.0,
        check_val_every_n_epoch=conf["training"]['check_val_every_n_epoch']
    )
    
    trainer.fit(model, dm)
    print("Final Results")
    pprint(trainer.callback_metrics)    
    
    # save checkpoints loss
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    # save best model
    state_dict = torch.load(checkpoint.best_model_path)
    model.load_state_dict(state_dict=state_dict["state_dict"])
    model.cpu()
    save_dir = Path(exp_dir, "best_model.pth")

    # test data
    dset_test = EasycomDataset(
        "test",
        conf["options"]["data_h5"],
        conf['model']['do_freqs_in'],
        conf["data"]["n_mics"],
        conf["data"]["n_rfft"],
        {  "az" :  [0, None, 1], "el" : [0, None, 1]}
    )
    dset_test.prepare_data()
    coords, extra = dset_test[:]
    ref = extra["atfs"]
    est = model(coords)[0]
    ref = ref.detach().cpu().numpy()
    est = est.detach().cpu().numpy()
    results = eusipco_metrics(est, ref)

    print("rmse_time", results["rmse_time"])
    print("rmse_phase", results["rmse_phase"])
    print("lsd", results["lsd"])
    
    print("Model saved in", save_dir)
    to_save = {
        "model_state_dict" : model.state_dict,
        "loss" : trainer.callback_metrics,
        "epoch" : model.current_epoch,
        "results" : results
    }
    torch.save(to_save, save_dir)


if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    with open(parser.parse_known_intermixed_args()[0].config) as f:
        def_conf = yaml.safe_load(f)
    def_conf['optional arguments'] = {}
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    pprint(arg_dic)
    main(arg_dic)
