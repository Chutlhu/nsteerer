import os
import json
import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
from pytorch_lightning import loggers
import optuna
import warnings

from pprint import pprint
from pathlib import Path

from src.models.models import HRTF_FIELD
from src.data.easycom import EasycomDataModule, EusipcoDataset

class PyTorchLightningPruningCallback(Callback):
    """PyTorch Lightning callback to prune unpromising trials.
    See `the example <https://github.com/optuna/optuna-examples/blob/
    main/pytorch/pytorch_lightning_simple.py>`__
    if you want to add a pruning callback which observes accuracy.
    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``pytorch_lightning.LightningModule.training_step`` or
            ``pytorch_lightning.LightningModule.validation_epoch_end`` and the names thus depend on
            how this dictionary is formatted.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        super().__init__()

        self._trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # When the trainer calls `on_validation_end` for sanity check,
        # do not call `trial.report` to avoid calling `trial.report` multiple times
        # at epoch 0. The related page is
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        epoch = pl_module.current_epoch

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self.monitor)
            )
            warnings.warn(message)
            return

        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)

def train_neural_steer(pts, obs, conf, exp_dir, viz_data=None, opt_trial=None):

    pl.seed_everything(conf["seed"])

    dset = EusipcoDataset(pts, obs, do_freqs_in=conf["do_freqs_in"]) 
    n_rfft = conf["n_rfft"]
    assert n_rfft in [257, 513, 1025, 2049]

    n_obs = len(dset)
    n_val  = int((conf["val_perc"] / 100) * n_obs)
    n_train = n_obs - n_val
    print("N tra", n_train)
    print("N val", n_val)

    dset_train, dset_val = torch.utils.data.random_split(dset, [n_train, n_val])
    dm = EasycomDataModule(
        train_set = dset_train,
        val_set = dset_val,
        batch_size=conf["batch_size"],
        num_workers=4,
    )

    # Define model and optimizer
    optim_conf = {
        "half_lr" : conf["half_lr"],
        "lr"      : conf["lr"],
        "weight_decay" : 0.,
        "lars"    : conf["lars"]
}
    acu_geo_params = {
        "mic_pos" : conf["mic_pos"].tolist(),
        "fs" : conf["fs"],
        "n_rfft" : n_rfft
    }

    assert conf['mic_pos'].shape[0] == 3
    n_mics = conf["mic_pos"].shape[-1]
    in_dim = 3 if conf["do_freqs_in"] else 2
    if "_PHASE" in conf["architecture"]:
        out_dim = 2 + n_mics
    else:
        out_dim = 2 * (1 + n_mics) + 1

    model_conf = {
        "do_freqs_in": conf["do_freqs_in"],
        "in_dim": in_dim,
        "out_dim": out_dim,
        "do_skip": conf["do_skip"],
        "architecture": conf["architecture"],
        "hidden_dim" : conf["hidden_dim"],
        "hidden_num" : conf["hidden_num"],
        "feature_scale_ang" : conf["scale_ang"],
        "feature_scale_freq" : conf["scale_freq"],
        "freq_masking": False,
        "freq_range": [200, 20000],
        "do_svect": conf["do_svect"],
        "do_bar"  : conf["do_bar"],
        "do_delay": conf["do_delay"],
        "do_calib": conf["do_calib"],
    }

    loss_conf = {
        "loss_atf" : conf["loss_atf_lam"],
        "loss_atf_fun": "l2"  ,
        "loss_mag" : conf["loss_mag_lam"],
        "loss_mag_fun": "l1"  ,
        "loss_mag_scale": "log",
        "loss_phase": conf["loss_phase_lam"],
        "loss_phase_fun": "l1",
        "loss_air" : conf["loss_air_lam"],
        "loss_air_fun" : "l1",
        "loss_air_ndoas": 180,
        "loss_air_nsmpl": 150,
        "loss_causality": conf["loss_caus_lam"],
        "loss_causality_crit": conf["loss_caus_crit"],
        "loss_causality_fun": "l1",
        "loss_rtf": conf["loss_rft_lam"],
        "loss_rtf_fun": "l1",
    }
    
    model = HRTF_FIELD(
        exp_dir, 
        **model_conf, 
        **acu_geo_params, 
        **{"loss_config_dict" : loss_conf},
        **{"optim_config_dict": optim_conf},
    )
    if not viz_data is None:
        model.viz_hr_coords = viz_data["coords"]
        model.viz_hr_target = viz_data["target"]

    # Just after instantiating, save the args. Easy loading in the future.
    conf.pop("mic_pos")
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor=conf["es_metric"], mode="min", save_top_k=3, verbose=True
    )
    callbacks.append(checkpoint)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    if not opt_trial is None:
        opt_callbacks = PyTorchLightningPruningCallback(
            opt_trial, monitor=conf["es_metric"])
        callbacks.append(opt_callbacks)
    
    if conf["early_stopping"]:
        callbacks.append(EarlyStopping(monitor=conf["es_metric"], mode="min", patience=10, verbose=True))

    tb_logger = loggers.TensorBoardLogger(exp_dir)
    trainer = pl.Trainer(
        max_epochs=conf["epochs"],
        logger=tb_logger,
        callbacks=callbacks,
        default_root_dir=exp_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=2.0,
        check_val_every_n_epoch=conf['val_interval']
    )

    trainer.fit(model, dm)
    print("Final Results")
    pprint(trainer.callback_metrics)    

    # save checkpoints loss
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    if len(best_k.items()): 
        with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
            json.dump(best_k, f, indent=0)

    # save best model
    state_dict = torch.load(checkpoint.best_model_path)
    model.load_state_dict(state_dict=state_dict["state_dict"])

    save_dir = Path(exp_dir, "best_model.pth")
    to_save = {
        "path_to_checkpoint" : checkpoint.best_model_path,
        "last_loss" : trainer.callback_metrics,
        "last_epoch" : model.current_epoch,
        "hparams" : tb_logger.hparams,
    }
    torch.save(to_save, save_dir)
       
    # try to load
    model = HRTF_FIELD.load_from_checkpoint(checkpoint.best_model_path)
    model.cpu()
    return model