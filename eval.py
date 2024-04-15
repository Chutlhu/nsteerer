import os
import argparse
import json
import yaml


import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import loggers

import numpy as np

from pathlib import Path
from src.models.models import HRTF_FIELD
from src.data.easycom import EasycomDataModule, EasycomDataset

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument("--path_to_config", type=str)
parser.add_argument("--path_to_data_hr", type=str)
parser.add_argument("--path_to_data_lr", type=str)
parser.add_argument("--path_to_model", type=str)
parser.add_argument("--train_val_portion", type=int)
parser.add_argument("--model", type=str)

def main(args):

    path_to_config = args.path_to_config
    path_to_data_lr = args.path_to_data_lr
    if args.path_to_data_hr is None:
        path_to_data_hr = path_to_data_lr
    else:
        path_to_data_hr = args.path_to_data_hr
    path_to_model = args.path_to_model
    device = args.device

    with open(Path(path_to_config)) as f:
        conf = yaml.safe_load(f)

    dset_train = EasycomDataset(
        "train",
        path_to_data_lr,
        conf['model']['do_freqs_in'],
        conf["data"]["n_mics"],
        conf["data"]["normalize_coords"],
        {  "az" :  [0, -1, 2], "el" : [0, -1, 2]}
    )
    
    dset_test = EasycomDataset(
        "test",
        path_to_data_hr,
        conf['model']['do_freqs_in'],
        conf["data"]["n_mics"],
        conf["data"]["normalize_coords"],
        {  "az" :  [0, -1, 1], "el" : [0, -1, 1]}
    )

    # # split train/val/test dataset
    # az_idx = np.arange(test_set_freq_lr.n_az)
    # el_idx = np.arange(test_set_freq_lr.n_el)
    # az_el_idx = np.stack(np.meshgrid(az_idx, el_idx, indexing='ij'), axis=-1).reshape(-1,2)
    # n_doa = az_el_idx.shape[0]
    # indices = np.arange(n_doa)
    # indices = np.random.permutation(np.arange(n_doa))
    # n_train_perc = conf["training"]["train_val_split_perc"]
    # n_train = int((n_doa / 100) * n_train_perc)
    # n_val = n_doa - n_train
    # train_idx, val_idx = indices[:n_train], indices[n_train:n_train+n_val]    
    # test_set_freq_lr.prepera_data(az_el_idx[train_idx,:])
    # test_set_freq_hr.prepera_data(az_el_idx[val_idx,:])
    dset_train.prepera_data()
    dset_test.prepera_data()

    doas_train = dset_train.doas
    hrtf_train = dset_train.atfs

    doas_test = dset_test.doas
    hrtf_test = dset_test.atfs

    if args.model == 'spline':
        from scipy.interpolate import SmoothSphereBivariateSpline
        lat = doas_train[:,1].cpu().numpy() + np.pi/2
        lon = doas_train[:,0].cpu().numpy()
        data = np.abs(hrtf_train.cpu().numpy()).squeeze()

    else:
        # Define model and optimizer
        acu_geo_params = {
            "mic_pos" : dset_train.mic_pos,
            "fs" : dset_train.fs,
            "nrfft" : conf["data"]["nrfft"]
        }
        path_to_exp = Path(path_to_model)
        # import ipdb; ipdb.set_trace()
        model = HRTF_FIELD(path_to_exp.parents[0], 
                            **conf["model"], 
                            **acu_geo_params, 
                            **{"loss_config_dict" : conf["loss"]},
                            **conf["optim"])


        best_model_dict = torch.load(Path(path_to_model))

        # Load best model
        import ipdb; ipdb.set_trace()
        model = best_model_dict["model_state_dict"]
        model.cpu()

    for data_freq in [test_set_freq_lr, test_set_freq_hr]:
        
        data = data_freq[:]
        data = data.to(device)
        print(data)
        1/0


if __name__ == "__main__":
    from pprint import pprint
    args = parser.parse_args()

    # Change this to specify GPU
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    conf = vars(args)
    pprint(conf)
    main(args)