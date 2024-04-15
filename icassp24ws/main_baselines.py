import h5py
import pickle
import argparse
import numpy as np
import json

import matplotlib.pyplot as plt
from pathlib import Path
import time

from src.metrics import *
from src import baselines

from eusipco_train import train_neural_steer
from data_utils import ( set_seed,
                    prune_domain,
                    fake_periodicity_ang_domain,
                    downsample_angles_and_flatten,
                    expand_coordinate_dims)

parser = argparse.ArgumentParser()
parser.add_argument("--data_h5", default='')
parser.add_argument("--seed", default=666, type=int)
parser.add_argument("--ds", default=2, type=int)
parser.add_argument("--grid_type", default="regular")
parser.add_argument("--grid_pattern", default="pincheck")
parser.add_argument("--model", default="scf")
parser.add_argument("--domain", default='freq', type=str)
parser.add_argument("--nfft", default=512, type=int)
parser.add_argument("--n_mics", type=int, default=1)
parser.add_argument("--exp_dir", type=str, default="exp/tmp", help="Full path to save best validation model")

def load_data(path_to_data, ind_mics):
    n_mics = len(ind_mics)
    f = h5py.File(path_to_data,'r')
    fs = int(f["fs"][()])
    doas = np.array(f['doas']) # (ndarray) [n_az x n_el x 2]
    airs = np.array(f['airs'])[:,:,ind_mics,:]       # (ndarray) [n_az x n_el x n_chan x n_smpl]
    mic_pos = np.array(f['mic_pos'])[:,ind_mics]     # (ndarray) [3 x n_chan]
    f.close()

    target = airs    
    coords = doas

    assert coords.shape[:2] == target.shape[:2]

    return coords, target, mic_pos, fs


def make_data(data_conf, exp_dir):

    h5_path = data_conf["h5_path"]
    ind_mics = data_conf["ind_mics"]
    domain = data_conf["domain"]
    nfft = data_conf["nfft"]
    az_lim = data_conf["az_lim"]  # degree
    el_lim = data_conf["el_lim"]  # degree
    feat_lim = data_conf["feat_lim"] # digital freqs
    ds = data_conf["ds"]
    grid_type = data_conf["grid_type"]
    grid_pattern = data_conf["grid_pattern"]

    if grid_type == "regular":
        assert ds in [1,2,3,4,5]
    elif grid_type == "random":
        assert ds > 1 and ds < 100
    else:
        raise ValueError("Wrong grid ds for grid_type, got {} with ds {}".format(grid_type, ds))


    ## MAKE DATA
    print("Input Data")
    gt_coords, gt_target, mic_pos, fs = load_data(h5_path, ind_mics=ind_mics)
    
    if domain == "freq":
        gt_target = np.fft.rfft(gt_target, n=nfft, axis=-1)

    print("GT data:", gt_coords.shape, gt_target.shape)
    to_plot = np.rad2deg(gt_coords.reshape(-1,2))

    plt.figure(figsize=(12,6))
    plt.scatter(to_plot[:,0], to_plot[:,1], facecolors='none', edgecolors='k', label='GroundTruth')

    gt_coords, gt_target = fake_periodicity_ang_domain(gt_coords, gt_target)
    print("GT fake periodicity:", gt_coords.shape, gt_target.shape)
    to_plot = np.rad2deg(gt_coords.reshape(-1,2))
    plt.scatter(to_plot[:,0], to_plot[:,1], alpha=0.5, label='After exended')

    gt_coords, gt_target = prune_domain(gt_coords, gt_target, az_lim, el_lim, feat_lim)
    print("GT prune:", gt_coords.shape, gt_target.shape)
    to_plot = np.rad2deg(gt_coords.reshape(-1,2))
    plt.scatter(to_plot[:,0], to_plot[:,1], edgecolors='C1', facecolors=None, label='After pruning') 

    print("Consider freqs and coordinates")
    gt_coords, gt_target = expand_coordinate_dims(gt_coords, gt_target)
    print("GT exp:", gt_coords.shape, gt_target.shape)
    
    # Prepare traning dataset
    print("Dowsample and flatten")
    ds_coords, ds_target, diff_coords, diff_target = downsample_angles_and_flatten(
        gt_coords, gt_target, ds=ds, 
        grid_type=grid_type, grid_pattern=grid_pattern
    )
    print("DS:", ds_coords.shape, ds_target.shape)

    ds_n_az = np.unique(ds_coords.reshape(-1,3)[:,0]).shape[0]
    ds_n_el = np.unique(ds_coords.reshape(-1,3)[:,1]).shape[0]
    ds_n_feat = np.unique(ds_coords.reshape(-1,3)[:,2]).shape[0]
    gt_n_az = np.unique(gt_coords.reshape(-1,3)[:,0]).shape[0]
    gt_n_el = np.unique(gt_coords.reshape(-1,3)[:,1]).shape[0]
    gt_n_feat = np.unique(gt_coords.reshape(-1,3)[:,2]).shape[0]
    diff_n_az = np.unique(diff_coords.reshape(-1,3)[:,0]).shape[0]
    diff_n_el = np.unique(diff_coords.reshape(-1,3)[:,1]).shape[0]
    diff_n_feat = np.unique(diff_coords.reshape(-1,3)[:,2]).shape[0]
    
    to_plot_ds = np.rad2deg(ds_coords[:,0,:])
    to_plot_diff = np.rad2deg(diff_coords[:,0,:])
    plt.scatter(to_plot_ds[:,0], to_plot_ds[:,1], facecolors='C3', edgecolors='C3', label='Traning Points') 
    plt.scatter(to_plot_diff[:,0], to_plot_diff[:,1], marker="X", facecolors='C4', edgecolors='C4', label='Testing Diff Point') 
    plt.xlabel("Azimuth [degree]")
    plt.ylabel("Elevation [degree]")
    plt.title("Input Coordinates")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(Path(exp_dir) / Path("input_data_grid.pdf"), dpi=300)
    plt.close()

    data_dict = {
        "fs" : fs,
        "mic_pos" : mic_pos,
        "ds" : {"coords":ds_coords,"target":ds_target, "n_az":ds_n_az, "n_el":ds_n_el, "n_feat":ds_n_feat},
        "gt" : {"coords":gt_coords,"target":gt_target, "n_az":gt_n_az, "n_el":gt_n_el, "n_feat":gt_n_feat},
        "diff" : {"coords":diff_coords,"target":diff_target, "n_az":diff_n_az, "n_el":diff_n_el, "n_feat":diff_n_feat},
    }

    return data_dict 


def run(conf):

    exp_dir = Path(conf["exp_dir"])
    exp_dir.mkdir(exist_ok=True, parents=True)

    set_seed(seed=conf["seed"])
    
    if conf["n_mics"] == 1:
        ind_mics = [1]
    else:
        ind_mics = range(conf["n_mics"])

    domain = conf["domain"]
    nfft = conf["nfft"]

    data_conf = {
        "h5_path" : Path(conf["data_h5"]),
        "ind_mics" : ind_mics,
        "domain" : domain,
        "nfft"  : nfft,
        "az_lim" : [-15, 375], # degree
        "el_lim" : [-91,  91],  # degree
        "feat_lim" : [0, None],
        "grid_type" : conf["grid_type"],
        "grid_pattern": conf["grid_pattern"],
        "ds" : conf["ds"],
    }

    ## DATA
    data_dict = make_data(data_conf, exp_dir)

    ## INIT MODEL
    model =  baselines.SCFInterpolator(algo='linear')

    ## INIT METRICS
    compute_metrics = lambda x, y : eusipco_metrics(x, y, domain, nfft, freq_axis=-2)

    print('-- FITTING --')
    start_time = time.time()
    model.fit(data_dict["ds"]["coords"], data_dict["ds"]["target"])
    fitting_time = time.time() - start_time

    # eval on ds
    print('-- VALIDATION -- ')
    eval_metrics = {}
    start_time = time.time()
    est_train = model(data_dict["ds"]["coords"])
    eval_time = time.time() - start_time
    eval_metrics["val"] = compute_metrics(est_train, data_dict["ds"]["target"])

    print("-- TEST DIFF --")
    start_time = time.time()
    est_diff = model(data_dict["diff"]["coords"])
    test_time = time.time() - start_time
    eval_metrics["diff"] = compute_metrics(est_diff, data_dict["diff"]["target"])

    print('-- TEST ALL --')
    start_time = time.time()
    est_test = model(data_dict["gt"]["coords"])
    test_time = time.time() - start_time
    eval_metrics["test"] = compute_metrics(est_test, data_dict["gt"]["target"])

    eval_metrics["train"] = { "time" : fitting_time }
    eval_metrics['val']["time"] = eval_time
    eval_metrics['test']["time"] = test_time

    print(eval_metrics)

    ## PLOT
    data_pts = data_dict["gt"]["coords"]
    data_ref = data_dict["gt"]["target"]
    data_est = est_test

    idx_equator_coords = data_pts[...,1] == 0
    ref_to_plot = data_ref[idx_equator_coords].reshape(data_ref.shape[0], *data_ref.shape[2:])[...,1]  
    est_to_plot = data_est[idx_equator_coords].reshape(data_ref.shape[0], *data_ref.shape[2:])[...,1]

    if domain == "time":
        est_freq = np.fft.rfft(est_to_plot, n=nfft, axis=-1)
        ref_freq = np.fft.rfft(ref_to_plot, n=nfft, axis=-1)
        est_time = est_to_plot[:, :80]
        ref_time = ref_to_plot[:, :80]
    elif domain == "freq":
        est_time = np.fft.irfft(est_to_plot, n=nfft, axis=-1)[:, :80]
        ref_time = np.fft.irfft(ref_to_plot, n=nfft, axis=-1)[:, :80]
        est_freq = est_to_plot
        ref_freq = ref_to_plot

    # plot time domain
    plt.figure(figsize=(10,6))
    plt.suptitle(f"Time Domain - Elevation = 0 deg")
    plt.subplot(311)
    plt.title("Estimated")
    plt.imshow(est_time.T, aspect='auto')
    plt.colorbar()
    plt.ylabel("Time [sec]")
    plt.subplot(312)
    plt.title("Reference")
    plt.imshow(ref_time.T, aspect='auto')
    plt.colorbar()
    plt.subplot(313)
    plt.title(f'RMSE: {eval_metrics["test"]["rmse_time"]:.4f}')
    plt.imshow(np.abs(ref_time - est_time).T, aspect='auto')
    plt.xlabel("Azimuths [degree]")
    plt.ylabel("Time [sec]")
    plt.colorbar()
    save_to = Path(exp_dir, 'Time_vs_Azimuth_el0.png')
    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()


    # # plot in freq domain
    plt.figure(figsize=(10,8))
    plt.suptitle(f"Freq domain - Elevation = 0 deg")
    
    plt.subplot(321)
    plt.title("Reference: Mag")
    plt.imshow(20*np.log10(np.abs(est_freq)+ EPS).T, aspect='auto', vmin=-40)
    plt.colorbar()
    plt.ylabel("Frequencies [Hz]")
    
    plt.subplot(322)
    plt.title("Reference: Phase")
    plt.imshow(np.unwrap(np.angle(est_freq), axis=-1).T, aspect="auto")
    plt.colorbar()

    plt.subplot(323)
    plt.title("Estimated: Mag")
    plt.imshow(20*np.log10(np.abs(ref_freq) + EPS).T, aspect="auto", vmin=-40)
    plt.ylabel("Frequencies [Hz]")
    plt.colorbar()

    plt.subplot(324)
    plt.title("Estimated: Phase")
    plt.imshow(np.unwrap(np.angle(est_freq), axis=-1).T, aspect="auto")
    plt.ylabel("Frequencies [Hz]")
    plt.colorbar()

    plt.subplot(325)
    plt.title(f'RMSE: {eval_metrics["test"]["lsd"]:.4f}')
    plt.imshow(20*np.log10(np.abs(ref_freq - est_freq)+ EPS).T, aspect="auto", vmin=-40)
    plt.xlabel("Azimuths [degree]")
    plt.ylabel("Frequencies [Hz]")
    plt.colorbar()

    plt.subplot(326)
    plt.title(f'RMSE: {eval_metrics["test"]["rmse_phase"]:.4f}')
    plt.imshow(np.unwrap(np.angle(ref_freq - est_freq), axis=-1).T, aspect="auto")
    plt.xlabel("Azimuths [degree]")
    plt.ylabel("Frequencies [Hz]")
    plt.colorbar()

    save_to = Path(exp_dir, 'Freq_vs_Azimuth_el0.png')
    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()

    # results_dict = {
    #     "conf" : conf,
    #     "metrics" : eval_metrics,
    #     "model_params" : model_params,
    # }

    # save_to = Path(exp_dir, 'results.pkl')
    # with open(save_to, 'wb') as f:
    #     pickle.dump(results_dict, f) 
    # print(f"Done. Results saved in {exp_dir}")

    # del eval_metrics['val']['rmse_phase_freq']
    # del eval_metrics['diff']['rmse_phase_freq']
    # del eval_metrics['test']['rmse_phase_freq']
    # del eval_metrics['val']['lsd_freq']
    # del eval_metrics['diff']['lsd_freq']
    # del eval_metrics['test']['lsd_freq']
    
    # save_to = Path(exp_dir, "results.json")
    # with open(save_to, "w") as f:
    #     json.dump(eval_metrics, f, indent=0)

    # save_to = Path(exp_dir, "main_conf.json")
    # with open(save_to, "w") as f:
    #     json.dump(conf, f, indent=0)

    # return eval_metrics


if __name__ == '__main__':
    args = parser.parse_args()
    conf = vars(args)
    run(conf)