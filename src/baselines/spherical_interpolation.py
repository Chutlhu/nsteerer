import h5py
import pickle
import argparse
import numpy as np
import scipy.interpolate as interp
from tqdm import tqdm

import matplotlib.pyplot as plt
from pathlib import Path
import scipy
import time

from src.metrics import *

from pprint import pprint


parser = argparse.ArgumentParser()
parser.add_argument("--data_h5")
parser.add_argument("--n_eigh", default=32, type=int)
parser.add_argument("--ds", default=1, type=int)
parser.add_argument("--grid_type", default="regular")
parser.add_argument("--exp_dir", type=str, default="exp/", help="Full path to save best validation model")


def load_data(path_to_data, ind_mics, data='atfs'):

    n_mics = len(ind_mics)
    f = h5py.File(path_to_data,'r')
    fs = int(f["fs"][()])
    doas = np.array(f['doas']) # (ndarray) [n_az x n_el x 2]
    airs = np.array(f['airs'])[:,:,ind_mics,:]       # (ndarray) [n_az x n_el x n_chan x n_smpl]
    atfs = np.array(f['atfs'])[...,ind_mics]         # (ndarray) [n_az x n_el x n_rfft x n_chan]
    rtfs = np.array(f['rtfs'])[...,ind_mics,ind_mics] # (ndarray) [n_az x n_el x n_rfft x n_chan x n_chan]
    mic_pos = np.array(f['mic_pos'])[:,ind_mics]     # (ndarray) [3 x n_chan]
    f.close()

    atfs = atfs.transpose(0,1,3,2)

    assert atfs.shape[:3] == airs.shape[:3]
    print(atfs.shape)

    if data == "atfs":
        targets = atfs
    if data == "airs":
        targets = airs
    
    coords = doas

    assert coords.shape[:2] == targets.shape[:2]

    return coords, targets


def prune_domain(coords, target, az_lim, el_lim, feat_lim):
    idx_coords_az = (np.rad2deg(coords[:,:,0]) >= az_lim[0]) & (np.rad2deg(coords[:,:,0]) <= az_lim[1])
    idx_coords_el = (np.rad2deg(coords[:,:,1]) >= el_lim[0]) & (np.rad2deg(coords[:,:,1]) <= el_lim[1])
    n_el = np.count_nonzero(np.count_nonzero(idx_coords_az & idx_coords_el, axis=0))
    n_az = np.count_nonzero(np.count_nonzero(idx_coords_az & idx_coords_el, axis=1))
    coords = coords[idx_coords_az & idx_coords_el].reshape(n_az, n_el, 2)
    target = target[idx_coords_az & idx_coords_el].reshape(n_az, n_el, target.shape[-2], target.shape[-1])
    target = target[...,feat_lim[0]:feat_lim[1]]
    return coords, target


def downsample_angles_and_flatten(coords, target, ds, grid_type='regular'):

    if grid_type == 'regular':
        # target = target[::ds,::ds,:,:].reshape(-1, target.shape[-1])
        # coords = coords[::ds,::ds,:,:].reshape(-1, coords.shape[-1])
        ds_target = target.reshape(-1, *target.shape[-2:])[::ds,:,:]
        ds_coords = coords.reshape(-1, *coords.shape[-2:])[::ds,:,:]

    elif grid_type == 'random':
        ds_target = target.reshape(-1, *target.shape[-2:])
        ds_coords = coords.reshape(-1, *coords.shape[-2:])
        assert ds_target.shape[:-1] == ds_coords.shape[:-1]
        n_obs = ds_coords.shape[0]
        n_ds_obs = int(np.round(n_obs * ds))
        idx = np.random.randint(0, n_obs, n_ds_obs)
        ds_target = ds_target[idx,...]
        ds_coords = ds_coords[idx,...]

    else:
        raise ValueError("Wrong grid type")
    
    assert ds_target.shape[-2:] == target.shape[-2:]
    assert ds_coords.shape[-2:] == coords.shape[-2:] 

    return ds_coords, ds_target


def expand_coordinate_dims(coords, targets):
    
    n_dim = targets.shape[-1]
    targets_dims = targets.transpose(0,1,3,2)

    n_az, n_el = coords.shape[:2]
    domain_support = np.linspace(0., 1., n_dim)
    coords_dims = np.repeat(coords[:,:,None,:], n_dim, axis=2)
    dims = np.tile(domain_support, (n_az, n_el, 1))
    coords_dims = np.concatenate([coords_dims, dims[...,None]], axis=-1)

    assert targets_dims.shape[:3] == coords_dims.shape[:3]
    return coords_dims, targets_dims


def sph2cart(az, el, r):
    return np.stack([
        r * np.cos(el) * np.cos(az),
        r * np.cos(el) * np.sin(az),
        r * np.sin(el)
    ], axis=1)


def sph2cart_mesh_with_dims(coords, targets):
    assert len(coords.shape) == 4

    n_az, n_el, n_dims = coords.shape[:3]
    coords_flat = coords.reshape(-1,3)

    coords_cart = sph2cart(coords_flat[:,0], coords_flat[:,1], 2*np.ones_like(coords_flat[:,0]))
    coords_cart = coords_cart.reshape(n_az, n_el, 1, n_dims, 3)
    _coords_cart = np.zeros((n_az, n_el, 1, n_dims, 4))    
    _coords_cart[...,0] = coords_cart[...,0]
    _coords_cart[...,1] = coords_cart[...,1]
    _coords_cart[...,2] = coords_cart[...,2]
    _coords_cart[...,3] = coords[...,None,:,-1]
    return _coords_cart, targets


def haversine_cdist(XA, XB):
    lat1 = XA[1]
    lat2 = XB[1]
    lon1 = XA[0]
    lon2 = XB[0]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    h = 2 * np.arcsin(np.sqrt(a))
    ddist = np.abs(XA[2] - XB[2])
    return np.sqrt(h**2 + ddist**2)


class SCF():
    def __init__(self, coords, rirs, params):
        assert coords.shape[:-1] == rirs.shape[:-1]
        self.coords = coords
        self.n_feat = coords.shape[-2]
        self.rirs = rirs
        self.n_eigh = params["n_eigh"]
        self.n_fft = 2 * (self.n_feat - 1)
        self.current_epoch = 0
        self.n_mics = rirs.shape[-1]
        self.scf_interp = None
        # self.fit()


    # def fit(self):
        # # make rir matrix
        # H = self.rirs.reshape(-1,*self.rirs.shape[-2:]).transpose(1,0,2)
        # assert H.shape[0] == self.n_feat
        # self.h_bar = np.mean(H, axis=1, keepdims=True)
        # assert self.h_bar.shape[0] == self.n_feat
        
        # pts = self.coords[...,0,:-1].reshape(-1,2)

        # self.scf_interp = []

        # for i in range(self.n_mics):
        #     # compute the base function
        #     C = np.cov(H[...,i])
        #     vals, vects = np.linalg.eigh(C)
        #     # sort
        #     vals = vals[::-1][:self.n_eigh]
        #     self.eigen_vect = vects[:,::-1][:,:self.n_eigh]

        #     # compute the SCF
        #     scf = self.eigen_vect.conj().T @ (H[...,i] - self.h_bar[...,i])

        #     # iterpolate SCF
        #     obs = scf.T
        #     interp_fun = fitting(pts, obs, method="linear")
        #     scf = interp_fun(pts)
        #     print(np.mean(np.abs(obs - scf)**2))
        #     est = self.h_bar[...,i].T + scf @ self.eigen_vect.T
        #     print(np.mean(np.abs(self.rirs[...,i] - est)**2))


    def __call__(self, coords):
        test_pts = coords[...,0,:-1].reshape(-1,2)

        H = self.rirs.reshape(-1,*self.rirs.shape[-2:]).transpose(1,0,2)
        assert H.shape[0] == self.n_feat
        self.h_bar = np.mean(H, axis=1, keepdims=True)
        assert self.h_bar.shape[0] == self.n_feat
        train_pts = self.coords[...,0,:-1].reshape(-1,2)

        assert test_pts.shape[1:] == train_pts.shape[1:]
        
        self.n_eigh = 257
        h_i = []
        for i in range(self.n_mics):
            # compute the base function
            C = np.cov(H[...,i])
            vals, vects = np.linalg.eigh(C)
            # sort
            vals = vals[::-1][:self.n_eigh]
            self.eigen_vect = vects[:,::-1][:,:self.n_eigh]

            # compute the SCF
            scf = self.eigen_vect.T.conj() @ H[...,i]
            est_ = self.eigen_vect @ scf
            print(np.mean(np.abs(H[...,i] - est_)**2))
            
            # iterpolate SCF
            obs = scf.T
            interp_fun = fitting(train_pts, obs, method="linear")
            scf = interp_fun(train_pts).T
            print(np.mean(np.abs(obs - scf.T)**2))

            # est = self.h_bar[...,i].T + scf @ self.eigen_vect.T
            est = self.eigen_vect @ scf
            print(np.mean(np.abs(self.rirs[...,i].T - est)**2))

            scf = interp_fun(test_pts).T
            est = self.eigen_vect @ scf
            h_i.append(est.T)

        h = np.stack(h_i, axis=-1)
        return h


def fitting(pts, obs, method, norm="euclidean", params=None):

    print("PTS int", pts.shape)
    print("OBS int", obs.shape)

    if method == "linear":
        return interp.LinearNDInterpolator(pts, obs, rescale=True, fill_value=0.)

    elif method == 'rbf':

        if norm == 'haversine':
            norm = haversine_cdist

        pts_list = [pts[:,d] for d in range(pts.shape[1]) ]
        return interp.Rbf(*pts_list, obs.squeeze(), norm=norm, epsilon=0.1)
    
    elif method == 'scf':
        return SCF(pts, obs, params)
    
    else:
        raise ValueError("No method is selected")


def eval(model, pts, method):

    if method == "linear":
        return model(pts)

    elif method == 'rbf':
        pts_list = [pts[:,d] for d in range(pts.shape[1]) ]
        return model(*pts_list)
    
    elif method == 'scf':
        return model(pts)
    


def main(args):

    h5_path = Path('./data/easycom/Easycom_N-1020_fs-48k_nrfft-257.h5')
    ind_mics = [1]
    data = "atfs"
    az_lim = [  0, 360] # degree
    el_lim = [-90,  90]  # degree
    feat_lim = [0, None] 
    coord_system = 'sphere'
    features = 'cplx'
    method = 'scf'
    norm = 'haversine'
    params = {
        "n_eigh" : args.n_eigh,
    }
    grid_type = args.grid_type
    ds = args.ds

    print("Input Data")
    gt_coords, gt_target = load_data(h5_path, ind_mics=ind_mics, data=data)
    print("GT data:", gt_coords.shape, gt_target.shape)

    print("Remove angles and freqs")
    gt_coords, gt_target = prune_domain(gt_coords, gt_target, az_lim, el_lim, feat_lim)
    print("GT prune:", gt_coords.shape, gt_target.shape)

    print("Consider freqs and coordinates")
    gt_coords, gt_target = expand_coordinate_dims(gt_coords, gt_target)
    print("GT exp:", gt_coords.shape, gt_target.shape)

    # Prepare traning dataset
    print("Dowsample and flatten")
    ds_coords, ds_target = downsample_angles_and_flatten(gt_coords, gt_target, ds=ds, grid_type=grid_type)
    print("DS:", ds_coords.shape, ds_target.shape)

    print("DS exp:", ds_coords.shape, ds_target.shape)
    if coord_system == 'cart':
        gt_coords, gt_target = sph2cart_mesh_with_dims(gt_coords, gt_target)
        ds_coords, ds_target = sph2cart_mesh_with_dims(ds_coords, ds_target)

    up_obs = gt_target
    up_pts = gt_coords 
    ds_obs = ds_target
    ds_pts = ds_coords

    # change features
    if features == "abs":
        ds_obs = np.abs(ds_obs)
        up_obs = np.abs(up_obs)

    # fitting
    print('Fitting')
    start_time = time.time()
    model = fitting(ds_pts, ds_obs, 
        method=method,
        norm=norm,
        params=params
    )
    fitting_time = time.time() - start_time

    # eval on ds
    print('Val')
    eval_metrics = {}
    start_time = time.time()
    ds_est_target = eval(model, ds_pts, method)
    ds_eval_time = time.time() - start_time
    eval_metrics["val"] = eusipco_metrics(ds_est_target, ds_target, n_fft=512)
    # eval on up
    print('Test')
    start_time = time.time()
    up_est_target = eval(model, up_pts, method)
    up_eval_time = time.time() - start_time
    eval_metrics["test"] = eusipco_metrics(up_est_target, gt_target.reshape(*up_est_target.shape), n_fft=512)

    ds_est_target = ds_est_target.reshape(ds_target.shape)
    up_est_target = up_est_target.reshape(gt_target.shape)

    eval_metrics["train"] = { "time" : fitting_time }
    eval_metrics['val']["time"] = ds_eval_time
    eval_metrics['test']["time"] = up_eval_time

    print("rmse_time", eval_metrics["diff"]["rmse_time"])
    print("rmse_phase", eval_metrics["diff"]["rmse_phase"])
    print("lsd", eval_metrics["diff"]["lsd"])
    print("coherence", eval_metrics["diff"]["coherence"])


    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    if grid_type == "regular":
        plt.subplot(121)
        plt.title(f'LSD Error on UP {eval_metrics["test"]["lsd"]:.4f}')
        plt.imshow(np.abs(gt_target[:,9,:].squeeze()).T, aspect=1)
        plt.subplot(122)
        plt.title(f'LSD Error on UP {eval_metrics["test"]["lsd"]:.4f}')
        plt.imshow(np.abs(up_est_target[:,9,:].squeeze()).T, aspect=1)
        plt.show()

        fun = lambda x : np.real(np.fft.fft(x, axis=-1, n=512))[:,:512]
        plt.subplot(121)
        plt.title(f'LSD Error on UP {eval_metrics["test"]["lsd"]:.4f}')
        plt.imshow(fun(gt_target[:,9,:].squeeze()).T, aspect=0.3, origin='lower')
        plt.subplot(122)
        plt.title(f'LSD Error on UP {eval_metrics["test"]["lsd"]:.4f}')
        plt.imshow(fun(up_est_target[:,9,:].squeeze()).T, aspect=0.1, origin='lower')
        plt.show()

    elif grid_type == "random":
        plt.subplot(121)
        plt.title(f'LSD Error on UP {eval_metrics["test"]["lsd"]:.4f}')
        plt.imshow(np.abs(gt_target[:,9,:].squeeze()).T, aspect=1)
        plt.subplot(122)
        plt.title(f'LSD Error on UP {eval_metrics["test"]["lsd"]:.4f}')
        plt.imshow(np.abs(up_est_target[:,9,:].squeeze()).T, aspect=1)

    save_to = Path(exp_dir, 'linear_interpolation.png')
    plt.savefig(save_to)
    plt.show()
    plt.close()


    conf = vars(args)

    results_dict = {
        "conf" : conf,
        "metrics" : eval_metrics   
    }

    save_to = Path(exp_dir, 'linear_interpolation.pkl')
    with open(save_to, 'wb') as f:
        pickle.dump(results_dict, f) 


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)