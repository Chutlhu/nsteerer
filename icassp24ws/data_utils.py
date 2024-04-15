import numpy as np
import random
import torch
import os

def prune_domain(coords, target, az_lim, el_lim, feat_lim):
    idx_coords_az = (np.rad2deg(coords[:,:,0]) >= az_lim[0]) & (np.rad2deg(coords[:,:,0]) <= az_lim[1])
    idx_coords_el = (np.rad2deg(coords[:,:,1]) >= el_lim[0]) & (np.rad2deg(coords[:,:,1]) <= el_lim[1])
    n_el = np.count_nonzero(np.count_nonzero(idx_coords_az & idx_coords_el, axis=0))
    n_az = np.count_nonzero(np.count_nonzero(idx_coords_az & idx_coords_el, axis=1))
    coords = coords[idx_coords_az & idx_coords_el].reshape(n_az, n_el, 2)
    target = target[idx_coords_az & idx_coords_el].reshape(n_az, n_el, target.shape[-2], target.shape[-1])
    target = target[...,feat_lim[0]:feat_lim[1]]
    return coords, target


def fake_periodicity_ang_domain(coords, target, az_lim_left=30, az_lim_right=330):
    n_az, n_el, n_varin = coords.shape
    n_feat, n_varout = target.shape[2:]
    assert coords.shape[-1] == 2
    assert target.shape[:2] == coords.shape[:2]
    coords = coords.reshape(-1,2)
    target = target.reshape(-1, *target.shape[-2:])
    assert target.shape[:1] == coords.shape[:1]

    idx_right = coords[:,0] <= np.deg2rad(az_lim_left)
    doas_right = coords[idx_right,...]
    doas_right[:,0] += 2 * np.pi
    idx_left = coords[:,0] >= np.deg2rad(az_lim_right)
    doas_left = coords[idx_left,...]
    doas_left[:,0] -= 2 * np.pi
    coords = np.concatenate([doas_left, coords, doas_right], axis=0)
    target = np.concatenate([target[idx_left,...], target, target[idx_right,...]], axis=0)
    n_az = len(np.unique(coords[:,0]))
    coords = coords.reshape(n_az, n_el, n_varin)
    target = target.reshape(n_az, n_el, n_feat, n_varout)
    return coords, target


def downsample_angles_and_flatten(coords, target, ds, grid_type='regular', grid_pattern="checkboard"):

    if grid_type == 'regular':
        # target = target[::ds,::ds,:,:].reshape(-1, target.shape[-1])
        # coords = coords[::ds,::ds,:,:].reshape(-1, coords.shape[-1])
        if coords.shape[0] % ds == 1:
            coords = coords[:-1,...]
            target = target[:-1,...]

        if grid_pattern == "checkboard":
            ds_target = target.reshape(-1, *target.shape[-2:])
            ds_coords = coords.reshape(-1, *coords.shape[-2:])
            idx_doa = np.arange(ds_coords.shape[0])
            idx_doa_ds = np.arange(ds_coords.shape[0])[::ds]
            idx_doa_diff = np.setdiff1d(idx_doa, idx_doa_ds)
            ds_target = target.reshape(-1, *target.shape[-2:])[idx_doa_ds,...]
            ds_coords = coords.reshape(-1, *coords.shape[-2:])[idx_doa_ds,...]
            diff_target = target.reshape(-1, *target.shape[-2:])[idx_doa_diff,...]
            diff_coords = coords.reshape(-1, *coords.shape[-2:])[idx_doa_diff,...]
        elif grid_pattern == "pincheck":
            idx_az = np.arange(coords.shape[0])
            idx_el = np.arange(coords.shape[1])
            idx_az_ds = np.arange(coords.shape[0])[::ds]
            idx_el_ds = np.arange(coords.shape[1])[::ds]
            idx_az_diff = np.setdiff1d(idx_az, idx_az_ds)
            idx_el_diff = np.setdiff1d(idx_el, idx_el_ds)
            ds_target = target[np.ix_(idx_az_ds,idx_el_ds)].reshape(-1, *target.shape[-2:])
            ds_coords = coords[np.ix_(idx_az_ds,idx_el_ds)].reshape(-1, *coords.shape[-2:])
            diff_target = target[np.ix_(idx_az_diff,idx_el_diff)].reshape(-1, *target.shape[-2:])
            diff_coords = coords[np.ix_(idx_az_diff,idx_el_diff)].reshape(-1, *coords.shape[-2:])
        else:
            raise ValueError("Wrong grid pattern")
            
    elif grid_type == 'random':
        _target = target.reshape(-1, *target.shape[-2:])
        _coords = coords.reshape(-1, *coords.shape[-2:])
        assert _target.shape[:-1] == _coords.shape[:-1]
        n_obs = _coords.shape[0]
        n_ds_obs = int(np.round(n_obs * ds / 100))
        all_idx = np.arange(0, n_obs)
        np.random.shuffle(all_idx)
        idx = all_idx[:n_ds_obs]
        ds_target = _target[idx,...]
        ds_coords = _coords[idx,...]

        diff_idx = all_idx[n_ds_obs:]
        assert len(diff_idx) + len(idx) == len(all_idx)
        diff_target = _target[diff_idx,...]
        diff_coords = _coords[diff_idx,...]

    else:
        raise ValueError("Wrong grid type")

    assert ds_target.shape[-2:] == target.shape[-2:]
    assert ds_coords.shape[-2:] == coords.shape[-2:]

    return ds_coords, ds_target, diff_coords, diff_target
    # return _coords, _target, diff_coords, diff_target


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

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False