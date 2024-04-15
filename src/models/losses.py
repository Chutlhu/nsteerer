import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

import matplotlib.pyplot as plt
import numpy as np
import torchaudio.functional as T
from src.utils.dsp_utils import htransforms_wikipedia

import auraloss

EPS = 1E-8

def unwrap(phi, dim=-1):
    assert dim == -1, 'unwrap only supports dim=-1 for now'
    dphi = diff(phi, same_size=True)
    dphi_m = ((dphi+np.pi) % (2 * np.pi)) - np.pi
    dphi_m[(dphi_m==-np.pi)&(dphi>0)] = np.pi
    phi_adj = dphi_m-dphi
    phi_adj[dphi.abs()<np.pi] = 0
    return phi + phi_adj.cumsum(dim)


def diff(x, dim=-1, same_size=False):
    assert dim == -1, 'diff only supports dim=-1 for now'
    if same_size:
        return F.pad(x[...,1:] - x[...,:-1], (1,0))
    else:
        return x[...,1:]-x[...,:-1]


class LossATF(_Loss):
    def __init__(self, function='l2'):
        super().__init__()
        self.function = function
        if self.function == 'l2':
            self.fun = torch.nn.MSELoss()
        elif self.function == 'l1':
            self.fun = torch.nn.L1Loss()
    
    
    def forward(self, est_atfs, atfs):          
        loss = torch.mean(torch.abs(est_atfs - atfs)**2)
        return loss


class LossATF_MAG(_Loss):
    def __init__(self, function='l2', scale='lin'):
        super().__init__()
        self.function = function
        if self.function == 'l2':
            self.fun = torch.nn.MSELoss()
        elif self.function == 'l1':
            self.fun = torch.nn.L1Loss()
        self.scale = scale
    
    
    def forward(self, est_atfs, atfs, reduction='mean'):          
        if self.scale == 'log':
            # est_mag = 20 * torch.log10(torch.abs(est_atfs) + EPS)
            # mag = 20 * torch.log10(torch.abs(atfs) + EPS)
            atfs = 20 * torch.log10(torch.abs(atfs))
            est_atfs = 20 * torch.log10(torch.abs(est_atfs))
        if self.function == 'l2':
            loss = nn.functional.mse_loss(est_atfs, atfs, reduction=reduction)
        elif self.function == 'l1':
            loss = nn.functional.l1_loss(est_atfs, atfs, reduction=reduction)

        return loss


class LossATF_PHASE(_Loss):
    def __init__(self, function='l2'):
        super().__init__()
        self.function = function

    
    def forward(self, est_atfs, atfs, reduction='mean'):
        est_angles = torch.angle(est_atfs)
        angles = torch.angle(atfs)
        if self.function == 'l2':
            loss_cos = nn.functional.mse_loss(torch.cos(est_angles), torch.cos(angles), reduction=reduction)
            loss_sin = nn.functional.mse_loss(torch.sin(est_angles), torch.sin(angles), reduction=reduction)
        elif self.function == 'l1':
            loss_cos = nn.functional.l1_loss(torch.cos(est_angles), torch.cos(angles), reduction=reduction)
            loss_sin = nn.functional.l1_loss(torch.sin(est_angles), torch.sin(angles), reduction=reduction)
        return loss_cos + loss_sin



def unique_points(coords, model, atfs_freqs_ext, n_doas=None):
    doas = coords[:,:2]
    unique, idx, counts = torch.unique(doas, dim=0, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0], device=model.device), cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]
    doas = doas[first_indicies[:n_doas,...]] # n_doas x 2
    target = atfs_freqs_ext[first_indicies[:n_doas,...]] # n_doas x n_chan x n_smpl
    n_doas = doas.shape[0]
    assert target.shape[-1] == model.n_rfft
    allfreqs = torch.linspace(0, 1, model.n_rfft, device=model.device)
    allfreqs = allfreqs.repeat(n_doas)[:,None]
    doas = doas.repeat_interleave(model.n_rfft, 0)
    coords_flatten = torch.concat([doas, allfreqs], dim=-1)
    assert coords_flatten.shape[0] == target.shape[0] * target.shape[-1]
    return coords_flatten, target




class LossAIR(_Loss):
    
    def __init__(self, function, ndoas, nsmpl, n_rfft):
        super().__init__()
        
        self.n_doas = ndoas
        self.n_smpl = nsmpl

        if function == 'logcosh':
            self.loss = auraloss.time.LogCoshLoss(eps=1e-6)
        elif function == 'l2':
            self.loss = torch.nn.MSELoss()
        elif function == 'l1':
            self.loss = torch.nn.L1Loss()
        elif function == 'sisdr':
            self.loss = auraloss.time.SISDRLoss()
        self.function = function

    def forward(self, est_atfs, atfs, n_fft):
        assert est_atfs.shape[-2] == atfs.shape[-2] == n_fft // 2 + 1

        airs = torch.fft.irfft(atfs, n_fft, dim=-2).real # n_doas x n_rfft x n_chan
        est_airs = torch.fft.irfft(est_atfs, n_fft, dim=-2).real # n_doas x n_rfft x n_chan

        # if do_freqs_in:
        #     import ipdb; ipdb.set_trace()
        #     # coords, atfs = unique_points(coords, model, atfs_freqs_ext, n_doas=self.n_doas)
        #     atfs = atfs_freqs_ext
        #     airs = torch.fft.irfft(atfs, model.n_fft, dim=-2)
        #     _atfs = model(coords)[0]
        #     # IDFT
        #     _atfs = _atfs.reshape(atfs.shape[0], model.n_rfft, _atfs.shape[-1]).permute(0,2,1)
        #     _airs = torch.fft.irfft(_atfs, model.n_fft, dim=-1).real # n_chan x n_rfft
        # else:
        if self.function == 'sisdr':
            est = est_airs[...,:self.n_smpl,:].permute(0,2,1)
            ref = airs[...,:self.n_smpl,:].permute(0,2,1)
            noise = torch.randn(1,1,1000, device=est.device)
            est = T.convolve(noise, est)[0]
            ref = T.convolve(noise, ref)[0]
            val = self.loss(est, ref)
        else:
            val = self.loss(est_airs[...,:self.n_smpl,:], airs[...,:self.n_smpl,:])
        return val

def unique_coords(coords, model, n_doas):
    doas = coords[:,:2]
    unique, idx, counts = torch.unique(doas, dim=0, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0], device=model.device), cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]
    doas = doas[first_indicies[:n_doas,...]] # n_doas x 2
    allfreqs = torch.linspace(0, 1, model.n_rfft, device=model.device)
    allfreqs = allfreqs.repeat(doas.shape[0])[:,None]
    doas = doas.repeat_interleave(model.n_rfft, 0)
    coords_flatten = torch.concat([doas, allfreqs], dim=-1)
    return coords_flatten

class LossHilbertCausality(_Loss):
    def __init__(self, n_rfft, function='l2', criterion="unwrap"):
        super().__init__()
        self.function = function
        if self.function == 'l2':
            self.fun = torch.nn.MSELoss()
        elif self.function == 'l1':
            self.fun = torch.nn.L1Loss()
        self.n_rfft = n_rfft
        self.n_fft = 2 * (n_rfft - 1)
        self.criterion = criterion

    def forward(self, est_atfs, atfs=None, n_rfft=None):
        assert est_atfs.shape[-2] == n_rfft
        if not atfs is None:
            assert est_atfs.shape == atfs.shape
        
        if self.criterion == "hilbert":
            atfs_hilbert = htransforms_wikipedia(
                est_atfs.swapaxes(-2,-1).real).swapaxes(-2,-1)
            assert atfs_hilbert.shape == est_atfs.shape
            atfs_hilbert_neg_real = -1 * atfs_hilbert.real
            atfs_fourier_imag = est_atfs.imag

            loss = self.fun(atfs_fourier_imag, atfs_hilbert_neg_real)

        elif self.criterion == "slope":
            dp_dw = torch.relu(torch.diff(torch.angle(est_atfs), dim=1))
            loss = self.fun(dp_dw, torch.zeros_like(dp_dw))

        elif self.criterion == "unwrap":
            phase = unwrap(torch.angle(atfs).swapaxes(-2,-1))
            est_phase = unwrap(torch.angle(est_atfs).swapaxes(-2,-1))
            loss = self.fun(est_phase, phase)
        return loss
    
def tril_values(x, dim=-1):
    if len(x.shape) == 4:
        mask = torch.ones(x.shape[dim], x.shape[dim])
        return x[:,:,mask.triu()==1]
    else:
        raise ValueError("can deal with this shape {}".format(x.shape))
    
class LossRTF(_Loss):
    def __init__(self, function='l2'):
        super().__init__()
        self.function = function
        if self.function == 'l2':
            self.fun = torch.nn.MSELoss()
        elif self.function == 'l1':
            self.fun = torch.nn.L1Loss()

    def forward(self, est_atfs, ref_atfs):
        n_mics = est_atfs.shape[-1]
        # compute rtf
        if n_mics < 5:
            obs_rtf_mics = tril_values(est_atfs[...,:,None] / est_atfs[...,None,:]).swapaxes(-2,-1)
            ref_rtf_mics = tril_values(ref_atfs[...,:,None] / ref_atfs[...,None,:]).swapaxes(-2,-1)
            loss = self.fun(torch.angle(obs_rtf_mics), torch.angle(ref_rtf_mics))
        if n_mics > 4:
            # obs_rtf_mics = tril_values(est_atfs[...,:4,None] / est_atfs[...,None,:4]).swapaxes(-2,-1)
            # ref_rtf_mics = tril_values(ref_atfs[...,:4,None] / ref_atfs[...,None,:4]).swapaxes(-2,-1)
            # loss_mics = self.fun(unwrap(torch.angle(obs_rtf_mics)), unwrap(torch.angle(ref_rtf_mics)))
            # obs_rtf_aids = tril_values(est_atfs[...,4:,None] / est_atfs[...,None,4:]).swapaxes(-2,-1)
            # ref_rtf_aids = tril_values(ref_atfs[...,4:,None] / ref_atfs[...,None,4:]).swapaxes(-2,-1)
            # loss_aids = self.fun(unwrap(torch.angle(obs_rtf_aids)), unwrap(torch.angle(ref_rtf_aids)))
            # obs_rtf_mics = tril_values(est_atfs / est_atfs[...,None,:]).swapaxes(-2,-1)
            # ref_rtf_mics = tril_values(ref_atfs / ref_atfs[...,None,:]).swapaxes(-2,-1)
            # loss = self.fun(torch.angle(obs_rtf_mics), torch.angle(ref_rtf_mics))
            est_rtf_r = torch.angle(est_atfs / est_atfs[...,-1,None])
            ref_rtf_r = torch.angle(ref_atfs / ref_atfs[...,-1,None])
            est_rtf_l = torch.angle(est_atfs / est_atfs[...,-2,None])
            ref_rtf_l = torch.angle(ref_atfs / ref_atfs[...,-2,None])
            loss_cos_r = nn.functional.l1_loss(torch.cos(est_rtf_r), torch.cos(ref_rtf_r))
            loss_sin_r = nn.functional.l1_loss(torch.sin(est_rtf_r), torch.sin(ref_rtf_r))
            loss_cos_l = nn.functional.l1_loss(torch.cos(est_rtf_l), torch.cos(ref_rtf_l))
            loss_sin_l = nn.functional.l1_loss(torch.sin(est_rtf_l), torch.sin(ref_rtf_l))
            loss = (loss_cos_r + loss_sin_r + loss_cos_l + loss_sin_l)/4
        return loss