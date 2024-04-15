import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np

from skimage.restoration import unwrap_phase
from pathlib import Path
from pprint import pprint

from src.models.losses import *
from src.models.implicits import *
from src.metrics import *
from src.utils.dsp_utils import htransforms_wikipedia
from src.utils.torch_utils import keras_decay, MipLRDecay
# from torchlars import LARS
# from pl_bolts.optimizers.lars import LARS

def _to_np(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

EPS = 1e-8


class fMFN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_num, hidden_dim, feature_scales=[128.]):
        super(fMFN, self).__init__()

        filter_fun = FourierFilter
            
        quantization_interval = 2 * np.pi
        assert len(feature_scales) == in_dim
        input_scales = [round((np.pi * freq / (hidden_num + 1))
                / quantization_interval) * quantization_interval for freq in feature_scales]
        self.K = hidden_num
        self.hidden_dim = hidden_dim
        self.filters_freq = nn.ModuleList(
            [filter_fun(1, hidden_dim * 1, input_scales[-1:]) for _ in range(self.K)])
        self.filters_ang = nn.ModuleList(
            [filter_fun(2, hidden_dim * 1, input_scales[:2]) for _ in range(self.K)])
        self.linear = nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(self.K - 1)] + [torch.nn.Linear(hidden_dim, out_dim)])
        self.linear.apply(mfn_weights_init)

    def forward(self, x):
        # Recursion - Equation 3
        zi_freq = self.filters_freq[0](x[:,-1:]).reshape(x.shape[0], self.hidden_dim, 1)  # Eq 3.a
        zi_ang = self.filters_ang[0](x[:,:2]).reshape(x.shape[0], self.hidden_dim, 1)  # Eq 3.a
        zi = torch.sum(zi_freq * zi_ang, dim=-1)
        for i in range(self.K - 1):
            zi = self.linear[i](zi) * torch.sum(
                self.filters_ang[i + 1](x[:,:2]).reshape(x.shape[0], self.hidden_dim, 1) \
                * self.filters_freq[i + 1](x[:,-1:]).reshape(x.shape[0], self.hidden_dim, 1),
                dim=-1)  # Eq 3.b

        x = self.linear[-1](zi)  # Eq 3.c
        return x


class MFN_PHASE(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=256, out_dim=1, hidden_num=4, 
                 feature_scales=128.):
        super(MFN_PHASE, self).__init__()
        
        self.out_dim = out_dim
        self.model = MFN(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=3*out_dim, 
                         hidden_num=hidden_num, feature_scales=feature_scales)
        
    def forward(self, x):
        x = self.model(x)
        x_mag, x_arg_x, x_arg_y = x.split(self.out_dim, dim=-1)
        x_arg = torch.atan2(x_arg_y, x_arg_x)
        return x_mag, x_arg 


class MFN_PHASE_CASCADE(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=256, out_dim=1, hidden_num=4, 
                 feature_scales=128.):
        super(MFN_PHASE_CASCADE, self).__init__()
        
        filter_fun = FourierFilter
            
        quantization_interval = 2 * np.pi
        assert len(feature_scales) == in_dim
        input_scales = [round((np.pi * freq / (hidden_num + 1))
                / quantization_interval) * quantization_interval for freq in feature_scales]

        self.Kf = hidden_num
        self.filters = nn.ModuleList(
            [filter_fun(in_dim, hidden_dim, input_scales, quantization_interval) for _ in range(self.Kf)])
        self.linear_mag = nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(self.Kf - 1)] + [torch.nn.Linear(hidden_dim, out_dim)])

        self.linear_mag_match = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear_mag.apply(mfn_weights_init)
        self.linear_mag_match.apply(mfn_weights_init)

        self.Kp = hidden_num // 2 + 1 
        self.linear_arg = nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(self.Kp - 1)] + [torch.nn.Linear(hidden_dim, 2*out_dim)])
        self.linear_arg.apply(mfn_weights_init)
        self.out_dim = out_dim

    def forward(self, x):

        # Recursion - Equation 3
        zi_mag = self.filters[0](x)  # Eq 3.a
        for i in range(self.Kf - 1):
            zi_mag = self.linear_mag[i](zi_mag) * self.filters[i + 1](x)  # Eq 3.b

        x_mag = self.linear_mag[-1](zi_mag)  # Eq 3.c

        zi_arg = self.linear_mag_match(zi_mag) * self.filters[0](x) 
        for i in range(self.Kp - 1):
            zi_arg = self.linear_arg[i](zi_arg) * self.filters[i + 1](x)

        x_arg = self.linear_arg[-1](zi_arg)
        x_arg_x, x_arg_y = x_arg.split(self.out_dim, dim=-1)
        x_arg = torch.atan2(x_arg_y, x_arg_x)
        return x_mag, x_arg 


class fSIREN(nn.Module):
    def __init__(self, 
            in_dim=2, out_dim=1, hidden_dim=256, hidden_num=4, 
            feature_scales=128., do_skip=False, K=3):
        super(fSIREN, self).__init__()

        self.K = K
        self.out_dim = out_dim
        self.hidden_num = hidden_num
        self.model_ang = SIREN(
            in_dim-1, out_dim=out_dim * self.K, hidden_dim=hidden_dim, hidden_num=hidden_num,
            feature_scales=feature_scales[:-1], do_skip=do_skip)
        self.hidden_dim = hidden_dim
        self.model_freq = SIREN(
            1, out_dim=out_dim * self.K, hidden_dim=hidden_dim, hidden_num=hidden_num,
            feature_scales=feature_scales[-1:], do_skip=do_skip)

    def forward(self, x):
        varin_shape = x.shape[:-1]
        w_freq = self.model_freq(x[...,-1:]).reshape(*varin_shape, self.out_dim, self.K)
        w_ang =  self.model_ang(x[...,:2]).reshape(*varin_shape, self.out_dim, self.K)
        x = torch.sum(w_freq * w_ang, dim=-1)
        return x


class fSIREN_PHASE(nn.Module):
    def __init__(self, 
            in_dim=2, out_dim=1, hidden_dim=256, hidden_num=4, 
            feature_scales=128., do_skip=False, K=1):
        super(fSIREN_PHASE, self).__init__()

        self.K = K
        self.out_dim = out_dim
        self.hidden_num = hidden_num
        self.model_ang = SIREN(
            in_dim-1, out_dim=out_dim * self.K * 3, hidden_dim=hidden_dim, hidden_num=hidden_num,
            feature_scales=feature_scales[:-1], do_skip=do_skip)
        self.hidden_dim = hidden_dim
        self.model_freq = SIREN(
            1, out_dim=out_dim * self.K, hidden_dim=hidden_dim, hidden_num=hidden_num,
            feature_scales=feature_scales[-1:], do_skip=do_skip)

    def forward(self, x):
        varin_shape = x.shape[:-1]
        w_freq = self.model_freq(x[...,-1:]).reshape(*varin_shape, self.out_dim, self.K)
        w_ang =  self.model_ang(x[...,:2]).reshape(*varin_shape, self.out_dim, self.K, 3)
        x_mag = torch.sum(w_freq * w_ang[...,0], dim=-1)
        w_ang = torch.atan2(w_ang[...,1], w_ang[...,2])
        x_arg = torch.sum(w_freq * w_ang, dim=-1)
        return x_mag, x_arg


class fSIREN_PHASE_CASCADE(nn.Module):

    def __init__(self, 
            in_dim=2, out_dim=1, hidden_dim=256, hidden_num=4, 
            feature_scales=128., do_skip=False, K=1):
        super(fSIREN_PHASE_CASCADE, self).__init__()
        
        self.K = K
        self.out_dim = out_dim
        self.hidden_num = hidden_num
        self.model_ang = SIREN(
            in_dim-1, out_dim=out_dim * self.K, hidden_dim=hidden_dim, hidden_num=hidden_num,
            feature_scales=feature_scales[:-1], do_skip=do_skip)
        self.hidden_dim = hidden_dim
        self.model_freq = SIREN(
            1, out_dim=out_dim * self.K, hidden_dim=hidden_dim, hidden_num=hidden_num,
            feature_scales=feature_scales[-1:], do_skip=do_skip)
        self.model_arg = SIREN(
            in_dim -1 + out_dim, out_dim=out_dim * self.K * 2, hidden_dim=hidden_dim, hidden_num=max(hidden_num//2,1),
            feature_scales=feature_scales[:-1] + out_dim * [1], do_skip=do_skip)

    def forward(self, x):
        varin_shape = x.shape[:-1]
        w_freq = torch.tanh(self.model_freq(x[...,-1:]).reshape(*varin_shape, self.out_dim, self.K))
        w_ang =  self.model_ang(x[...,:2]).reshape(*varin_shape, self.out_dim, self.K)
        x_mag = torch.sum(w_freq * w_ang, dim=-1)

        w_arg = self.model_arg(torch.cat((x[...,:2], torch.tanh(x_mag)), dim=-1)).reshape(*varin_shape, self.out_dim, self.K, 2)
        x_arg = torch.atan2(w_arg[...,0], w_arg[...,1])
        x_arg = torch.sum(w_freq * x_arg, dim=-1)
        return x_mag, x_arg


class fSIREN_PHASE_CASCADE_MAG(nn.Module):

    def __init__(self, 
            in_dim=2, out_dim=1, hidden_dim=256, hidden_num=4, 
            feature_scales=128., do_skip=False, K=8):
        super(fSIREN_PHASE_CASCADE_MAG, self).__init__()
        
        self.K = K
        self.out_dim = out_dim
        self.hidden_num = hidden_num
        self.model_mag = SIREN(
            in_dim -1 + out_dim, out_dim=out_dim * self.K, hidden_dim=hidden_dim, hidden_num=hidden_num,
            feature_scales=feature_scales[:-1], do_skip=do_skip)
        
        self.hidden_dim = hidden_dim
        self.model_freq = SIREN(
            1, out_dim=out_dim * self.K, hidden_dim=hidden_dim, hidden_num=hidden_num,
            feature_scales=feature_scales[-1:], do_skip=do_skip)
        self.model_arg = SIREN(
            in_dim - 1, out_dim=out_dim * self.K * 2, hidden_dim=hidden_dim, hidden_num=hidden_num,
            feature_scales=feature_scales[:-1] + out_dim * [1], do_skip=do_skip)

    def forward(self, x):
        varin_shape  = x.shape[:-1]
        w_arg = self.model_arg(x[:,:2]).reshape(*varin_shape, self.out_dim, self.K, 2)
        x_arg_y = torch.sum(w_freq * w_arg[...,0], dim=-1)
        x_arg_x = torch.sum(w_freq * w_arg[...,1], dim=-1)
        x_arg = torch.atan2(x_arg_y, x_arg_x)
        
        w_freq = torch.tanh(self.model_freq(x[:,-1:]).reshape(*varin_shape, self.out_dim, self.K))
        w_ang =  self.model_mag(torch.cat((x[:,:2], torch.tanh(x_arg)), dim=-1)).reshape(*varin_shape, self.out_dim, self.K)
        x_mag = torch.sum(w_freq * w_ang, dim=-1)

        return x_mag, x_arg


class SIREN_PHASE(nn.Module):

    def __init__(self, 
            in_dim=2, out_dim=1, hidden_dim=256, hidden_num=4, 
            feature_scales=128., do_skip=False):
        super(SIREN_PHASE, self).__init__()
        self.out_dim = out_dim
        self.model = SIREN(
            in_dim, out_dim=3*out_dim, hidden_dim=hidden_dim, hidden_num=hidden_num,
            feature_scales=feature_scales, do_skip=do_skip)
        self.out_dim = out_dim

    def forward(self, x):
        x = self.model(x)
        x_mag, x_arg_x, x_arg_y = x.split(self.out_dim, dim=-1)
        x_arg = torch.atan2(x_arg_y, x_arg_x)
        return x_mag, x_arg

class RFF_PHASE_CASCADE(nn.Module):

    def __init__(self, 
            in_dim=2, out_dim=1, hidden_dim=256, hidden_num=4, 
            feature_scales=128., do_skip=False):
        super(RFF_PHASE_CASCADE, self).__init__()
        
        self.model_mag = RFF(
            in_dim, out_dim=out_dim, hidden_dim=hidden_dim, hidden_num=hidden_num,
            hidden_act_fun='relu', feature_dim=1024, feature_scale=feature_scales, do_skip=do_skip)
        self.model_arg = RFF(
            in_dim + out_dim, out_dim=2*out_dim, hidden_dim=hidden_dim, hidden_num=max(hidden_num//2,1),
            hidden_act_fun='relu', feature_dim=1024, feature_scale=feature_scales, do_skip=do_skip)
        self.out_dim = out_dim


    def forward(self, x):
        x_mag = self.model_mag(x)
        x_arg = self.model_arg(torch.cat((x, torch.tanh(x_mag)), dim=-1))
        x_arg_x, x_arg_y = x_arg.split(self.out_dim, dim=-1)
        x_arg = torch.atan2(x_arg_y, x_arg_x)
        return x_mag, x_arg

class SSN_PHASE_CASCADE(nn.Module):

    def __init__(self, 
            in_dim=2, out_dim=1, hidden_dim=256, hidden_num=4, w=30,
            feature_scales=128., do_skip=False):
        super(SSN_PHASE_CASCADE, self).__init__()
        
        self.model_mag = SSN(
            in_dim, out_dim=out_dim, hidden_dim=hidden_dim, hidden_num=hidden_num,
            w=[w], feature_scales=feature_scales, do_skip=do_skip)
        self.model_arg = SSN(
            in_dim + out_dim, out_dim=2*out_dim, hidden_dim=hidden_dim, hidden_num=max(hidden_num//2,1),
            w=[w], feature_scales=feature_scales + out_dim * [1], do_skip=do_skip)
        self.out_dim = out_dim


    def forward(self, x):
        x_mag = self.model_mag(x)
        x_arg = self.model_arg(torch.cat((x, torch.tanh(x_mag)), dim=-1))
        x_arg_x, x_arg_y = x_arg.split(self.out_dim, dim=-1)
        x_arg = torch.atan2(x_arg_y, x_arg_x)
        return x_mag, x_arg


class SIREN_PHASE_CASCADE(nn.Module):

    def __init__(self, 
            in_dim=2, out_dim=1, hidden_dim=256, hidden_num=4, 
            feature_scales=128., do_skip=False):
        super(SIREN_PHASE_CASCADE, self).__init__()
        
        self.model_mag = SIREN(
            in_dim, out_dim=out_dim, hidden_dim=hidden_dim, hidden_num=hidden_num,
            feature_scales=feature_scales, do_skip=do_skip)
        self.model_arg = SIREN(
            in_dim + out_dim, out_dim=2*out_dim, hidden_dim=hidden_dim, hidden_num=max(hidden_num//2,1),
            feature_scales=feature_scales + out_dim * [32], do_skip=do_skip)
        self.out_dim = out_dim


    def forward(self, x):
        x_mag = self.model_mag(x)
        x_arg = self.model_arg(torch.cat((x, torch.tanh(x_mag)), dim=-1))
        x_arg_x, x_arg_y = x_arg.split(self.out_dim, dim=-1)
        x_arg = torch.atan2(x_arg_y, x_arg_x)
        return x_mag, x_arg
    

class SIREN_PHASE_CASCADE_MAG(nn.Module):

    def __init__(self, 
            in_dim=2, out_dim=1, hidden_dim=256, hidden_num=4, 
            feature_scales=128., do_skip=False):
        super(SIREN_PHASE_CASCADE_MAG, self).__init__()
        
        self.model_arg = SIREN(
            in_dim, out_dim=2*out_dim, hidden_dim=hidden_dim, hidden_num=hidden_num,
            feature_scales=feature_scales, do_skip=do_skip)
        self.model_mag = SIREN(
            in_dim + out_dim, out_dim=out_dim, hidden_dim=hidden_dim, hidden_num=min(hidden_num//2,1),
            feature_scales=feature_scales + out_dim * [1], do_skip=do_skip)
        self.out_dim = out_dim


    def forward(self, x):
        x_arg = self.model_arg(x)
        x_arg_x, x_arg_y = x_arg.split(self.out_dim, dim=-1)
        x_arg = torch.atan2(x_arg_y, x_arg_x)
        x_mag = self.model_mag(torch.cat((x, torch.tanh(x_arg)), dim=-1))
        return x_mag, x_arg


class HRTF_FIELD(pl.LightningModule):

    def __init__(self,  
                 exp_dir:str,
                 do_freqs_in:bool,
                 architecture:str,
                 in_dim:int,
                 out_dim:int,
                 hidden_num:int, 
                 hidden_dim:int,
                 do_skip:bool, 
                 feature_scale_ang:float,
                 feature_scale_freq:float,
                 freq_masking: bool,
                 freq_range: list,
                 do_svect: bool,
                 do_bar: bool,
                 do_delay: bool,
                 do_calib: bool,
                 mic_pos:list,
                 fs:float,
                 n_rfft:int,
                 loss_config_dict:dict,
                 optim_config_dict:dict,
        ):

        super(HRTF_FIELD, self).__init__()

        self.save_hyperparameters()
        self.automatic_optimization = True
        
        self.path_to_exp_dir = Path(exp_dir)
        self.path_to_save_fig = self.path_to_exp_dir / Path("figures")
        self.path_to_save_fig.mkdir(parents=True, exist_ok=True)

        self.do_freqs_in = do_freqs_in
        if not do_freqs_in:
            out_dim = (out_dim - 1) * n_rfft + 1
        self.fs = fs
        self.n_rfft = n_rfft
        self.n_fft = 2 * (n_rfft - 1)
        self.architecture = architecture
        mic_pos = np.array(mic_pos)
        assert mic_pos.shape[0] == 3
        self.n_mic = mic_pos.shape[-1]

        self.do_freq_mask = freq_masking
        self.fmin = freq_range[0]
        self.fmax = freq_range[1]
        
        self.do_svect = bool(do_svect)
        self.do_bar  = bool(do_bar)
        self.do_delay = bool(do_delay)

        if self.do_freqs_in:
            feature_scales = 2 * [feature_scale_ang] + [feature_scale_freq]
        else:
            feature_scales = 2 * [feature_scale_ang]

            
        if architecture == "SIREN":
            self.model = SIREN(
                in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, hidden_num=hidden_num, 
                feature_scales=feature_scales, do_skip=do_skip
            )
        elif architecture == "WIRE":
            self.model = WIRE(
                in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, hidden_num=hidden_num, 
                feature_scales=feature_scales, do_skip=do_skip
            )
        elif architecture == "SIREN_PHASE":
            self.model = SIREN_PHASE(
                in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, hidden_num=hidden_num, 
                feature_scales=feature_scales, do_skip=do_skip
            )
        elif architecture == "fSIREN":
            self.model = fSIREN(
                in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, hidden_num=hidden_num, 
                feature_scales=feature_scales, do_skip=do_skip
            )
        elif architecture == "fSIREN_PHASE":
            self.model = fSIREN_PHASE(
                in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, hidden_num=hidden_num, 
                feature_scales=feature_scales, do_skip=do_skip
            )
        elif architecture == "SIREN_PHASE_CASCADE":
            self.model = SIREN_PHASE_CASCADE(
                in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, hidden_num=hidden_num, 
                feature_scales=feature_scales, do_skip=do_skip
            )
        elif architecture == "RFF_PHASE_CASCADE":
            self.model = RFF_PHASE_CASCADE(
                in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, hidden_num=hidden_num, 
                feature_scales=feature_scales, do_skip=do_skip
            )
        elif architecture == "SSN_PHASE_CASCADE":
            self.model = SSN_PHASE_CASCADE(
                in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, hidden_num=hidden_num, 
                w=30, feature_scales=feature_scales, do_skip=do_skip
            ) 
        elif architecture == "SIREN_PHASE_CASCADE_MAG":
            self.model = SIREN_PHASE_CASCADE_MAG(
                in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, hidden_num=hidden_num, 
                feature_scales=feature_scales, do_skip=do_skip
            )
        elif architecture == "fSIREN_PHASE_CASCADE":
            self.model = fSIREN_PHASE_CASCADE(
                in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, hidden_num=hidden_num, 
                feature_scales=feature_scales, do_skip=do_skip)
        elif architecture == 'fMFN':
            self.model = fMFN(
                in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, hidden_num=hidden_num,
                feature_scales=feature_scales)
        elif architecture == 'MFN':
            self.model = MFN(
                in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, hidden_num=hidden_num,
                feature_scales=feature_scales)
        elif architecture == "MFN_PHASE_CASCADE":
            self.model = MFN_PHASE_CASCADE(
                in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, hidden_num=hidden_num,
                feature_scales=feature_scales)
        elif architecture == 'MFN_PHASE':
            self.model = MFN_PHASE(
                in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, hidden_num=hidden_num,
                feature_scales=feature_scales,
            )
        else:
            raise ValueError(f"Expected architecture name, found {architecture}")
        
        self.loss_conf = loss_config_dict
        self.optim_conf = optim_config_dict
        
        self.losses = {}
        if self.loss_conf["loss_mag"] > 0:
            self.losses['loss_mag'] = {
                "fun" : LossATF_MAG(
                            self.loss_conf['loss_mag_fun'],
                            self.loss_conf['loss_mag_scale']),
                "lam" : self.loss_conf['loss_mag']
            }
        if self.loss_conf["loss_phase"] > 0:
            self.losses['loss_phase'] = {
                "fun" : LossATF_PHASE(
                            self.loss_conf['loss_phase_fun']),
                "lam" : self.loss_conf['loss_phase']
            }
        if self.loss_conf["loss_atf"] > 0:
            self.losses['loss_atf'] = {
                "fun" : LossATF(
                            self.loss_conf['loss_atf_fun']),
                "lam" : self.loss_conf['loss_atf']
            }
        if self.loss_conf["loss_air"] > 0:
            self.losses['loss_air'] = {
                "fun" : LossAIR(
                            self.loss_conf['loss_air_fun'],
                            ndoas=self.loss_conf['loss_air_ndoas'],
                            nsmpl=self.loss_conf['loss_air_nsmpl'],
                            n_rfft=self.n_rfft),
                "lam" : self.loss_conf['loss_air']
            }
        if self.loss_conf["loss_causality"] > 0:
            self.losses['loss_causality'] = {
                "fun" : LossHilbertCausality(
                            self.n_rfft,
                            self.loss_conf['loss_causality_fun'],
                            self.loss_conf['loss_causality_crit']),
                "lam" : self.loss_conf['loss_causality']
            }
        if self.loss_conf["loss_rtf"] > 0:
            self.losses['loss_rtf'] = {
                "fun" : LossRTF(
                            self.loss_conf['loss_rtf_fun']),
                "lam" : self.loss_conf['loss_rtf']
            }
       
        self.loss_sisdr = LossAIR(
                            "sisdr",
                            ndoas=self.loss_conf['loss_air_ndoas'],
                            nsmpl=self.loss_conf['loss_air_nsmpl'],
                            n_rfft=self.n_rfft)

        self.mic_pos_init = nn.Parameter(torch.from_numpy(mic_pos).float(), requires_grad=False)
        self.ref_pos_init = nn.Parameter(torch.from_numpy(np.array([[-0.07,0.,0.]]).T).float(), requires_grad=False)
        self.dist_offset_sample_init = nn.Parameter(torch.Tensor([20., 24]).float(), requires_grad=False)
        
        # FREE PARAMS
        self.do_calib = do_calib
        if self.do_calib:
            self.eps_speed_of_sound = nn.Parameter(torch.Tensor([0.]), requires_grad=True).float()
            self.eps_dist_sample = nn.Parameter(torch.Tensor([0., 0.]), requires_grad=True, ).float()
            self.eps_mic_pos = nn.Parameter(0.01*(2*torch.rand_like(self.mic_pos_init)-1), requires_grad=True).float()
        else:
            self.eps_speed_of_sound = nn.Parameter(torch.Tensor([0.]), requires_grad=False).float()
            self.eps_dist_sample = nn.Parameter(torch.Tensor([0., 0.]), requires_grad=False, ).float()
            self.eps_mic_pos = nn.Parameter(torch.zeros_like(self.mic_pos_init), requires_grad=False).float()

        self.my_params_calib = ["eps_speed_of_sound", "eps_dist_sample","eps_mic_pos",]
        self.my_params_not_unfreeze = ["mic_pos_init", "ref_pos_init"]
        # For visuzalization
        self.viz_lr_set = None
        self.viz_hr_set = None

        self.is_training = False
        

    def compute_freq_mask(self, coords):
        dfreqs = coords[:,-1] # in [0,1]
        freqs = dfreqs * self.fs/2
        mask = (freqs >= self.fmin) & (freqs < self.fmax)
        return mask
    

    def forward(self, xin, stage_train=False): # x := BxC(Batch, InputChannels)

        B = xin.shape[0]

        # NORMALIZATION
        if self.do_freqs_in:
            az, el, dfreqs = xin.split(1,-1)
            # here we expect var0 in [0,2pi], var1 in [-pi,pi] and var2 in [0,1]
            _xin = xin * torch.Tensor([1/np.pi, 2/np.pi, 2]).to(self.dtype).to(self.device)[None,:]
            _xin =_xin + torch.Tensor([-1, 0, -1]).to(self.dtype).to(self.device)[None,:]
        else:
            az, el = xin.split(1,-1)
            dfreqs = torch.linspace(0, 1, self.n_rfft).to(self.device)[:,None]
            # here we expect var0 in [0,2pi], var1 in [-pi,pi] and var2 in [0,1]
            _xin = xin * torch.Tensor([1/np.pi, 2/np.pi]).to(self.dtype).to(self.device)[None,:]
            _xin =_xin + torch.Tensor([-1, 0]).to(self.dtype).to(self.device)[None,:]
       
        freqs = self.fs/2 * dfreqs


        # if self.current_epoch == 0:
        #     for n, param in self.model.named_parameters():
        #         print(n)
        #         param.requires_grad = False

        # if self.current_epoch == 40:
        #     for n, param in self.model.named_parameters():
        #         param.requires_grad = False

        #     # freeze backbone layers
        #     for n, param in self.named_parameters():
        #         if not n in self.my_params_calib:
        #             param.requires_grad = False 
        # else:
        #     for n, param in self.named_parameters():
        #         if not n in self.my_params_not_unfreeze:
        #             param.requires_grad = True

        x = self.model(_xin)

        # ARCHITECTURE
        if "_PHASE" in self.architecture:
            x_mag, x_arg = x
            if self.do_freqs_in:
                assert x_mag.shape[-1] > 1 + self.n_mic
                assert x_arg.shape[-1] == 1 + self.n_mic + 1
                atf_bar_mag_arg  = torch.stack([x_mag[...,0], x_arg[...,0]], dim=-1)[...,None,:]
                assert atf_bar_mag_arg.shape[0] == B
                assert atf_bar_mag_arg.shape[-1] == 2
                gain_dir_mag_arg = torch.stack([x_mag[...,1:1+self.n_mic], x_arg[...,1:1+self.n_mic]], dim=-1)
                assert gain_dir_mag_arg.shape[0] == B
                assert gain_dir_mag_arg.shape[-2:] == (self.n_mic,2)
            else:
                B, D = x_mag.shape
                assert x_mag.shape[-1] == (1+self.n_mic) * self.n_rfft + 1
                assert x_arg.shape[-1] == (1+self.n_mic) * self.n_rfft + 1
                atf_bar_mag = x_mag[:,:self.n_rfft].contiguous().reshape(B,self.n_rfft,1,1)
                atf_bar_arg = x_arg[:,:self.n_rfft].contiguous().reshape(B,self.n_rfft,1,1)
                atf_bar_mag_arg  = torch.concat([atf_bar_mag, atf_bar_arg], dim=-1)
                gain_dir_mag = x_mag[:,self.n_rfft:self.n_rfft+self.n_mic*self.n_rfft].contiguous().reshape(B,self.n_rfft,self.n_mic,1)
                gain_dir_arg = x_arg[:,self.n_rfft:self.n_rfft+self.n_mic*self.n_rfft].contiguous().reshape(B,self.n_rfft,self.n_mic,1)
                gain_dir_mag_arg  = torch.concat([gain_dir_mag, gain_dir_arg], dim=-1)

            # global_delay_phase = x_arg[:,-1:]
            # assert global_delay_phase.shape == (B,1)

        else:
            if self.do_freqs_in:
                B, F, D = x.shape
                assert x.shape[-1] == 2*(1+self.n_mic) + 1
                atf_bar_mag_arg = x[...,:2].contiguous().reshape(B,F,1,2)                         # Batch x 1   x Cplx
                gain_dir_mag_arg = x[...,2:2+2*self.n_mic].contiguous().reshape(B,F,self.n_mic,2) # Batch x Mic x Cplx
                # global_delay_phase = x[:,:,-1].contiguous().reshape(B,F,1).mean(dim=1)
            else:
                coord_shape = x.shape[:-1]
                assert x.shape[-1] == (2*(1+self.n_mic)) * self.n_rfft + 1
                atf_bar_mag_arg = x[:,:2*self.n_rfft].contiguous().reshape(
                    *coord_shape,self.n_rfft,1,2)
                gain_dir_mag_arg = x[:,2*self.n_rfft:2*self.n_rfft+2*self.n_mic*self.n_rfft].contiguous().reshape(
                    *coord_shape,self.n_rfft,self.n_mic,2)    
                # global_delay_phase = x[:,-1].contiguous().reshape(B,1)


        # PHYSICAL MODEL
        if self.do_bar:
            atf_bar_mag = torch.exp(gain_dir_mag_arg[...,0])
            atf_bar_phase = atf_bar_mag_arg[...,1]
        else:
            atf_bar_mag = torch.ones_like(atf_bar_mag_arg[...,0])
            atf_bar_phase = torch.zeros_like(atf_bar_mag_arg[...,1])
        atf_bar = atf_bar_mag * torch.exp(1j * atf_bar_phase) # Batch x 1
        
        gain_dir_mag = 20*torch.exp(gain_dir_mag_arg[...,0])
        gain_dir_phase = gain_dir_mag_arg[...,1]
        gain_directivity = gain_dir_mag * torch.exp(1j * gain_dir_phase) # Batch x Mic

        # compute steering vector for each sensor
        mic_pos = self.mic_pos_init + self.eps_mic_pos
        vect_mics = -(mic_pos - self.ref_pos_init)
        vect_doa = torch.concat([
            torch.cos(el)*torch.cos(az),
            torch.cos(el)*torch.sin(az),
            torch.sin(el)
        ], dim=-1)
        self.c = 343. + 5. * torch.tanh(self.eps_speed_of_sound)
        toas_far_free = (vect_doa @ vect_mics) / self.c
        if self.do_freqs_in:
            svect = torch.exp(- 1j * 2 * np.pi * freqs * toas_far_free) #  Batch x Mic
        else:
            svect = torch.exp(- 1j * 2 * np.pi * freqs.unsqueeze(0) * toas_far_free.unsqueeze(1))
        if not self.do_svect:  
            svect = torch.ones_like(svect)
        src_distance_offset_sample = self.dist_offset_sample_init + 20*torch.tanh(self.eps_dist_sample)
        global_delay_phase = src_distance_offset_sample / self.fs
        global_delay_phase = torch.concat(4*[global_delay_phase[:1]] + 2*[global_delay_phase[1:]])
        if not self.do_delay:
            global_delay_phase = torch.zeros_like(global_delay_phase)
        if self.do_freqs_in:
            gain_offset = torch.exp(- 1j * 2 * np.pi * freqs * global_delay_phase[None,None,:])
        else:
            gain_offset = torch.exp(- 1j * 2 * np.pi * freqs * global_delay_phase).unsqueeze(0)  #  Batch x 1

        # if self.current_epoch < 100 and self.is_training:
        #     atfs = gain_offset * svect * atf_bar
        # else:
        #     print('full atf')
        atfs = gain_offset * svect * gain_directivity * atf_bar
        #     self.eps_dist_sample.requires_grad = False
        #     self.eps_mic_pos.requires_grad = False
        #     self.eps_speed_of_sound.requires_grad = False
        

        extra = {
            'gain_offset' : gain_offset,
            'svect' : svect,
            'gain_directivity' : gain_directivity,
            'toas_far_free' : toas_far_free,
            'atf_bar' : atf_bar,
            'atf_bar_mag' : atf_bar_mag,
            'atf_bar_phase' : atf_bar_phase,
            'gain_dir_mag' : gain_dir_mag,
            'gain_dir_phase' : gain_dir_phase,
            'global_delay_phase' : global_delay_phase,
            'raw_output' : x
        }
        return atfs, extra
      
        
    def _common_step(self, batch, batch_idx:int, stage:str):
        self.is_training = stage == 'train'
        # It is independent of forward
        coords, atfs, target_freq_ext = batch
        # Prediction
        est_atfs, extra = self.forward(coords)
        assert est_atfs.shape == atfs.shape
        assert est_atfs.dtype == atfs.dtype

        if stage == "train":
            for n, param in self.named_parameters():
                if (param.requires_grad) and (not param.grad is None):
                    if not torch.all(torch.isfinite(param.grad)):
                        import ipdb; ipdb.set_trace()
      

        # # Reconstruction loss
        loss_dict = {}
        loss_dict['tot'] = 0
        
        for _loss_name in self.losses:
            _loss_lam = self.losses[_loss_name]["lam"]
            
            if _loss_lam == 0:
                continue

            if _loss_name == "loss_atf":
                loss_dict[_loss_name] = self.losses[_loss_name]["fun"](est_atfs, atfs)
            if _loss_name == "loss_mag":
                if self.do_freqs_in:
                    # loss_dict[_loss_name] = 0
                    # idx = torch.randint(0, est_atfs.shape[1], (coords.shape[0],))
                    # idx = torch.sort(idx)[0]
                    # _est_atfs = est_atfs[:,idx,...]
                    # _atfs = atfs[:,idx,...]
                    # eps = 1/100
                    # res_sorted = self.losses[_loss_name]["fun"](_est_atfs, _atfs, reduction='none')
                    # res_sorted = res_sorted.permute(1,2,0)
                    # B = res_sorted.shape[-1]
                    # M = torch.triu(torch.ones(B, B), diagonal=1).T.to(self.device)
                    # mat = res_sorted @ M.T
                    # W = torch.exp(- eps * mat)
                    # loss_dict[_loss_name] = torch.mean(W*res_sorted)
                    loss_dict[_loss_name] = self.losses[_loss_name]["fun"](est_atfs, atfs)
                else:
                    loss_dict[_loss_name] = self.losses[_loss_name]["fun"](est_atfs, atfs)
            if _loss_name == "loss_phase":
                if self.do_freqs_in:
                    loss_dict[_loss_name] = 0
                    idx = torch.randint(0, est_atfs.shape[1], (coords.shape[0],))
                    idx = torch.sort(idx)[0]
                    _est_atfs = est_atfs[:,idx,...]
                    _atfs = atfs[:,idx,...]
                    eps = 1/100
                    res_sorted = self.losses[_loss_name]["fun"](_est_atfs, _atfs, reduction='none')
                    res_sorted = res_sorted.permute(1,2,0)
                    B = res_sorted.shape[-1]
                    M = torch.triu(torch.ones(B, B), diagonal=1).T.to(self.device)
                    mat = res_sorted @ M.T
                    W = torch.exp(- eps * mat)
                    loss_dict[_loss_name] = torch.mean(W*res_sorted)
                    # loss_dict[_loss_name] = self.losses[_loss_name]["fun"](est_atfs, atfs)
                else:
                    loss_dict[_loss_name] = self.losses[_loss_name]["fun"](est_atfs, atfs)
                    
            if _loss_name == "loss_air":
                loss_dict[_loss_name] = self.losses[_loss_name]["fun"](est_atfs, atfs, self.n_fft)
                
            if _loss_name == "loss_rtf":
                loss_dict[_loss_name] = self.losses[_loss_name]["fun"](extra['svect'], atfs)

            if _loss_name == "loss_causality":
                if self.do_freqs_in:
                    F = est_atfs.shape[1]
                    D = int(est_atfs.shape[0]**0.5)
                    freqs = torch.rand(F, device=self.device)
                    azi = 2 * np.pi * torch.rand(D, device=self.device)
                    ele = np.pi * (2 * torch.rand(D, device=self.device) - 1)
                    varin_ = torch.stack(torch.meshgrid([azi, ele, freqs], indexing='ij'), dim=-1)
                    varin_ = varin_.reshape(D*D, F, 3)
                else:
                    D = int(est_atfs.shape[0]**0.5)
                    azi = 2 * np.pi * torch.rand(D, device=self.device)
                    ele = np.pi * (2 * torch.rand(D, device=self.device) - 1)
                    varin_ = torch.stack(torch.meshgrid([azi, ele], indexing='ij'), dim=-1)
                    varin_ = varin_.reshape(D*D, 2)
                
                est_atfs_, extra_ = self.forward(varin_)

                    
                if self.loss_conf['loss_causality_crit'] == "unwrap":
                    loss_dict[_loss_name] = self.losses[_loss_name]["fun"](est_atfs, atfs, self.n_rfft)
                else:
                    gain_caus = self.losses[_loss_name]["fun"](extra_['gain_directivity'], None, self.n_rfft)
                    if self.do_bar:                
                        bar_caus = self.losses[_loss_name]["fun"](extra_['atf_bar'], None, self.n_rfft)
                        loss_dict[_loss_name] = (gain_caus + bar_caus)/2
                    else:
                        loss_dict[_loss_name] =  gain_caus
                        
                
            loss_dict['tot'] +=  _loss_lam * loss_dict[_loss_name]

        
            self.log(f'{stage}/{_loss_name}',  loss_dict[_loss_name], on_epoch=True)
        

        #   LOGs, PRINTs and PLOTs
        self.log(f'{stage}/loss', loss_dict['tot'], on_epoch=True, sync_dist=True)

        loss_atfs = torch.sqrt(torch.mean(torch.square(torch.abs(est_atfs - atfs))))
        self.log(f'{stage}/rmse_freq', loss_atfs, on_epoch=True, sync_dist=True)
        
        loss_sisdr = self.loss_sisdr(est_atfs, atfs, self.n_fft)
        self.log(f'{stage}/sisdr', loss_sisdr, on_epoch=True, sync_dist=True)
        
        if stage == 'val' and batch_idx == 0:
            for _loss_name in loss_dict.keys():
                print(_loss_name, loss_dict[_loss_name])
            print('loss_sisdr', loss_sisdr)
        
        try:
            res_dict = eusipco_metrics(
                _to_np(est_atfs), 
                _to_np(atfs),
                n_fft=self.n_fft,
                dim=1
            )
        except Exception as e:
            print(e)
            import ipdb; ipdb.set_trace()


        self.log(f'{stage}/lsd', res_dict["lsd"], on_epoch=True, sync_dist=True)
        self.log(f'{stage}/rmse_phase', res_dict["rmse_phase"], on_epoch=True, sync_dist=True)
        self.log(f'{stage}/rmse_time',  res_dict["rmse_time"],  on_epoch=True, sync_dist=True)

        return loss_dict['tot']


    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')
    

    def validation_step(self, batch, batch_idx):
        # torch.set_grad_enabled(True) # needed for divergenge
        return self._common_step(batch, batch_idx, 'val')


    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        if batch_idx > 0:
            return
        
        src_distance_offset_sample = self.dist_offset_sample_init + 20*torch.tanh(self.eps_dist_sample)

        print('Free params')
        print(f'.... speed eps.......: {_to_np(self.eps_speed_of_sound[0]):1.2f}')
        print(f'.... mic calib eps...: {_to_np(self.eps_mic_pos.norm(dim=0))}')
        print(f'.... dst sampl eps...: {_to_np(src_distance_offset_sample)}')

        lightning_optimizer = self.optimizers()  # self = your model
        for param_group in lightning_optimizer.optimizer.param_groups:
            print(".... LR....:", param_group['lr'])

        if not self.viz_hr_coords is None:
            print('HERE SR!')
            coords = self.viz_hr_coords
            if not self.do_freqs_in:
                assert len(coords.shape) == 4
                coords = coords[:,:,0,:2].reshape(-1,2)
            else:
                coords = coords.reshape(-1, *coords.shape[-2:])
            coords = torch.from_numpy(coords).float().to(self.device)

            atfs = self.viz_hr_target
            n_az, n_el, n_rfft, n_mic = atfs.shape
            assert n_rfft > n_mic 
            atfs = torch.view_as_complex(torch.view_as_real(torch.from_numpy(atfs)).float()).to(self.device)
            
            ## FORWARD
            _atfs, extra = self.forward(coords)
            
            toas_far_free = extra['toas_far_free']
            if self.do_freqs_in:
                toas_far_free = toas_far_free.reshape(n_az, n_el, n_rfft, n_mic)
            else:
                toas_far_free = toas_far_free.reshape(n_az, n_el, n_mic)

            nfft = 2 * (n_rfft - 1)
            _atfs = _atfs.reshape(atfs.shape)
            assert _atfs.shape[-2] == n_rfft
            _airs = torch.fft.irfft(_atfs, n=nfft, dim=-2)[:,:,:n_rfft,:]
            assert _airs.shape == (n_az, n_el, n_rfft, n_mic)
            airs = torch.fft.irfft(atfs, n=nfft, dim=-2)[:,:,:n_rfft,:]

            assert airs.shape == _airs.shape
            airs = _to_np(airs)
            atfs = _to_np(atfs)
            coords = _to_np(coords)
            _atfs = _to_np(_atfs)
            _airs = _to_np(_airs)
            toas_far_free = _to_np(toas_far_free)

            if self.do_freqs_in:
                coords = coords.reshape(n_az, n_el, n_rfft, 3)
            else:
                coords = coords.reshape(n_az, n_el, 2)

            el_deg = 0
            idx_equator_coords = coords[...,1] == np.deg2rad(el_deg)
            if self.do_freqs_in:
                coords = coords[idx_equator_coords].reshape(n_az, n_rfft, 3)
                azimuths = coords[:,0,0]
                toas_far_free  = toas_far_free[idx_equator_coords].reshape(n_az, n_rfft, n_mic)[:,0,:]
            else:
                coords = coords[idx_equator_coords].reshape(n_az, 2)
                azimuths = coords[:,0]
                toas_far_free  = toas_far_free[idx_equator_coords].reshape(n_az, n_mic)

            atfs = atfs[idx_equator_coords].reshape(n_az, n_rfft, n_mic)
            _atfs = _atfs[idx_equator_coords].reshape(n_az, n_rfft, n_mic)
            airs = airs[idx_equator_coords].reshape(n_az, n_rfft, n_mic)
            _airs = _airs[idx_equator_coords].reshape(n_az, n_rfft, n_mic)


            """ PLOTs """
            widths  = [1]*(2 * 3)
            heights = [1]* n_mic
            gs_kw = dict(width_ratios=widths, height_ratios=heights)
            fig, axarr = plt.subplots(
                    figsize=(12, 3 * n_mic), ncols=len(widths), nrows=len(heights),  squeeze=False, constrained_layout=True, gridspec_kw=gs_kw)        
            fig.suptitle(f"Validation -- Epoch : {self.current_epoch}")
            
            for sample_idx, col in enumerate([0,3]):
                
                for c in range(n_mic):
                
                    az_deg = np.rad2deg(azimuths[sample_idx])
                    dist = src_distance_offset_sample[0] if c < 4  else src_distance_offset_sample[1]

                    # Log Magnitude ATFs
                    axarr[c,col].plot( airs[sample_idx,:150,c], 'C0', label='True')
                    axarr[c,col].plot(_airs[sample_idx,:150,c], 'C1', label='Pred', alpha=0.7)
                    axarr[c,col].axvline(x=self.fs*toas_far_free[sample_idx, c] + _to_np(dist))
                    axarr[c,col].set_xlabel('Time (samples)')
                    axarr[c,col].set_title(f'AIR mic {c} :: az:{az_deg} el:{el_deg}')
                    axarr[c,col].legend()
                    
                    # Log Magnitude ATFs
                    # _fun = lambda x : np.real(x) 
                    _fun = lambda x : 10*np.log10(np.abs(x)) 
                    axarr[c,col+1].plot(_fun( atfs[sample_idx,:,c]), 'C0', label='True')
                    axarr[c,col+1].plot(_fun(_atfs[sample_idx,:,c]), 'C1', label='Pred')
                    axarr[c,col+1].set_xlabel('Frequencies (bins)')
                    axarr[c,col+1].set_title(f'Log10 ATF.Mag mic {c}')
                    axarr[c,col+1].legend()

                    # Log Magnitude ATFs
                    # _fun = lambda x : np.imag(x) 
                    _fun = lambda x : unwrap_phase(np.angle(x))
                    axarr[c,col+2].plot(_fun( atfs[sample_idx,:,c]), 'C0', label='True')
                    axarr[c,col+2].plot(_fun(_atfs[sample_idx,:,c]), 'C1', label='Pred')
                    axarr[c,col+2].set_ylim([-180,10])
                    axarr[c,col+2].set_xlabel('Frequencies (bins)')
                    axarr[c,col+2].set_title(f'Arg ATF mic {c}')
                    axarr[c,col+2].legend()    
            
            fig.savefig(self.path_to_save_fig / Path(f"last_atfs_mics.png"))
            plt.close() 

            fig, axarr = plt.subplots(3, n_mic, figsize=[6*n_mic, 6*6], squeeze=False)
            for i in range(n_mic):
                dist = src_distance_offset_sample[0] if i < 4  else src_distance_offset_sample[1]
                # Target AIRS
                axarr[0, i].set_title(f'Target mic {i}')
                axarr[0, i].imshow(airs[:,:100,i].T, extent=[-24,390,100,0], aspect=6)
                axarr[0, i].set_xticks(np.rad2deg(azimuths)[::12])
                axarr[0, i].set_xlim(np.rad2deg(azimuths)[0], np.rad2deg(azimuths)[-1])
                axarr[0, i].plot(np.rad2deg(azimuths), self.fs*toas_far_free[:, i] + _to_np(dist))
                axarr[0, i].axhline(y=_to_np(dist), c='k')
                axarr[0, i].axvline(x=90, c='k')
                axarr[0, i].axvline(x=180, c='k')
                axarr[0, i].axvline(x=270, c='k')

                # Estimated AIRS
                axarr[1, i].set_title(f'Estimated mic {i}')
                axarr[1, i].imshow(_airs[:,:100,i].T, extent=[-24,390,100,0], aspect=6)
                axarr[1, i].set_xticks(np.rad2deg(azimuths)[::12])
                axarr[1, i].set_xlim(np.rad2deg(azimuths)[0], np.rad2deg(azimuths)[-1])
                axarr[1, i].plot(np.rad2deg(azimuths), self.fs*toas_far_free[:, i] + _to_np(dist))
                axarr[1, i].axhline(y=_to_np(dist), c='k')
                axarr[1, i].axvline(x=90, c='k')
                axarr[1, i].axvline(x=180, c='k')
                axarr[1, i].axvline(x=270, c='k')

                # ERROR
                axarr[2, i].set_title(f'Error mic {i}')
                axarr[2, i].imshow(np.abs(airs - _airs)[:,:100,i].T, extent=[-24,390,100,0], aspect=6)
                axarr[2, i].set_xticks(np.rad2deg(azimuths)[::12])
                axarr[2, i].set_xlim(np.rad2deg(azimuths)[0], np.rad2deg(azimuths)[-1])
                axarr[2, i].axhline(y=_to_np(dist), c='k')
                axarr[2, i].axvline(x=90, c='k')
                axarr[2, i].axvline(x=180, c='k')
                axarr[2, i].axvline(x=270, c='k')
        
            # fig.savefig(self.path_to_save_fig / Path(f"spects_epoch-{self.current_epoch}.png"))
            plt.tight_layout()
            # plt.savefig(Path(f"last_airs_map.png"))
            plt.savefig(self.path_to_save_fig / Path(f"last_airs_map.png"))
            plt.close()

            fig, axarr = plt.subplots(3, n_mic, figsize=[6*n_mic, 6*6], squeeze=False)
            for i in range(n_mic):
                dist = src_distance_offset_sample[0] if i < 4  else src_distance_offset_sample[1]

                # Target AIRS
                axarr[0, i].set_title(f'Target mic {i}')
                axarr[0, i].imshow(20*np.log10(np.abs(atfs[:,:,i])).T, extent=[-30,390,n_rfft,0], aspect=3, vmin=-20, vmax=50)
                axarr[0, i].set_xticks(np.rad2deg(azimuths)[::12])
                axarr[0, i].axvline(x=90, c='k')
                axarr[0, i].axvline(x=180, c='k')
                axarr[0, i].axvline(x=270, c='k')

                # Estimated AIRS
                axarr[1, i].set_title(f'Estimated mic {i}')
                axarr[1, i].imshow(20*np.log10(np.abs(_atfs[:,:,i])).T, extent=[-30,390,n_rfft,0], aspect=3, vmin=-20, vmax=50)
                axarr[1, i].set_xticks(np.rad2deg(azimuths)[::12])
                axarr[1, i].axvline(x=90, c='k')
                axarr[1, i].axvline(x=180, c='k')
                axarr[1, i].axvline(x=270, c='k')

                # ERROR
                error = 20*np.log10(np.abs(atfs[:,:,i])).T - 20*np.log10(np.abs(_atfs[:,:,i])).T
                axarr[2, i].set_title(f'Error mic {i}')
                axarr[2, i].imshow(error, extent=[-30,390,n_rfft,0], aspect=3)
                axarr[2, i].set_xticks(np.rad2deg(azimuths)[::12])
                axarr[2, i].axhline(y=_to_np(dist), c='k')
                axarr[2, i].axvline(x=90, c='k')
                axarr[2, i].axvline(x=180, c='k')
                axarr[2, i].axvline(x=270, c='k')


            # fig.savefig(self.path_to_save_fig / Path(f"spects_epoch-{self.current_epoch}.png"))
            plt.tight_layout()
            # plt.savefig(Path(f"last_atfs_map.png"))
            plt.savefig(self.path_to_save_fig / Path(f"last_atfs_map.png"))
            plt.close()
        return

    def configure_optimizers(self):
        
        params = list(self.model.named_parameters())
        def is_first_phase(n) : return "arg" in n and ".0." in n

        lr=self.optim_conf['lr']
        grouped_parameters = [
                {"params": [p for n, p in params if is_first_phase(n)], 'lr': lr/2},
                {"params": [p for n, p in params if not is_first_phase(n)], 'lr': lr},
        ]

        optimizer = torch.optim.AdamW(
            grouped_parameters, lr=lr, weight_decay=self.optim_conf['weight_decay']
        )

        # optimizer = torch.optim.Adam(self.parameters(), 
        #                              lr=self.optim_conf['lr'], 
        #                               weight_decay=self.optim_conf['weight_decay'])
        # if self.optim_conf["lars"]:
        # optimizer = LARS(optimizer=optimizer, eps=1e-8, trust_coef=0.001)
        #     optimizer = LARS(self.parameters(),
        #                     lr=self.optim_conf['lr'], 
        #                     weight_decay=0)

        # optimizer = torch.optim.LBFGS(self.parameters(), lr=self.optim_conf["lr"])

        scheduler = None
        if self.optim_conf["half_lr"]:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20)
            # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: keras_decay(step))
            # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        #     scheduler = MipLRDecay(optimizer, lr_init=self.optim_conf['lr'], lr_final=5e-5, 
        #                             max_steps=3000, lr_delay_steps=100, lr_delay_mult=0.1)
        else:
            scheduler = None

        return [optimizer], {"scheduler": scheduler, 
                             "monitor": "train/rmse_time",
                             "strict" : True,
                             "frequency": 1,
                             "interval": "epoch"}


    # config.add_argument("--coarse_weight_decay", type=float, default=0.1)
    # config.add_argument("--lr_init", type=float, default=1e-3)
    # config.add_argument("--lr_final", type=float, default=5e-5)
    # config.add_argument("--lr_delay_steps", type=int, default=2500)
    # config.add_argument("--lr_delay_mult", type=float, default=0.1)
    # config.add_argument("--weight_decay", type=float, default=1e-5)