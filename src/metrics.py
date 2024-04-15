import torch
import numpy as np
from scipy.spatial.distance import cosine
from skimage.restoration import unwrap_phase
from asteroid.metrics import get_metrics
import auraloss

EPS = 1e-15

def spectral_distortion(pred, gt, masks=None, scale="linear", xp=torch):
    """
    As implemented by https://github.com/yzyouzhang/hrtf_field/blob/main/train.py
    """
    if scale in ["linear", "lin"]:
        lsd_elements = xp.square(gt - pred)
    elif scale == "log":
        lsd_elements = xp.square(20 * xp.log10(xp.abs(gt) / xp.abs(pred)))
    else:
        raise ValueError("Either log or linear scale")
    if masks is not None:
        square_sum = (lsd_elements * masks).sum()
        mask_sum = masks.sum()
        lsd = square_sum / mask_sum
        return xp.sqrt(lsd), square_sum.item(), mask_sum.item()
    return xp.sqrt(lsd_elements.mean())


def stack_unwrap(x, seed):
    return np.stack([unwrap_phase(x[i,...], seed=seed) for i in range(x.shape[0])], axis=0)

def lsd(est, ref, do_freq=False):
    return rmse(est,ref,compute_log=True, do_freq=do_freq)

def rmse(est, ref, compute_log=False, do_freq=False):

    if compute_log:
        est = 20 * np.log10(np.abs(est) + EPS)
        ref = 20 * np.log10(np.abs(ref) + EPS)
    if do_freq:
        return np.sqrt(np.mean(np.square(np.abs(ref - est)), axis=tuple(range(0, est.ndim-1))))
    else:
        return np.sqrt(np.mean(np.square(np.abs(ref - est))))

def rmse_phase(est, ref, unwrap=False, seed=666, do_freq=False):
    est_phase = np.angle(est)
    ref_phase = np.angle(ref)
    if unwrap:
        est_phase = stack_unwrap(est_phase, seed)
        ref_phase = stack_unwrap(ref_phase, seed)
    
    return rmse(est_phase, ref_phase, do_freq)

def bss_metrics(est, ref, fs):
    clean = np.random.randn(1, int(1*fs))
    ref = np.convolve(clean, ref)
    est  = np.convolve(clean, est)
    metrics_dict = get_metrics(ref, ref, est, fs=fs, metrics_lis="all")
    return metrics_dict

def cosine_distance(est, ref):
    est = torch.from_numpy(est)
    ref = torch.from_numpy(ref)
    cos_sim = 1 - torch.nn.functional.cosine_similarity(est, ref, dim=1)
    return torch.mean(cos_sim).detach().cpu().item()


def eusipco_metrics(est, ref, n_fft=None, dim=-1):
    # assert len(est.shape) == 3r
    try:
        assert est.shape[dim] == n_fft//2 + 1
    except:
        print(f"Found {est.shape[dim]} vs  {n_fft//2 + 1}")
    
    sisdr = auraloss.time.SISDRLoss()
    
    res_dict = {
        "rmse_phase_freq" : rmse_phase(est, ref, do_freq=True),
        "lsd_freq" :   lsd(est, ref, do_freq=True),
        "rmse_phase" : rmse_phase(est, ref),
        "rmse_time"  : rmse(np.fft.irfft(est, n=n_fft, axis=dim), 
                            np.fft.irfft(ref, n=n_fft, axis=dim)),
        "lsd"        : lsd(est, ref),
        "coherence"  : cosine_distance(
                        np.fft.irfft(est, n=n_fft, axis=dim), 
                        np.fft.irfft(ref, n=n_fft, axis=dim)),
        "sisdr" : sisdr(torch.from_numpy(np.fft.irfft(est, n=n_fft, axis=dim)),
                        torch.from_numpy(np.fft.irfft(ref, n=n_fft, axis=dim)),
                        ).item()
    }
    return res_dict


if __name__ == "__main__":
    mix = np.random.randn(1, 16000)
    clean = np.random.randn(1, 16000)
    est = np.random.randn(1, 16000)
    metrics_dict = get_metrics(clean, clean, est, sample_rate=8000,
                               metrics_list='all')
    from pprint import pprint
    pprint(metrics_dict)