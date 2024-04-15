import torch
import numpy as np
import resampy

def resample(x, old_fs, new_fs):
    return resampy.resample(x.T, old_fs, new_fs).T

def make_same_length(x, y, kind='max', pad=0):
    Nx = len(x)
    Ny = len(y)
    if kind == 'max':
        N = max(Nx, Ny) + pad
        xo = np.zeros(N)
        yo = np.zeros(N)
        xo[:Nx] = x
        yo[:Ny] = y
    elif kind == 'min':
        N = min(Nx, Ny)
        xo = np.zeros(N)
        yo = np.zeros(N)
        xo[:N] = x[:N]
        yo[:N] = y[:N]
    return xo, yo

def classic_deconvolution(y, x):
    '''
    Deconvolution with Classic method: X = Y/H
    '''
    if not y.shape == x.shape:
        raise ValueError('y and h should be of the same size')
    h = np.real(np.fft.ifft(np.fft.fft(y)/np.fft.fft(x)))
    return h

def wiener_deconvolution(y, x, pad=None):
    X = np.fft.fft(x, pad)
    Y = np.fft.fft(y, pad)
    Sxx = X * np.conj(X)
    Syx = Y * np.conj(X)
    H = Syx / Sxx
    return np.real(np.fft.ifft(H))

def center(x):
    return x - np.mean(x, axis=0)


def normalize(x):
    return x/np.max(np.abs(x))


def htransforms_wikipedia(data):
    N = 2*(data.shape[-1]-1)
    # Allocates memory on GPU with size/dimensions of signal
    transforms = torch.fft.fft(data, N, axis=-1)
    transforms[..., 1:N//2]          *= -1j # positive frequency
    transforms[..., (N+2)//2 + 1: N] *= +1j # negative frequency
    transforms[...,0] = 0; # DC signal
    if N % 2 == 0:
        transforms[..., N//2] = 0  # the (-1)**n term
    
    # Do IFFT on GPU
    return torch.fft.ifft(transforms)[:,:,:N//2+1]


def noise_psd(N, psd = lambda f: 1):
        X_white = np.fft.rfft(np.random.randn(N))
        S = psd(np.fft.rfftfreq(N))
        # Normalize S
        S = S / np.sqrt(np.mean(S**2))
        X_shaped = X_white * S
        return np.fft.irfft(X_shaped)

def PSDGenerator(f):
    return lambda N: noise_psd(N, f)

@PSDGenerator
def white_noise(f):
    return 1

@PSDGenerator
def blue_noise(f):
    return np.sqrt(f)

@PSDGenerator
def violet_noise(f):
    return f

@PSDGenerator
def brownian_noise(f):
    return 1/np.where(f == 0, float('inf'), f)

@PSDGenerator
def pink_noise(f):
    return 1/np.where(f == 0, float('inf'), np.sqrt(f))

@PSDGenerator
def low_noise(f):
    return 1/np.where(f == 0, float('inf'), f)