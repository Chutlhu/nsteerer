import argparse
import h5py

import librosa
import numpy as np

from tqdm import tqdm
from pathlib import Path

import src.utils.acu_utils as acu
import src.utils.geo_utils as geo

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="Full path to the easycom dataset")
parser.add_argument("--save_dir", default="./", help="Full path to save generated data")
parser.add_argument("--n_rfft",  type=int,  default=1025, help="num of freq bins")
parser.add_argument("--az_extended",  type=int,  default=0, help="num of freq bins")
parser.add_argument("--out_name",  type=str,  default="Easycom")


def main(conf):
    
    data_path = Path(conf['data_dir'])
    
    # AIR        (dict) dictionary {'IR': (nSample,nDirection,nChannel),'fs': (int),'directions': (N,2),'nChan': (int)}
    AIR = {'IR': [],'fs': [],'directions': [],'nChan': [], 'azi': [], 'ele': []}
    # IR: (ndarray) Impulse Responses [nSample x nDirection x nChan]
    # fs: (int) sample rate in Hz
    # directions: (ndarray) (azimuth,elevation) in radians [nDirection x 2] 
    # nChan: (int) number of array's sensor/channel
    # azi: sorted unique azimuths (radians) [nDirection x 1]
    # ele: sorted unique elevations (radians) [nDirection x 1]
    f = h5py.File(data_path,'r')
    AIR['fs'] = int(list(f['SamplingFreq_Hz'])[0][0])
    AIR['IR'] = np.array(f['IR']) # (ndarray) [nSample x nDirection x nChan]
    AIR['ele'] = (np.pi/2)-np.array(f['Theta']) # (ndarray) elevation in radians [1 x nDirection]
    AIR['azi'] = np.array(f['Phi']) # (ndarray) azimuth in radians [1 x nDirection]
    AIR['directions'] = np.concatenate((AIR['azi'],AIR['ele']),axis=0).T # (ndarray) [nDirection x 2]
    AIR['ele'] = np.sort(np.unique(AIR['ele'])) # (ndarray) [nElevation x 1]
    AIR['azi'] = np.sort(np.unique(AIR['azi'])) # (ndarray) [nAzimuth x 1]
    AIR['nChan'] = AIR['IR'].shape[-1]
    
    
    print('IR', AIR['IR'].shape)
    print('ele', AIR['ele'].shape)
    print('azi', AIR['azi'].shape)
    print('doa', AIR['directions'].shape)
    print(np.rad2deg(AIR['directions']))
    
    n_azimuth = len(AIR['azi'])
    n_elevation = len(AIR['ele'])
    n_samples, n_direction, n_chan = np.shape(AIR['IR']) # nSamples x nDirections x nChan
    
    mic_pos = acu.get_easycom_array(arr_center=np.zeros([3,1])).mic_pos # 3 x nMic
    n_mic = mic_pos.shape[-1]
    assert n_mic == n_chan
    
    # Conf
    n_rfft = conf['n_rfft']
    nfft = 2*(n_rfft-1)
    fs = AIR['fs']

    # init    
    doas = AIR['directions'] # nDoas, 2
    airs = AIR['IR'].transpose([1,2,0]) # nDoas, nChan, nSampl

    # implement periodicity
    conf['az_extended'] = bool(conf['az_extended'])
    if conf['az_extended']:
        idx_doas_360to390 = doas[:,0] < np.deg2rad(30)
        doas_360to390 = doas[idx_doas_360to390]
        doas_360to390[:,0] += 2*np.pi
        idx_doas_m30to0 = doas[:,0] > np.deg2rad(330)
        doas_m30to0 = doas[idx_doas_m30to0]
        doas_m30to0[:,0] -= 2*np.pi
        doas = np.concatenate([doas_m30to0, doas, doas_360to390], axis=0)
        print('Extended doa', doas.shape)
        print(np.rad2deg(doas))
        n_azimuth = len(np.unique(doas[:,0]))
        airs = np.concatenate([airs[idx_doas_m30to0,:], airs, airs[idx_doas_360to390,:]], axis=0)
    n_direction, n_chan, n_samples = airs.shape # nDoas, nChan, nSampl

    atfs = np.zeros([n_direction, n_rfft, n_mic], dtype=np.complex128)
    rtfs = np.zeros([n_direction, n_rfft, n_mic, n_mic], dtype=np.complex128)

    # normalization by the average energy at the equator
    idx_doas_equator = (doas[:,1] < np.deg2rad(3)) & (doas[:,1] > np.deg2rad(-3))
    airs_equator = airs[idx_doas_equator, ...]
    mean_energy = np.sqrt(np.mean(airs_equator**2))
    airs = airs / mean_energy

    # output
    if conf['az_extended']:
        suffix = f"N-{n_direction}_fs-{fs//1000}k_nrfft-{n_rfft}_az_extended"
    else:
        suffix = f"N-{n_direction}_fs-{fs//1000}k_nrfft-{n_rfft}"
    save_dir = Path(conf["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    for n in tqdm(range(n_direction)):
        
        _airs = airs[n,:,:]
        assert _airs.shape == (n_mic, n_samples)
        
        ## Freq Steering Vector
        _atfs = np.fft.rfft(_airs, n=nfft, axis=-1).T
        assert _atfs.shape == (n_rfft, n_mic)
        atfs[n,:,:] = _atfs # n_obs x n_mic x n_rfft
        
        ## Relative Steering Vector estimation
        _rtfs = np.zeros([n_rfft, n_mic, n_mic], dtype=np.complex128)
        x = np.random.randn(fs//3)
        y = np.concatenate([np.convolve(x, _airs[i,:])[:,None] for i in range(n_mic)], axis=-1)
        Y = librosa.stft(y.T, n_fft=nfft, center=False)
        for ref_mic in range(n_mic):
            _rtfs[:,ref_mic,:] = np.mean(Y / Y[ref_mic,:,:][None,:,:], axis=-1).T
        assert np.allclose(np.abs(np.diag(np.mean(_rtfs, axis=0))), np.ones(n_mic))
        rtfs[n,:,:,:] = _rtfs
      
    assert doas.shape == (n_direction, 2)
    assert airs.shape == (n_direction, n_chan, n_samples)
    assert atfs.shape == (n_direction, n_rfft, n_chan)
    assert rtfs.shape == (n_direction, n_rfft, n_chan, n_chan)
        
    hf = h5py.File(save_dir / Path(f'{conf["out_name"]}_{suffix}.h5'), 'w')
    hf.create_dataset('doas',    data=doas.reshape(n_azimuth,n_elevation,2))
    hf.create_dataset('airs',    data=airs.reshape(n_azimuth,n_elevation,n_mic,n_samples))
    hf.create_dataset('atfs',    data=atfs.reshape(n_azimuth,n_elevation,n_rfft,n_mic))
    hf.create_dataset('rtfs',    data=rtfs.reshape(n_azimuth,n_elevation,n_rfft,n_mic,n_mic))
    hf.create_dataset('fs',      data=np.array(fs))
    hf.create_dataset('az',      data=AIR['azi'])
    hf.create_dataset('el',      data=AIR['ele'])
    hf.create_dataset('mic_pos', data=mic_pos)
    hf.close()

if __name__ == "__main__":
    from pprint import pprint

    arg_dic = vars(parser.parse_args())
    pprint(arg_dic)
    main(arg_dic)
