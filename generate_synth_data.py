import argparse
import h5py

import pyroomacoustics as pra
import librosa as lba
import numpy as np

from tqdm import tqdm
from pathlib import Path

import src.utils.acu_utils as acu
import src.utils.geo_utils as geo

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", default="data/simulated", help="Full path to save generated data")

def main(conf):
    
    # constants / config
    fs = conf['data']['fs']
    nrfft = conf['data']['nrfft']
    max_order = conf['data']['max_order']
    absorption = conf['data']['absorption']
    room_dim = conf['data']['room_dim']
    
    room = pra.ShoeBox(room_dim, fs=fs, max_order=max_order, absorption=absorption)
    RT60_sabine = room.rt60_theory()
    
    azimuths_deg = np.arange(start=0, stop=359, step=1)
    elevations_deg = np.array([0.001])
    distances = [2.5]

    grid_sph_coords = np.stack(np.meshgrid(distances, azimuths_deg, elevations_deg, indexing='ij'), axis=-1)
    n_dist, n_azimuth, n_elevation, _ = grid_sph_coords.shape
    _grid_sph_coords = grid_sph_coords.reshape(-1,3).T
    grid_cart_coords = geo.sph2cart(_grid_sph_coords, deg=True)
    n_dim, n_obs = _grid_sph_coords.shape

    mic_center = np.c_[[2.7,2.4,1.8]]
    mic_array = acu.get_oculus_array(
                    mic_center=mic_center)    

    # output
    suffix = f"N-{n_obs}_fs-{fs//1000}k_nrfft-{nrfft}_rt60-{RT60_sabine:1.2f}"
    save_dir = Path(conf["options"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)    
    hf = h5py.File(save_dir / Path(f'Synth_{suffix}.h5'), 'w')
    
    print('Generating all the RIRs... ')
    data = []
    max_length = 0
    mic_pos = mic_array.center + mic_array.mic_pos
    for n in tqdm(range(n_obs)):
        
        src_pos = mic_center[:,0] + grid_cart_coords[:,n]
        
        DOA = geo.DOASetup(
            _grid_sph_coords[:,n],    
            _grid_sph_coords[0,n],
            _grid_sph_coords[1,n],
            _grid_sph_coords[2,n],
            True)
        
        room = pra.ShoeBox(room_dim, fs=fs, max_order=max_order, absorption=absorption)
        room.add_source(src_pos)
        room.add_microphone_array(mic_pos)

        I = mic_array.n_mics
        J = 1

        room.compute_rir()
        rirs = room.rir
        
        for i in range(len(rirs)):
            for j in range(len(rirs[i])):
                rir = rirs[i][j]
                assert not np.allclose(rir, np.zeros_like(rir))
        
        curr_max_lentgh = max([max(room.rir[i][j].shape) for i in range(I) for j in range(J)])
        if curr_max_lentgh > max_length:
            max_length = curr_max_lentgh

        sample = {
            'DOA' : DOA,
            'razel' : DOA.razel,
            'xyz' : src_pos,
            'rirs' : room.rir,
        }
        
        data.append(sample)
        
    L = max_length
    F = nrfft
    airs = np.zeros((n_obs,I,J,L))
    atfs = np.zeros((n_obs,I,J,F,2))
    doas = np.zeros((n_obs,I,J,3))
    
    print('Computing steering vectors... ')
    for n in tqdm(range(n_obs)):
        for i in range(I):
            for j in range(J):
                
                _rir = data[n]['rirs'][i][j].squeeze()
                _razel = data[n]['razel'].squeeze()
                
                l = len(_rir)
                airs[n,i,j,:l] = _rir[:l]
                
                nfft = 2*(F-1)
                _rir_fft = np.fft.rfft(_rir, nfft)
                                
                atfs[n,i,j,:,0] = _rir_fft.real
                atfs[n,i,j,:,1] = _rir_fft.imag
                
                doas[n,i,j,:] = _razel
    
    hf.create_dataset('doas', data=doas.reshape(n_dist,n_azimuth,n_elevation,I,J,3))
    hf.create_dataset('airs', data=airs.reshape(n_dist,n_azimuth,n_elevation,I,J,L))
    hf.create_dataset('atfs', data=atfs.reshape(n_dist,n_azimuth,n_elevation,I,J,F,2))
    hf.close()
    

if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    
    with open(Path("local/conf.yml")) as f:
        def_conf = yaml.safe_load(f)
    def_conf['optional arguments'] = {}
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    pprint(arg_dic)
    main(arg_dic)
