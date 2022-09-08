from pathlib import Path

import numpy as np
from dataset import smir_datasets
from features import dpd
from utils.save import save_ratio_list

from .shd import extract_shd


def extract_dpd_ratios(
    audio_file: str,
    position: list or tuple,
    # audio_folder: str = None,
    # smir_folder: str = None,
    # smir_room: str = None,
    # samples: tuple = (0, 192000),
    fs: int = 48000,
    n_fft: int = 1024,
    # n_channels: int = 32,
    # n_shd: int = 4,
    fl: float = 2608.0,
    fh: float = 5126.0,
    # olap: int = 4,
    j_nu: int = 25,
    j_tau: int = 4,
    n_spcorr: int = 4,
    # mic_name: str = 'em32',
    smir_name: str = 'spargair',
    save=False,
    load_from_path=None,
    **kwargs
):
    SMIRDataset = smir_datasets[smir_name]

    ''' Params
    '''
    fimin = int(round(fl / fs * n_fft))
    fimax = int(round(fh / fs * n_fft))

    ''' Extract
    '''
    if load_from_path:
        p = Path(load_from_path)
        assert p.isdir(), f"Given path is not a directory: {str(p)}"

        load_path = p / SMIRDataset._pos_dir(position) / audio_file
        if load_path.exists():
            Anm = np.load(load_path.with_suffix('.npy'))
        else:
            print("File", load_path, "is not available, passing...")
            return
    else:
        Anm = extract_shd(audio_file, position, fs=fs, n_fft=n_fft,
                          fl=fl, fh=fh, j_nu=j_nu, smir_name=smir_name, **kwargs)
    
    ratio_list = dpd.generate_ratios_list(Anm, n_spcorr, fimin, fimax, j_tau, j_nu)

    ''' Saves
    '''
    if save:
        save_path = Path.cwd() / SMIRDataset._pos_dir(position) / audio_file
        save_ratio_list(
            ratio_list, save_path.with_suffix('.npy'), verbose=True)

    return ratio_list
