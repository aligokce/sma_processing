from pathlib import Path

import numpy as np

from dataset import smir_datasets
from features import legendre as lg
from utils.save import save_lg_count

from .shd import extract_shd


def extract_rent(
    audio_file: str,
    position: list or tuple,
    # audio_folder: str = None,
    # smir_folder: str = None,
    # smir_room: str = None,
    # samples: tuple = (0, 192000),
    fs: int = 48000,
    n_fft: int = 1024,
    # n_channels: int = 32,
    n_shd: int = 4,
    fl: float = 2608.0,
    fh: float = 5126.0,
    # olap: int = 4,
    j_nu: int = 25,
    # j_tau: int = 4,
    # n_spcorr: int = 4,
    # mic_name: str = 'em32',
    smir_name: str = 'spargair',
    npix: int = 192,
    tol: float = 0.1,
    save=False,
    load_from_path=None,
    **kwargs
):
    SMIRDataset = smir_datasets[smir_name]

    fimin = int(round(fl / fs * n_fft))
    fimax = int(round(fh / fs * n_fft))

    save_path = Path.cwd() / SMIRDataset.pos2dir(position)
    if len(list(save_path.glob(f"{audio_file.split('.')[0]}*.npy"))) > 0:
        print("Passing already extracted audios...")
        return


    ''' Extract
    '''
    if load_from_path:
        p = Path(load_from_path)
        assert p.isdir(), f"Given path is not a directory: {str(p)}"

        load_path = p / SMIRDataset.pos2dir(position) / audio_file
        if load_path.exists():
            Anm = np.load(load_path.with_suffix('.npy'))
        else:
            print("File", load_path, "is not available, passing...")
            return
    else:
        Anm = extract_shd(audio_file, position, 
        n_shd=n_shd, fl=fl, fh=fh, j_nu=j_nu, smir_name=smir_name, 
                          **kwargs)

    srf = lg.srf.srf_healpix(n_shd, Anm, npix)
    dictionary = lg.generate_legendre_dict_healpix(n_shd, npix)
    rent = np.zeros(srf.shape[:2], dtype=float)

    rent[:, fimin:fimax+j_nu] = lg.calculate_rent_batch(dictionary, srf[:, fimin:fimax+j_nu])


    ''' Saves
    '''
    if save:
        save_path = Path.cwd() / SMIRDataset.pos2dir(position) / audio_file
        save_path_rent = save_path.with_name(save_path.stem + "_rent" + ".npy")
        save_lg_count(rent, save_path_rent, verbose=False)

    return rent


def job(*args, **kwargs):
    extract_rent(*args, **kwargs)
