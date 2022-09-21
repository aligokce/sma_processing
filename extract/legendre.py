from pathlib import Path

import numpy as np
from dataset import smir_datasets
from features import legendre as lg
from tqdm import tqdm
from utils.save import save_lg_count

from .shd import extract_shd


def extract_legendre_count(
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

    save_path = Path.cwd() / SMIRDataset._pos_dir(position)
    if len(list(save_path.glob(f"{audio_file.split('.')[0]}*.npy"))) > 0:
        print("Passing already extracted audios...")
        return


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
        Anm = extract_shd(audio_file, position, n_shd=n_shd, smir_name=smir_name,
                          **kwargs)

    srf = lg.srf.srf_healpix(n_shd, Anm, npix)
    dictionary = lg.generate_legendre_dict_healpix(n_shd, npix)
    count = np.zeros(srf.shape[:2], dtype=int)
    rent = np.zeros(srf.shape[:2], dtype=float)

    # TODO: limit tind and find to non-zero bins only, from fL and fH params?
    for tind in tqdm(range(srf.shape[0]), desc="Extracting Legendre kernel counts..."):
        for find in range(fimin, fimax + j_nu):
            y = srf[tind, find]  # (npix, )
            result = lg.omp(dictionary, y, dtype=complex, tol=tol)
            
            count[tind, find] = len(result.active)
            rent[tind, find] = 1 - result.err[0]

    ''' Saves
    '''
    if save:
        save_path = Path.cwd() / SMIRDataset._pos_dir(position) / audio_file
        save_path_count = save_path.with_name(save_path.stem + "_count" + ".npy")
        save_lg_count(count, save_path_count, verbose=False)
        save_path_rent = save_path.with_name(save_path.stem + "_rent" + ".npy")
        save_lg_count(rent, save_path_rent, verbose=False)

    return count, rent


def job(*args, **kwargs):
    extract_legendre_count(*args, **kwargs)
