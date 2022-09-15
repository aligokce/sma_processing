from pathlib import Path
from typing import Tuple

import healpy as hp
import numpy as np
import scipy
from joblib import Memory
from tqdm import tqdm

from ..utils import healpix_angles


cache_dir = Path(__file__).parents[2] / ".cache"
memory = Memory(cache_dir)


def LegendreKernel(N: int, dir1: Tuple[float, float], dir2: Tuple[float, float]) -> float:
    '''
    N-th order Legendre kernel for two angular directions
    '''
    Pn = scipy.special.legendre
    angdist = hp.rotator.angdist(dir1, dir2).item()
    return np.sum((2*n + 1)/(4*np.pi) * Pn(n)(np.cos(angdist)) for n in range(N + 1))


def LegendreKernelHealpix(N: int, dir: Tuple[float, float], npix: int) -> np.ndarray:
    '''
    N-th order Legendre kernel for an angular direction with respect to all 
    angular directions from a HEALPix grid
    '''
    gridangles = healpix_angles(npix, lonlat=False)
    return np.fromiter((LegendreKernel(N, dir, dref) for dref in gridangles), float)


@memory.cache
def generate_legendre_dict_healpix(N: int, npix: int):
    '''
    Generates a dictionary from Legendre kernels on all possible HEALPix directions
    '''
    Lr = LegendreKernelHealpix
    gridangles = healpix_angles(npix, lonlat=False)

    return [Lr(N, dir, npix) for dir in tqdm(
            gridangles, desc="Extracting Legendre kernels on HEALPix grid")]
