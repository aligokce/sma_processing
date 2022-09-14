import numpy as np
from scipy.special import sph_harm

from ..utils import healpix_angles


def srf(N, dir, Anm):
    '''
    Calculate steered response functional (SRF) on an angular direction given normalised 
    N-th order SHD coefficients

    Parameters
    ----------
    N: Decomposition order for spherical harmonic decomposition (SHD)
    dir: Angular direction as (colatitude, longtitude)
    Anm: Normalised SHD coefficients for each TF-bin
    '''
    theta, phi = dir  # colatitude, longtitude
    sum = np.zeros(Anm[0].shape) * 1j
    for n in range(N + 1):
        for m in range(-n, n+1):
            # TODO: NOT SURE ABOUT THIS
            sum += Anm[n + m + n*n] * sph_harm(m, n, phi, theta)  # inverse convention 
    return sum


def srf_healpix(N, Anm, npix):
    '''
    Calculate steered response functional (SRF) on all HEALPix angles

    Parameters
    ----------
    N: Decomposition order for spherical harmonic decomposition (SHD)
    Anm: Normalised SHD coefficients for each TF-bin
    npix: Number of pixels for HEALPix
    '''
    gridangles = healpix_angles(npix, lonlat=False)
    return np.stack(list(srf(N, d, Anm) for d in gridangles), axis=-1)
