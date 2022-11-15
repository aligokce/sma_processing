import numpy as np
from scipy.special import sph_harm

from ..utils import healpix_angles, sph_harm_healpix


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
    sum = np.zeros(Anm[0].shape, dtype=complex)
    for n in range(N + 1):
        for m in range(-n, n+1):
            sum += Anm[n + m + n*n] * sph_harm(m, n, phi, theta)  # inverse convention 
    return sum


def srf_healpix(N, Anm, npix, optimize=True, lonlat=False):
    '''
    Calculate steered response functional (SRF) on all HEALPix angles

    Parameters
    ----------
    N: Decomposition order for spherical harmonic decomposition (SHD)
    Anm: Normalised SHD coefficients for each TF-bin
    npix: Number of pixels for HEALPix
    '''
    S = sph_harm_healpix(npix, N, lonlat)
    return np.einsum('ntf, pn -> tfp', Anm, S, optimize=optimize)
