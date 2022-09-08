import numpy as np
import healpy as hp
from scipy import signal as sp, special as sp


def sph_jnyn(N, kr):
    '''
    Returns spherical Bessel functions of the first (jn) and second kind (yn) and their derivatives

    :param N: Function order
    :param kr: Argument
    :return: jn, jn', yn, yn'

    NOTE: Emulates the behaviour of sph_jnyn() in early versions of scipy (< 1.0.0).

    '''
    jn = np.zeros(N+1)
    jnp = np.zeros(N+1)
    yn = np.zeros(N+1)
    ynp = np.zeros(N+1)
    for n in range(N+1):
        jn[n] = sp.spherical_jn(n, kr)
        jnp[n] = sp.spherical_jn(n, kr, derivative=True)
        yn[n] = sp.spherical_yn(n, kr)
        ynp[n] = sp.spherical_yn(n, kr, derivative=True)
    return jn, jnp, yn, ynp


def rel_energy(y, value, valuetype='db') -> float:
    """
    Calculate energy relative to a signal's

    :param y: Signal array
    :param value: Relative value
    :param valuetype: Relative value unit/type ('db', 'ratio')
    :return: Energy relative to the input signal array
    """
    types_available = ['db', 'ratio']
    assert valuetype in types_available, f"Choose type from {types_available}"
    assert type(value) in [float, int], "Need input value as float or int"

    if valuetype == 'db':
        return np.linalg.norm(y)**2 * 10**(value*0.1)
    elif valuetype == 'ratio':
        return value * np.linalg.norm(y)**2


def healpix_angles(npix: int, lonlat=False):
    nside = hp.npix2nside(npix)
    gridangles = [hp.pix2ang(nside, i) for i in range(npix)]
    return gridangles
