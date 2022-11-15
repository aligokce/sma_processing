"""
    Referenced from higrid.dpd.dpd
"""
from collections import defaultdict
from pathlib import Path

import madmom as mm
import numpy as np
from joblib import Memory
from scipy import special as spec

from ..utils import sph_jnyn


cache_dir = Path(__file__).parents[2] / ".cache"
memory = Memory(cache_dir, verbose=0)


def preprocess_input(audio, n_channels, n_fft, olap):
    """
    Preprocess the input to represent in TF domain

    :param audio: (n_channels x length) matrix containing the audio signal channels from em32
    :param n_channels: Number of channels
    :param n_fft: FFT size
    :param olap: (n_fft / olap) is the hop_size of the STFT
    :return: STFT of the microphone array recordings

    Notes:
    Uses STFT from madmom library since onset detection is also done by it. This can be replaced with any other TF
    bin selection method (e.g. direct-path dominance, RENT, soundfield directivity etc.).
    """
    assert audio.shape[0] == n_channels, f"{audio.shape=} is not consistent with given parameter value, {n_channels=}"

    def _stft(signal, ch):
        return mm.audio.stft.STFT(signal[ch, :], frame_size=n_fft, hop_size=n_fft / olap, fft_size=n_fft)

    P = np.stack([_stft(audio, ch) for ch in range(n_channels)], axis=0)
    return P


@memory.cache
def getBmat(micstruct, findmin, findmax, NFFT, Fs, Ndec):
    """
    Return the array (e.g. em32) specific response equalisation matrix, B

    :param micstruct: Dictionary containing microphone properties
    :param findmin: Index of minimum frequency (int)
    :param findmax: Index of maximum frequency (int)
    :param NFFT: FFT size
    :param Fs: Sampling rate (Hz)
    :param Ndec: SHD order
    :return: Response equalisation matrices, B, for each frequency index
    """
    ra = micstruct['radius']
    Bmat = defaultdict()
    for find in range(findmin, findmax):
        freq = float(find) * Fs / NFFT
        kra = 2 * np.pi * freq / 344.0 * ra
        jn, jnp, yn, ynp = sph_jnyn(Ndec, kra)
        # jn, jnp, yn, ynp = spec.sph_jnyn(Ndec, kra) # scipy 0.19.1
        hn = jn - 1j * yn
        hnp = jnp - 1j * ynp
        bnkra = jn - (jnp / hnp) * hn
        bval = []
        for ind in range(Ndec + 1):
            for _ in range(2 * ind + 1):
                bval.append(bnkra[ind] * 4 * np.pi * (1j) ** ind)
        Bmat[find] = np.linalg.inv(np.matrix(np.diag(bval)))
    return Bmat


def getpvec(P, tind, find):
    """
    Return a single M-channel (e.g. 32 channel for em32) time-frequency bin

    :param P: List of STFTs of each channel
    :param tind: Time index
    :param find: Frequency index
    :return: Selected time frequency bin containing N (e.g. 32) channels
    """
    return P[:, tind, find][:, np.newaxis]


def getanmval(pvec, B, Y, W):
    """
    Return the SHD for a single time-frequency bin

    :param pvec: Vector containing M-channel STFTs of a single time-frequency bin
    :param B: Response equalisation matrix
    :param Y: SHD matrix
    :param W: Cubature matrix
    :return: SHD for a single time-frequency bin; (N+1)^2 by 1
    """
    anm = B @ Y @ W @ pvec
    return anm


@memory.cache
def getWY(micstruct, Ndec):
    """
    Return the array (e.g. em32) specific cubature and the SHD matrices, W and Y.H

    :param micstruct: Dictionary containing microphone properties
    :param Ndec: SHD order to be used in the cubature
    :return: Array specific cubature and the SHD matrices, W and Y.H
    """
    thes = micstruct['thetas']
    phis = micstruct['phis']
    w = micstruct['weights']
    W = np.matrix(np.diag(w), dtype=float)
    Y = np.matrix(np.zeros((len(thes), (Ndec + 1) ** 2), dtype=complex))
    for ind in range(len(thes)):
        y = []
        for n in range(Ndec + 1):
            for m in range(-n, n + 1):
                Ynm = spec.sph_harm(m, n, phis[ind], thes[ind])
                y.append(Ynm)
        Y[ind, :] = y
    W = W / np.diag(Y * Y.H) / 2.
    return W, Y.H


def getAnm(P, mstr, Bmat, findmin, findmax, Ndec):
    """
    Return the (N+1)^2-element list containing SHDs of STFTs

    :param P: STFTs of the M channels of recordings
    :param mstr: Dict containing the microphone array properties
    :param Bmat: List of response equalisation matrices
    :param findmin: Index of minimum frequency (int)
    :param findmax: Index of maximum frequency (int)
    :param Ndec: SHD order
    :return: ndarray of numpy matrices containing the SHDs of STFTs of array channels (channel, time, frequency)
    """
    W, Y = getWY(mstr, Ndec)
    A = np.zeros(((Ndec + 1) ** 2, *P[0].shape), dtype=complex)
    for find in range(findmin, findmax):
        B = Bmat[find]
        pv = P[:, :, find]
        anm = B @ Y @ W @ pv
        A[:, :, find] = anm
    return A
