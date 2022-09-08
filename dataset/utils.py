import struct
import wave
from os import listdir

import numpy as np
from scipy import signal as sp


def emulate_scene(insig, gain, irspath):
    """
    Emulates a scene by convolving a source input signal with em32 AIRs

    :param insig: (Single-channel) input signal
    :param gain: Gain (scalar)
    :param irspath: Path to the AIRs to be used
    :return: 32 channels of audio from an emulated em32 recording.
    """
    dr = listdir(irspath)
    dr = sorted(dr)
    wv = wavread(irspath + '/' + dr[0])
    ir = np.zeros((32, wv[0].shape[0]))
    for ind in range(32):
        wv = wavread(irspath + '/' + dr[ind])
        ir[ind, :] += wv[0].reshape((wv[0].shape[0]))

    sz = len(insig)
    out = np.zeros((32, sz))
    for ind in range(32):
        out[ind, :] = sp.fftconvolve(gain * insig, ir[ind, :], mode='same')
    return out


def wavread(wave_file):
    """
    Returns the contents of a wave file

    :param wave_file: Path to the wave_file to be read
    :return: (signal, sampling rate, number of channels)

    NOTE: Wavread solution was adapted from https://bit.ly/2Ubs9Jp
    """

    w = wave.open(wave_file)
    astr = w.readframes(w.getnframes())
    nchan = w.getnchannels()
    totsm = w.getnframes()
    sig = np.zeros((nchan, totsm))
    a = struct.unpack("%ih" % (w.getnframes() * w.getnchannels()), astr)
    a = [float(val) / pow(2, 15) for val in a]
    for ind in range(nchan):
        b = a[ind::nchan]
        sig[ind] = b

    sig = np.transpose(sig)
    fs = w.getframerate()
    w.close()
    return sig, fs, nchan
