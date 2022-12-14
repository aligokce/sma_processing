import numpy as np

# Ignore warnings related to metadata blocks
import warnings
from scipy.io import wavfile

warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)


def wavread(path):
    """
    Read WAVE files into an ndarray

    Parameters
    ----------
    path: str, PathLike or open file handle
        Path to the impulse response wav file.

    Returns
    -------
    sig: ndarray
        Data read from WAVE file.
    fs: int
        Sample rate of the WAVE file.
    """
    from scipy.io import wavfile

    fs, sig = wavfile.read(path)

    if sig.dtype == 'int16':
        nb_bits = 16  # -> 16-bit wav files
    elif sig.dtype == 'int32':
        nb_bits = 32  # -> 32-bit wav files
    max_nb_bit = float(2 ** (nb_bits - 1))

    sig = sig / (max_nb_bit + 1)
    return sig, fs

