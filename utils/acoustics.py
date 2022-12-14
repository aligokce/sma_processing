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


def measure_rt60(h, fs=1, decay_db=60):
    """
    Analyze the RT60 of an impulse response.
    
    Parameters
    ----------
    h: array_like
        The impulse response.
    fs: float or int, optional
        The sampling frequency of h (default to 1, i.e., samples).
    decay_db: float or int, optional
        The decay in decibels for which we actually estimate the time. Although
        we would like to estimate the RT60, it might not be practical. Instead,
        we measure the RT20 or RT30 and extrapolate to RT60.

    Referenced from ???
    """

    h = np.array(h)
    fs = float(fs)

    # The power of the impulse response in dB
    power = h**2
    energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder

    # remove the possibly all zero tail
    i_nz = np.max(np.where(energy > 0)[0])
    energy = energy[:i_nz]
    energy_db = 10 * np.log10(energy)
    energy_db -= energy_db[0]

    # -5 dB headroom
    i_5db = np.min(np.where(-5 - energy_db > 0)[0])
    e_5db = energy_db[i_5db]
    t_5db = i_5db / fs

    # after decay
    i_decay = np.min(np.where(-5 - decay_db - energy_db > 0)[0])
    t_decay = i_decay / fs

    # compute the decay time
    decay_time = t_decay - t_5db
    est_rt60 = (60 / decay_db) * decay_time

    return est_rt60


def measure_drr(path, correction=1.45e-3):
    """
    Analyse direct-to-reverberant ratio (DRR) of an impulse response.

    Parameters
    ----------
    path: str, PathLike or open file handle
        Path to the impulse response wav file.
    correction: float
        Correction duration for the measurement, in seconds.

    Referenced from ???
    """
    ir, fs = wavread(path)

    t0 = (ir ** 2).argmax()
    corr_samp = int(correction * fs)

    drr = 10 * np.log10(
        np.trapz(ir[ max(1, t0 - corr_samp) : t0 + corr_samp ] ** 2) /
        np.trapz(ir[t0 + corr_samp + 1 :] ** 2)
    )
    return drr
