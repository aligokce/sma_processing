import numpy as np
from scipy.signal import fftconvolve

from utils import wavread


def emulate_scene(sig_path, gain, ir_paths):
    """
    Emulate an acoustic scene for a sound.

    This function emulates room acoustics using an anechoic signal and a list of
    impulse responses from desired locations.

    Parameters
    ----------
    sig_path : str or PathLike
        Path to the anechoic sound file.
    gain : float or int
        Gain.
    ir_paths : List[PathLike]
        List of paths to the impulse responses.

    Returns
    -------
    NDArray
        Emulation signals, in the order given in `ir_paths`.
    """

    # We expect ir_paths to be ACN channel sorted
    sig, _ = wavread(sig_path)
    irs = [wavread(f)[0] for f in ir_paths]

    out = [fftconvolve(gain * sig, ir, mode='same') for ir in irs]
    return np.stack(out, axis=0)


def compose_scene(sig_path_list, ir_paths_list, gain=1, samples=(0, 48000)):
    """
    Composes multiple room acoustics emulations, generally for emulating
    simultaneous sources in an acoustic environment.

    Parameters
    ----------
    sig_path_list : List[PathLike]
        List of paths to the anechoic sound files.
    ir_paths_list : List[List[PathLike]]
        List of list of paths to the impulse responses. Expected to be in order.
        Inner list represents the channels, and outer list the source positions.
        Each list is associated with the anechoic sound given in `sig_path_list`
        at the same index.
    gain : int, optional
        Gain, by default 1
    samples : tuple, optional
        Start and stop samples, by default (0, 48000). Mind the sampling rate.

    Returns
    -------
    NDArray
        Emulation signals, in the order given in `ir_paths_list`.
    """
    # Assertions
    assert samples[1] > samples[0]
    assert len(sig_path_list) == len(ir_paths_list), \
        f"There are {len(sig_path_list)} audio(s) but {len(ir_paths_list)} IR position(s)."

    n_channels = len(ir_paths_list[0])
    assert n_channels > 0

    # Open a blank canvas
    sgo = np.zeros((n_channels, samples[1] - samples[0]))

    for sig_path, ir_paths in zip(sig_path_list, ir_paths_list):
        sgo += emulate_scene(sig_path, gain, ir_paths)
    return sgo
