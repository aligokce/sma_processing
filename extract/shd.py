from pathlib import Path

from dataset import smir_datasets
from features import shd
from microphone import microphones
from utils.save import save_shd


def extract_shd(
    audio_file: str,
    position: list or tuple,
    audio_folder: str = None,
    smir_folder: str = None,
    smir_room: str = None,
    samples: tuple = (0, 192000),
    fs: int = 48000,
    n_fft: int = 1024,
    n_channels: int = 32,
    n_shd: int = 4,
    fl: float = 2608.0,
    fh: float = 5126.0,
    olap: int = 4,
    j_nu: int = 25,
    mic_name: str = 'em32',
    smir_name: str = 'spargair',
    save=False,
    **kwargs
):
    Microphone = microphones[mic_name]
    SMIRDataset = smir_datasets[smir_name]

    ''' Microphone
    '''
    mic = Microphone().returnAsStruct()
    fimin = int(round(fl / fs * n_fft))
    fimax = int(round(fh / fs * n_fft))

    ''' Dataset
    '''
    smir = SMIRDataset(smir_folder)
    sig = smir.compose_scene(
        file_path_list = [str(Path(audio_folder) / audio_file)],
        pos_list = [position], 
        room = smir_room,
        samples = samples)

    ''' Spherical harmonic decomposition
    '''
    P = shd.preprocess_input(sig, n_fft, olap)  # return: STFT
    Bmat = shd.getBmat(mic, fimin, fimax + j_nu, n_fft, fs, n_shd)
    # Note: (Ndec + 1)**2 = 25 for Ndec = 4
    Anm = shd.getAnm(P, mic, Bmat, fimin, fimax + j_nu, n_shd)

    ''' Saves
    '''
    if save:
        save_path = Path.cwd() / SMIRDataset.pos2dir(position) / audio_file
        save_shd(Anm, save_path.with_suffix('.npy'), verbose=True)

    return Anm


def job(*args, **kwargs):
    extract_shd(*args, **kwargs)
