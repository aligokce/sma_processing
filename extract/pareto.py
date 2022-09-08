from pathlib import Path

from dataset import smir_datasets
from features import pareto
from utils.save import save_distrib_plot, save_config

from .dpd_ratios import extract_dpd_ratios


def extract_pareto_params(
    audio_file: str,
    position: list or tuple,
    # audio_folder: str = None,
    # smir_folder: str = None,
    # smir_room: str = None,
    # samples: tuple = (0, 192000),
    # fs: int = 48000,
    # n_fft: int = 1024,
    # n_channels: int = 32,
    # n_shd: int = 4,
    # fl: float = 2608.0,
    # fh: float = 5126.0,
    # olap: int = 4,
    # j_nu: int = 25,
    # j_tau: int = 4,
    # n_spcorr: int = 4,
    # mic_name: str = 'em32',
    smir_name: str = 'spargair',
    save=True,
    # load_from_path=None,
    **kwargs
):
    SMIRDataset = smir_datasets[smir_name]

    ''' Extract
    '''
    ratios = extract_dpd_ratios(
        audio_file, position, smir_name=smir_name, **kwargs)
    shape, location, scale = pareto.fit_ratios(ratios)

    ''' Saves
    '''
    if save:
        save_path = Path.cwd() / SMIRDataset._pos_dir(position) / audio_file
        save_distrib_plot(ratios, shape, location, scale,
                          save_path.with_suffix('.png'))
        # Save details and fitting results to a json file
        config = dict(
            sndfile=audio_file,
            pos_grid=position,
            dist_mic=SMIRDataset.get_distance(position),
            pareto_params=dict(
                shape=shape,
                scale=scale,
                location=location
            )
        )
        save_config(save_path.with_suffix('.json'), config)

    return shape, location, scale
