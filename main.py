import itertools
import sys
from functools import partial
from multiprocessing import Pool

import hydra
from omegaconf import DictConfig, OmegaConf

import extract
from dataset import smir_datasets


@hydra.main(config_path='configs', config_name='config', version_base=None)
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    audio_files = config.experiment.audio_files
    positions = config.experiment.positions
    if positions == 'all':
        positions = smir_datasets[config.experiment.smir_name].generate_positions()

    with Pool(processes=config.misc.n_threads) as pool:
        job = partial(
            eval(config.experiment.job),
            **config.paths,
            **config.experiment,
            **config.params,
        )
        args_list = []
        for file, pos in itertools.product(audio_files, positions):
            args_list.append((file, pos))

        pool.starmap(job, args_list)


if __name__ == "__main__":
    sys.argv.append('hydra.job.chdir=True')
    main()
