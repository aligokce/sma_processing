from functools import partial
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import extract
from dataset import smir_datasets


def get_files_and_positions(config):
    audio_files = config.experiment.audio_files
    if audio_files == 'all':
        audio_files = [f.name for f in Path(
            config.paths.audio_folder).glob("*.wav")]

    positions = config.experiment.positions
    if positions == 'all':
        positions = smir_datasets[config.experiment.smir_name].generate_positions()
    elif positions == 'perpendicular':
        positions = smir_datasets[config.experiment.smir_name].generate_perpendicular_positions()

    return audio_files, positions


def get_remaining_files_and_positions(config, input_generator):
    for file, pos in input_generator:
        _pos = smir_datasets[config.experiment.smir_name].pos2dir(pos)
        _file = Path(file).stem
        for _ in (Path.cwd() / _pos).glob(f"{_file}*"):
            break
        else:
            yield file, pos


def get_job(config):
    func = eval(config.experiment.job)
    job = partial(
        func,
        **config.paths,
        **config.experiment,
        **config.params,
    )

    return job


@hydra.main(config_path='configs', config_name='config', version_base=None)
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    audio_files, positions = get_files_and_positions(config)
    inputs = product(audio_files, positions)
    if config.resume:
        print("Resuming processing...")
        inputs = get_remaining_files_and_positions(config, inputs)

    if config.multiprocessing:
        with Pool(processes=config.n_threads) as pool:
            pool.starmap(
                get_job(config),
                inputs
            )
    else:
        for file, pos in inputs:
            print(f"Processing {file} at position {pos}")
            get_job(config)(file, pos)


if __name__ == "__main__":
    main()
