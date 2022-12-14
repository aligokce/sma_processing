from multiprocessing import Process
from pathlib import Path

import hydra
from omegaconf import DictConfig

from dataset import smir_datasets
from extractor import Extractor
from utils.extract import (create_input_generator, create_progress_bar,
                           print_config)


def run_serial(extractor, input_gen, save, n_audio, n_scenario):
    from time import sleep
    n_total: int = n_audio * n_scenario

    with create_progress_bar() as pgs:
        pgs.console.print(f"# of scenarios: {n_scenario}")
        pgs.console.print(f"# of anechoic sounds: {n_audio}")
        task = pgs.add_task("[green]Feature extraction...", total=n_total)

        for fpath, scenario in input_gen:
            pgs.console.print(f"\n>>> {fpath.stem}")
            pgs.console.print(f"{scenario}")
            extractor.job(fpath, scenario, save=save)
            pgs.advance(task)
            sleep(1.0)


def progress_bar_background(n_audio: int, n_scenario: int):
    from time import sleep
    n_total: int = n_audio * n_scenario

    with create_progress_bar() as pgs:
        print()
        pgs.console.print(f"# of scenarios: {n_scenario}")
        pgs.console.print(f"# of anechoic sounds: {n_audio}")
        task = pgs.add_task("[green]Feature extraction...", total=n_total)

        while True:
            output_cnt = len(list(Path.cwd().glob("**/*.npy")))
            pgs.update(task, completed=output_cnt)
            sleep(1.0)


def run_parallel(extractor, input_gen, save, n_threads):
    from functools import partial
    from multiprocessing import Pool

    job = partial(extractor.job, save=save)

    with Pool(processes=n_threads) as pool:
        pool.starmap(job, input_gen)


def prepare(config):
    ''' Dataset '''
    Dataset = smir_datasets[config.dataset._name_]
    dataset = Dataset(config.paths.smir)

    ''' Extractor '''
    extractor = Extractor(config.params, dataset)

    ''' Input generator '''
    audio_folder = config.paths.anechoic
    audio_files = list(Path(audio_folder).glob("**/*.wav"))

    input_gen = create_input_generator(
        audio_files, dataset, resume=config.resume)

    return dataset, extractor, audio_files, input_gen


@hydra.main(config_path='configs', config_name='config', version_base=None)
def main(config: DictConfig):
    print_config(config)
    
    # Prepare for extraction
    dataset, extractor, audio_files, input_gen = prepare(config)

    # Choose running serial or parallel
    if config.n_threads <= 1:
        run_serial(extractor, input_gen, save=config.save, n_audio=len(
            audio_files), n_scenario=len(dataset.metadata))
    else:
        pbar = Process(target=progress_bar_background, args=(
            len(audio_files), len(dataset.metadata)))
        pbar.start()
        run_parallel(extractor, input_gen, save=config.save,
                     n_threads=config.n_threads)
        pbar.join()


if __name__ == "__main__":
    main()
