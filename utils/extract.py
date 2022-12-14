from itertools import product
from pathlib import Path

import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf

from dataset import SMIRDataset


def create_progress_bar():
    from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                               SpinnerColumn, TaskProgressColumn, TextColumn,
                               TimeElapsedColumn, TimeRemainingColumn)

    return Progress(
        TextColumn("[progress.description]{task.description}"),
        TaskProgressColumn(),
        SpinnerColumn(),
        TimeElapsedColumn(),
        BarColumn(),
        TimeRemainingColumn(),
        MofNCompleteColumn()
    )


def print_config(config: DictConfig, resolve: bool = True):
    """
    Prints content of DictConfig using Rich library and its tree structure.

    Parameters
    ----------
    config : DictConfig
        Configuration composed by Hydra.
    resolve : bool, optional
        Whether to resolve reference fields of DictConfig, by default True

    Referenced from git@HazyResearch/state-spaces
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def create_input_generator(audio_files, dataset: SMIRDataset, resume=True):
    """
    Creates an input generator for feature extraction

    Parameters
    ----------
    audio_files : List[Path]
        List of anechoic audio files for emulation
    dataset : SMIRDataset
        Dataset class representing spherical microphone impulse responses
    resume : bool, optional
        Resume only for the unextracted scenarios, by default True
    verbose : bool, optional
        Print metadata, by default True

    Yields
    ------
    Generator[PathLike, Dict]
        Input anechoic file path and dictionary containing scenario metadata
    """
    save_folder = Path.cwd()  # hydra changes cwd
    scenarios = dataset.metadata.iterrows()

    for fpath, (_, sc) in product(audio_files, scenarios):
        save_name = f"{sc['mic_pos']}_{sc['src_pos']}_{sc['src_dir']}__{fpath.stem}.npy"
        if not (resume and (save_folder / save_name).exists()):
            yield fpath, sc
        else:
            continue
