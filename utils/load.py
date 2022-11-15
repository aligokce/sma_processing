from pathlib import Path

import numpy as np
import pandas as pd


def list_feature_files(output_path="./outputs", job="extract.legendre_count", suffix="_rent"):
    """ 
    Constructs a list of feature files given its specifications
    """
    
    dir_output = Path(output_path)
    dir_job = dir_output / Path(job)
    assert dir_job.is_dir(), "Given path is not a directory"

    _suffix = suffix or ""
    file_list = list(dir_job.glob(f"**/*{_suffix}.npy"))
    assert len(file_list), f"Cannot find any feature files at path: {dir_job}"

    return file_list


def load_feature(file_path, fimin, fimax, j_nu):
    """ 
    Loads TF-domain feature from a file, discarding empty frequencies
    """

    feature = np.load(file_path)
    return feature[:, fimin:fimax+j_nu]


def parse_metadata(file_path, dataset='spargair', suffix=None):
    """ 
    Parses feature metadata given its file path
    """

    from dataset import smir_datasets

    filename = file_path.stem.split(suffix or "")[0] + '.wav'
    pos_folder = file_path.parent.stem
    position = tuple(int(p) for p in pos_folder)  # TODO: Generalize to datasets
    distance = smir_datasets[dataset].get_distance(position)

    return filename, pos_folder, distance


def batch_load_metadata(file_list, verbose=True):
    """ 
    Loads features metadata into a pandas DataFrame for a given file list
    """

    data = []
    for i, f_path in enumerate(file_list):
        if verbose:
            print(f"{i+1}/{len(file_list)}", end='\r')

        filename, pos_grid, distance = parse_metadata(
            f_path, dataset='spargair', suffix='_rent')

        data.append(dict(
            sndfile=filename,
            pos_grid=pos_grid,
            dist_mic=distance,
            path=f_path
        ))

    return pd.DataFrame(data)


def batch_load_features(file_list, fimin, fimax, j_nu, verbose=True):
    """ 
    Batch loads features given an .npy/npz path list
    """
    feature_list = []
    for i, f_path in enumerate(file_list):
        if verbose:
            print(f"{i+1}/{len(file_list)}", end='\r')

        feature = load_feature(f_path, fimin, fimax, j_nu)
        feature_list.append(feature)

    return np.array(feature_list)


def batch_load_transform_features(file_list, transformer, fimin, fimax, j_nu, verbose=True):
    """ 
    Batch loads features and transforms using the function input
    
    NOTE: Transformer function is called inside the iteration loop. Note that you may
    want to use a transforming function that accepts batch input whenever
    possible, for that probably will be much faster compared to this.
    """
    assert callable(transformer), "'transformer' is not callable"

    transformed_features_list = []
    for i, f_path in enumerate(file_list):
        if verbose:
            print(f"{i+1}/{len(file_list)}", end='\r')

        feature = load_feature(f_path, fimin, fimax, j_nu)
        transformed_features_list += transformer(feature)

    return np.array(transformed_features_list)
