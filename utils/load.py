from pathlib import Path

import numpy as np
import pandas as pd


def load_json_to_df(
    job="extract.legendre_count",
    output_folder="./outputs"
):
    _output_dir = Path(output_folder).absolute()
    _results_dir = _output_dir / job
    result_files = list(_results_dir.glob(f"**/*.json"))

    print(f"Found {len(result_files)} files")

    df = pd.DataFrame()

    for i, file in enumerate(result_files):
        print(f"{i+1}/{len(result_files)}", end='\r')

        with open(file, 'r') as f:
            data = pd.read_json(f, orient='index').T
            # df.append(data, ignore_index=True)
            df = pd.concat((df, data), ignore_index=True)

    return df.infer_objects()


def load_npy_stats_to_df(
    job="extract.pareto_params",
    suffix=None,
    rv_name="gumbel_l",
    output_folder="./outputs",
    save=True,
    load_from_saved=True,
):
    from scipy import stats

    from dataset import smir_datasets
    spargair = smir_datasets['spargair']

    # Parameters
    fs = 48000
    n_fft = 1024
    fl = 2608.0
    fh = 5216.0
    j_nu = 25

    fimin = int(round(fl / fs * n_fft))
    fimax = int(round(fh / fs * n_fft))

    # Find corresponding .npy files
    _output_dir = Path(output_folder).absolute()
    _results_dir = _output_dir / job
    _suffix = suffix or ""
    result_files = list(_results_dir.glob(f"**/*{_suffix}.npy"))

    print(f"Found {len(result_files)} files")
    if not len(result_files): 
        return

    # Check if already saved exists
    if load_from_saved:
        saved_files = list(_results_dir.glob(f"*{suffix}_{rv_name}.csv"))
        if len(saved_files):  # either 0 or 1
            print("Found saved results file")
            df = pd.read_csv(saved_files[0])
            if len(df) == len(result_files):
                print("Saved results file is up to date")
                return df

    # Define random variable
    rv = eval(f"stats.{rv_name}")

    df = pd.DataFrame()

    rv_stats_list = []

    for i, f_npy in enumerate(result_files):
        print(f"{i+1}/{len(result_files)}", end='\r')

        npy = np.load(f_npy)
        if len(npy.shape) > 1:  # Flattened arrays already have necessary data only
            npy = npy[:, fimin : fimax + j_nu]  # Discard unused frequency indices
        
        rv_stats = rv.fit(npy.flatten())  # Discard frequency information
        rv_stats_list.append(rv_stats)

        # Metadata parsing
        filename = f_npy.stem.split(suffix)[0] + '.wav'
        pos_folder = f_npy.parent.stem
        position = [int(p) for p in pos_folder]
        distance = spargair.get_distance(position)

        # Load up the dataframe
        data = pd.DataFrame([dict(
            sndfile = filename,
            pos_grid = pos_folder,
            dist_mic = distance,
        )])
        df = pd.concat((df, data), ignore_index=True)

    rv_stats_arr = np.array(rv_stats_list)

    # Batch load rv stats into df
    for i in range(rv_stats_arr.shape[1]):
        df[f"{rv_name}_{i}"] = rv_stats_arr[:, i]

    # Save
    if save:
        _job_root = job.split(".")[1]
        df.to_csv(_results_dir.absolute() / f"{_job_root}{_suffix}_{rv_name}.csv", index=False)

    return df.infer_objects()
