import json
from pathlib import Path
import numpy as np


def save_shd(shd, save_path, verbose=True):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, shd)

    if verbose:
        print(f"SHD shape: {len(shd)} channels, {shd[0].shape}")
        print("SHD coefficients saved to:", save_path)


def save_ratio_list(ratio_list, save_path, verbose=True):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, ratio_list)

    if verbose:
        print(f"Ratio list: length={len(ratio_list)}")
        print("Ratio values saved to:", save_path)


def save_distrib_plot(ratio_list, shape, location, scale, save_path):
    import matplotlib.pyplot as plt
    from scipy.stats import genpareto

    max_singular_ratio = 60
    histogram_bin_freq = 2
    pdf_density = 500

    x = np.linspace(0, max_singular_ratio, pdf_density)  # TODO: Fixed value?
    fitted_data = genpareto.pdf(x, shape, loc=location, scale=scale)

    plt.figure()
    plt.hist(ratio_list, bins=range(
        0, max_singular_ratio+1, histogram_bin_freq), density=True)
    plt.plot(x, fitted_data, 'r-')

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)


def save_config(save_path, config, verbose=True):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config, f)

    if verbose:
        print("Config file saved to:", save_path)
