import numpy as np
from scipy.stats import genpareto

from ..dpd import generate_ratios_list


# def fit_ratios(Anm, Ndec_spcorr, fimin, fimax, Jtau, Jnu):
#     ratios_list = generate_ratios_list(Anm, Ndec_spcorr, fimin, fimax, Jtau, Jnu)
#     shape, location, scale = genpareto.fit(np.array(ratios_list))
#     return shape, location, scale


def fit_ratios(ratios_list):
    shape, location, scale = genpareto.fit(np.array(ratios_list))
    return shape, location, scale
