import numpy as np
from tqdm import trange


def spatial_corr(Anm, Ndec, find, tind, Jtau, Jnu):
    anm = Anm[:, tind:tind+Jtau, find:find+Jnu]
    anm = anm.reshape(((Ndec + 1) ** 2, -1))  # ACN channel ordering

    Ra = anm @ anm.conj().T
    Ra = Ra / (Jtau * Jnu)
    return Ra


def singular_ratio(Anm, Ndec, find, tind, Jtau, Jnu):
    '''
    From higrid.dpd.dpd
    '''
    Ra = spatial_corr(Anm, Ndec, find, tind, Jtau, Jnu)

    S = np.linalg.svd(Ra, compute_uv=False)  # Returns sorted
    ratio = S[0] / S[1]
    return ratio


def extract_ratios(Anm, Ndec_spcorr, fimin, fimax, Jtau, Jnu):
    imax = Anm.shape[1]

    ratios_list = []
    for tind in trange(imax - Jtau, desc="Calculating singular ratios"):
        for find in range(fimin, fimax):
            ratio = singular_ratio(Anm, Ndec_spcorr, find, tind, Jtau, Jnu)
            ratios_list.append(ratio)

    return np.array(ratios_list)
