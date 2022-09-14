import numpy as np
from numba import njit
from tqdm import trange


@njit
def spatial_corr(Anm, Ndec, find, tind, Jtau, Jnu):
    """
    From higrid.dpd.dpd
    """
    anm = np.zeros(((Ndec + 1) ** 2, 1), dtype=np.complex128)
    Ra = np.zeros(((Ndec + 1) ** 2, (Ndec + 1) ** 2), dtype=np.complex128)

    for fi in range(find, find + Jnu):
        for ti in range(tind, tind + Jtau):
            for ind in range((Ndec + 1) ** 2):
                anm[ind, 0] = Anm[ind][ti, fi]
            # Ra += (anm @ anm.H)
            Ra += (anm @ anm.conj().T)

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


def generate_ratios_list(Anm, Ndec_spcorr, fimin, fimax, Jtau, Jnu):
    imax = Anm[0].shape[0]

    ratios_list = []
    for tind in trange(imax - Jtau, desc="Calculating singular ratios"):
        for find in trange(fimin, fimax, leave=False):
            ratio = singular_ratio(Anm, Ndec_spcorr, find, tind, Jtau, Jnu)
            ratios_list.append(ratio)

    return np.array(ratios_list)
