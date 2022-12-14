import numpy as np

from .legendre import generate_legendre_dict_healpix


def calculate_rent_batch(dictionary, elements):
    '''Compute Residual Energy Test (RENT) values
    It uses single iteration Orthogonal Matching Pursuit (OMP) algorithm.

    dictionary: (n_features, n_components)
    elements: (..., n_features)

    Return:
    result: shape of (...)
    '''
    X = np.array(dictionary)
    Y = elements.reshape(-1, elements.shape[-1])  # features are the last dim

    rcov = np.dot(X.T, Y.T)
    active = np.argmax(np.abs(rcov), axis=0)

    results = []
    for i, y in zip(active, Y):
        # solve for coefficients modelling active/dominant source
        _, renergy, _, _ = np.linalg.lstsq(
            X[:, [i]], y, rcond=None)  # fetch residual energy only

        err = renergy / np.linalg.norm(y)**2
        results.append(1 - err)

    # back to a multiple domain (e.g. time-frequency) representation if given as such
    return np.array(results).reshape(elements.shape[:-1])


def extract(srf, n_shd, fimin, fimax, j_nu):
    n_pix = srf.shape[-1]
    dictionary = generate_legendre_dict_healpix(n_shd, n_pix)

    rent = np.zeros(srf.shape[:2], dtype=float)
    rent[:, fimin:fimax+j_nu] = calculate_rent_batch(dictionary, srf[:, fimin:fimax+j_nu])
    return rent
