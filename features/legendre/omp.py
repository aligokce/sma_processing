'''
Referenced from git+davebiagioni/pyomp
'''
import numpy as np


class Result(object):
    '''Result object for storing input and output data for omp.  When called from 
    `omp`, runtime parameters are passed as keyword arguments and stored in the 
    `params` dictionary.
    Attributes:
        X:  Predictor array after (optional) standardization.
        y:  Response array after (optional) standarization.
        ypred:  Predicted response.
        residual:  Residual vector.
        coef:  Solution coefficients.
        active:  Indices of the active (non-zero) coefficient set.
        err:  Relative error per iteration.
        params:  Dictionary of runtime parameters passed as keyword args.   
    '''

    def __init__(self, **kwargs):

        # to be computed
        self.X = None
        self.y = None
        self.ypred = None
        self.residual = None
        self.coef = None
        self.active = None
        self.err = None

        # runtime parameters
        self.params = {}
        for key, val in kwargs.items():
            self.params[key] = val

    def update(self, coef, active, err, residual, ypred):
        '''Update the solution attributes.
        '''
        self.coef = coef
        self.active = active
        self.err = err
        self.residual = residual
        self.ypred = ypred


def omp(X, y, ncoef=None, maxit=200, tol=1e-3, dtype=float):
    '''Compute sparse orthogonal matching pursuit solution with coefficients.

    Args:
        X: Dictionary array of size n_samples x n_features. 
        y: Reponse array of size n_samples x 1.
        ncoef: Max number of coefficients.  Set to n_features/2 by default.
        tol: Convergence tolerance.  If relative error is less than
            tol * ||y||_2, exit.
        verbose: Boolean, print some info at each iteration.

    Returns:
        result:  Result object.  See Result.__doc__
    '''

    def energy(x): return np.linalg.norm(x)**2

    # initialize result object
    result = Result(ncoef=ncoef, maxit=maxit, tol=tol)

    if type(X) is not np.ndarray:
        X = np.array(X)
    if type(y) is not np.ndarray:
        y = np.array(y)

    # check that n_samples match
    assert X.shape[0] == len(
        y), 'X and y must have same number of rows (samples)'

    # for rest of call, want y to have ndim=1
    y = np.reshape(y, (len(y),))

    # by default set max number of coef to half of total possible
    ncoef = int(X.shape[1]/2)

    # initialize things
    X_transpose = X.T                        # store for repeated use
    active = []
    coef = np.zeros(X.shape[1], dtype=dtype)  # solution vector
    residual = y                             # residual vector
    ypred = np.zeros(y.shape, dtype=dtype)
    # store for computing relative err
    ynorm = energy(y)
    err = np.zeros(maxit, dtype=float)       # relative err vector

    # convert tolerances to relative
    tol = tol * ynorm       # convergence tolerance
    # ztol = ztol * ynorm     # threshold for residual covariance

    # Check if response has zero norm, because then we're done. This can happen
    # in the corner case where the response is constant and you normalize it.
    if ynorm <= tol:     # the same as ||residual|| < tol * ||residual||
        result.update(coef, active, err[:1], residual, ypred)
        return result

    # ==========================
    for it in range(maxit):
        # compute residual covariance vector and check threshold
        rcov = np.dot(X_transpose, residual)
        i = np.argmax(np.abs(rcov))

        if i not in active:
            active.append(i)

        # solve for new coefficients on active set
        coefi, _, _, _ = np.linalg.lstsq(
            X[:, active], y, rcond=None)  # TODO: rcond?

        # update residual vector and error
        residual = y - np.dot(X[:, active], coefi)
        en_residual = np.linalg.norm(residual)**2
        err[it] = en_residual / ynorm

        if (en_residual < tol) | (len(active) >= ncoef) | (it == maxit-1):
            break

    result.update(coef, active, err[:(it+1)], residual, ypred)
    return result
