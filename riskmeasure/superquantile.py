""" Numerical estimators of the superquantile. """

import math
import numpy as np
import openturns as ot


def quantile(sample, alpha):
    """ Compute the empirical alpha-quantile of the sample.
    >>> quantile(SAMPLE, alpha=0.8)
    np.float64(5.0)
    >>> quantile(SAMPLE, alpha=0)
    np.float64(-1.0)
    >>> quantile(SAMPLE, alpha=1)
    np.float64(6.0)
    """
    sample = np.resize(sample, sample.getSize())
    sample = np.sort(sample)
    n = len(sample)
    ind = max(0, math.ceil(n*alpha-1))
    return sample[ind]


def _superquantile_standard(sample, alpha):
    """ Compute the empirical alpha-superquantile of the sample.
    >>> _superquantile_standard(SAMPLE, alpha=0.8)
    np.float64(5.5)
    >>> _superquantile_standard(SAMPLE, alpha=0)
    np.float64(3.2)
    >>> _superquantile_standard(SAMPLE, alpha=1)
    np.float64(6.0)
    """
    sample = np.resize(sample, sample.getSize())
    sample = np.sort(sample)
    n = len(sample)
    ind = max(0, math.ceil(n*alpha)-1)
    return np.sum(sample[ind:])/(n-ind)


def _superquantile_labopin(sample, alpha):
    """ Compute the empirical alpha-superquantile of the sample.
    >>> _superquantile_labopin(SAMPLE, alpha=0.8)
    np.float64(11.000000000000002)
    >>> _superquantile_labopin(SAMPLE, alpha=0)
    np.float64(3.2)
    >>> _superquantile_labopin(SAMPLE, alpha=1)
    np.float64(6.0)
    """
    sample = np.resize(sample, sample.getSize())
    sample = np.sort(sample)
    n = len(sample)
    ind = max(0, math.ceil(n*alpha)-1)
    if alpha == 1:
        return sample[-1]
    return 1/(1-alpha) * 1/n * np.sum(sample[ind:])


""" PEUT DONNER AUSSI LE QUANTILE EN DONNER LE PLUS PETIT MINIMISEUR"""
def _superquantile_min(sample, alpha):
    """ Compute the alpha-quantile using the minimisation formula.
    >>> _superquantile_min(SAMPLE, alpha=0.8)
    np.float64(6.0)
    >>> _superquantile_min(SAMPLE, alpha=0)
    np.float64(3.2)
    >>> _superquantile_min(SAMPLE, alpha=1)
    np.float64(6.0)
    """
    sample = np.array(sample)
    if alpha == 1:
        return np.max(sample)

    # Compute the minimum of phi_n(sample) where
    #   phi_n(c) = c + 1/(1-alpha) * mean((sample-c)_+)
    return np.min(sample+1/(1-alpha)*np.mean(
        (sample-sample[:, np.newaxis])*(sample-sample[:, np.newaxis] >= 0),
        axis=1))

def superquantile(sample, alpha, method='minimization'):
    """ Compute the alpha-superquantile of the sample using the given method.
    The possible methods are : minimization, standard, labopin.
    >>> superquantile(SAMPLE, alpha=0.8, method='standard')
    np.float64(5.5)
    >>> superquantile(SAMPLE, alpha=0.8, method='labopin')
    np.float64(11.000000000000002)
    >>> superquantile(SAMPLE, alpha=0.8)
    np.float64(6.0)
    >>> superquantile(SAMPLE, alpha=0.8, method='unknow_method')
    Traceback (most recent call last):
        ...
    Exception: The method 'unknow_method' is unknown.
    The available methods are given in the documentation.
    """
    if method == 'minimization':
        return _superquantile_min(sample, alpha)
    if method == 'standard':
        return _superquantile_standard(sample, alpha)
    if method == 'labopin':
        return _superquantile_labopin(sample, alpha)
    raise Exception(f"The method '{method}' is unknown.\n"
                    "The available methods are given in the documentation.")


if __name__ == "__main__":
    import doctest
    SAMPLE = ot.Sample.BuildFromPoint([5, 3, 6, -1, 3])
    doctest.testmod()
