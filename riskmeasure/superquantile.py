""" Numerical estimators of the superquantile. """

import math
import numpy as np
import openturns as ot


def quantile(sample, alpha):
    """ Compute the empirical alpha-quantile of the sample.
    >>> quantile(SAMPLE, alpha=0.8)
    5.0
    >>> quantile(SAMPLE, alpha=0)
    -1.0
    >>> quantile(SAMPLE, alpha=1)
    6.0
    """
    assert sample.getDimension() == 1
    sample_sorted = sample.sortAccordingToAComponent(0)
    n = len(sample)
    ind = max(0, math.ceil(n*alpha-1))
    return sample_sorted[ind, 0]


def _superquantile_standard(sample, alpha):
    """ Compute the empirical alpha-superquantile of the sample.
    >>> _superquantile_standard(SAMPLE, alpha=0.8)
    np.float64(5.5)
    >>> _superquantile_standard(SAMPLE, alpha=0)
    np.float64(3.2)
    >>> _superquantile_standard(SAMPLE, alpha=1)
    np.float64(6.0)
    """
    assert sample.getDimension() == 1
    sample_sorted = sample.sortAccordingToAComponent(0)
    n = len(sample)
    ind = max(0, math.ceil(n*alpha)-1)
    return np.sum(sample_sorted[ind:])/(n-ind)


def _superquantile_labopin(sample, alpha):
    """ Compute the empirical alpha-superquantile of the sample.
    >>> _superquantile_labopin(SAMPLE, alpha=0.8)
    np.float64(6.000000000000002)
    >>> _superquantile_labopin(SAMPLE, alpha=0)
    np.float64(3.2)
    >>> _superquantile_labopin(SAMPLE, alpha=1)
    np.float64(6.0)
    """
    assert sample.getDimension() == 1
    sample = np.resize(sample, sample.getSize())
    sample = np.sort(sample)
    n = len(sample)
    # ind = max(0, math.ceil(n*alpha)-1)
    ind = max(0, math.floor(n*alpha))
    if alpha == 1:
        return sample[-1]
    return 1/(1-alpha) * 1/n * np.sum(sample[ind:])


def _superquantile_min(sample, alpha):
    """ Compute the alpha-quantile using the minimisation formula.
    >>> _superquantile_min(SAMPLE, alpha=0.8)
    np.float64(6.0)
    >>> _superquantile_min(SAMPLE, alpha=0)
    np.float64(3.2)
    >>> _superquantile_min(SAMPLE, alpha=1)
    np.float64(6.0)
    >>> start = time.time()
    >>> _superquantile_min(SAMPLE_BIG, alpha=0.95)
    np.float64(2.0456746533999306)
    >>> end = time.time()
    >>> print(np.round(end-start,1))
    1.6
    """
    assert sample.getDimension() == 1
    sample = np.array(sample)
    if alpha == 1:
        return np.max(sample)

    # Compute the minimum of phi_n(sample) where
    #   phi_n(c) = c + 1/(1-alpha) * mean((sample-c)_+)
    return np.min(sample+1/(1-alpha)*np.mean(
        (sample-sample[:, np.newaxis])*(sample-sample[:, np.newaxis] >= 0),
        axis=1))


def _superquantile_min_sort(sample, alpha):
    """ Compute the alpha-quantile using the minimisation formula
        (and by sorting the sample).
    >>> _superquantile_min_sort(SAMPLE, alpha=0.8)
    np.float64(6.0)
    >>> _superquantile_min_sort(SAMPLE, alpha=0)
    np.float64(3.2)
    >>> _superquantile_min_sort(SAMPLE, alpha=1)
    np.float64(6.0)
    >>> start = time.time()
    >>> _superquantile_min_sort(SAMPLE_BIG, alpha=0.95)
    np.float64(2.0456746533999306)
    >>> end = time.time()
    >>> print(np.round(end-start, 2))
    0.02
    """
    assert sample.getDimension() == 1
    sample = np.sort(np.array(sample)[:, 0])
    if alpha == 1:
        return sample[-1]

    # Compute the minimum of phi_n(sample) where
    #   phi_n(c) = c + 1/(1-alpha) * mean((sample-c)_+)
    def phi_n(c):
        return c + 1/(1-alpha) * np.mean(np.maximum(0, sample-c))
    it = 0
    min1 = phi_n(sample[-1])
    min2 = phi_n(sample[-2])
    while min2 <= min1 and it < len(sample)-2:
        min1 = min2
        min2 = phi_n(sample[-it-3])
        it += 1
    return np.min([min1, min2])

def _superquantile_min_fast(sample, alpha):
    """ Compute the alpha-quantile using the minimisation formula
        (and by sorting the sample).
    >>> _superquantile_min_fast(SAMPLE, alpha=0.8)
    np.float64(6.0)
    >>> _superquantile_min_fast(SAMPLE, alpha=0)
    np.float64(3.2)
    >>> _superquantile_min_fast(SAMPLE, alpha=1)
    np.float64(6.0)
    >>> start = time.time()
    >>> _superquantile_min_fast(SAMPLE_BIG, alpha=0.95)
    np.float64(2.0456746533999306)
    >>> end = time.time()
    >>> print(np.round(end-start, 4))
    0.0002
    """
    assert sample.getDimension() == 1
    sample = np.sort(np.array(sample)[:, 0])
    if alpha == 1:
        return sample[-1]

    # Compute the minimum of phi_n(sample) where
    #   phi_n(c) = c + 1/(1-alpha) * mean((sample-c)_+)
    def phi_n(c):
        return c + 1/(1-alpha) * np.mean(np.maximum(0, sample-c))
    return phi_n(sample[int(np.floor(alpha*len(sample)))])

def _superquantile_min_convex(sample, alpha):
    """ Compute the alpha-quantile using the minimisation formula
        (and by sorting the sample and using the convexity properties).
    >>> _superquantile_min_convex(SAMPLE, alpha=0.8)
    np.float64(6.0)
    >>> _superquantile_min_convex(SAMPLE, alpha=0)
    np.float64(3.2)
    >>> _superquantile_min_convex(SAMPLE, alpha=1)
    np.float64(6.0)
    >>> start = time.time()
    >>> _superquantile_min_convex(SAMPLE_BIG, alpha=0.95)
    np.float64(2.0456746533999306)
    >>> _superquantile_min_sort(SAMPLE_BIG, alpha=0.95)
    np.float64(2.0456746533999306)
    >>> end = time.time()
    >>> print(np.round(end-start, 2))
    0.02
    """
    n = len(sample)
    if n < 4:
        return _superquantile_min(sample, alpha)

    assert sample.getDimension() == 1
    sample = np.sort(np.array(sample)[:, 0])
    if alpha == 1:
        return sample[-1]
    # Compute the minimum of phi_n(sample) where
    #   phi_n(c) = c + 1/(1-alpha) * mean((sample-c)_+)

    def phi_n(ind):
        return sample[ind] + 1/(1-alpha) \
                    * np.mean(np.maximum(0, sample-sample[ind]))

    ind1_init = 0
    ind2_init = int(2*(n-1)/4)
    ind3_init = int(3*(n-1)/4)
    ind4_init = n-1

    def minimisation(ind1, ind2, ind3, ind4):
        assert ind1 <= ind2 <= ind3 <= ind4
        if ind4-ind1 <= 4:
            return np.min([phi_n(ind1), phi_n(ind2),
                           phi_n(ind3), phi_n(ind4)])

        if not ind1 < ind2 < ind3 < ind4:
            # Cette condition permet de garantir le assert qui suit,
            # mais nuis à la rapidité du code'''
            ind2, ind3 = int(2*(ind4-ind1)/4)+ind1, int(3*(ind4-ind1)/4)+ind1
            ind2, ind3 = int(2*(ind4-ind1)/4)+ind1, int(3*(ind4-ind1)/4)+ind1
        assert ind1 < ind2 < ind3 < ind4

        if phi_n(ind1) <= phi_n(ind2):
            ind2, ind3, ind4 = int(2*(ind2-ind1)) + \
                ind1, int(3*(ind2-ind1)/4)+ind1, ind2
            return minimisation(ind1, ind2, ind3, ind4)

        if phi_n(ind2) <= phi_n(ind3):
            if ind2-ind1 >= ind3-ind1:
                ind2, ind3, ind4 = int((ind1+ind2)/2), ind2, ind3
                return minimisation(ind1, ind2, ind3, ind4)

            ind3, ind4 = int((ind2+ind3)/2), ind3
            return minimisation(ind1, ind2, ind3, ind4)

        if ind3-ind2 >= ind4-ind1:
            ind1, ind2 = ind2, int((ind2+ind3)/2)
            return minimisation(ind1, ind2, ind3, ind4)

        ind1, ind2, ind3 = ind2, ind3, int((ind3+ind4)/2)
        return minimisation(ind1, ind2, ind3, ind4)

    return minimisation(ind1_init, ind2_init, ind3_init, ind4_init)


def superquantile(sample, alpha, method='minimization'):
    """ Compute the alpha-superquantile of the sample using the given method.
    The possible methods are : minimization, standard, labopin.
    >>> superquantile(SAMPLE_REAL, alpha=0.8)
    np.float64(1.3718827614767617)
    >>> superquantile(SAMPLE_REAL, alpha=0.8, method='standard')
    np.float64(1.3661259601219395)
    >>> superquantile(SAMPLE_REAL, alpha=0.8, method='labopin')
    np.float64(1.3718827614767617)
    >>> superquantile(SAMPLE, alpha=0.8, method='unknow_method')
    Traceback (most recent call last):
        ...
    Exception: The method 'unknow_method' is unknown.
    The available methods are given in the documentation.
    """
    if method == 'minimization':
        return _superquantile_min_fast(sample, alpha)
    if method == 'standard':
        return _superquantile_standard(sample, alpha)
    if method == 'labopin':
        return _superquantile_labopin(sample, alpha)
    raise Exception(f"The method '{method}' is unknown.\n"
                    "The available methods are given in the documentation.")


if __name__ == "__main__":
    import doctest
    import time
    ot.RandomGenerator.SetSeed(0)
    SAMPLE = ot.Sample.BuildFromPoint([5, 3, 6, -1, 3])
    SAMPLE2 = ot.Sample.BuildFromPoint([5, 6, -1, 3])
    SAMPLE_REAL = ot.Normal(0, 1).getSample(500)
    SAMPLE_BIG = ot.Normal(0, 1).getSample(2*10**4)
    doctest.testmod()
