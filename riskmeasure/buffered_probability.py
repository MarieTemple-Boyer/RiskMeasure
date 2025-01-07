""" Numerical estimators of the buffered probability of excedance. """

import openturns as ot
import numpy as np
import scipy

from riskmeasure import superquantile


def failure_probability(sample, threshold):
    """ Compute the failure probability for the given threshold.
    >>> failure_probability(SAMPLE, 0)
    np.float64(0.8)
    >>> failure_probability(SAMPLE, 2)
    np.float64(0.8)
    >>> failure_probability(SAMPLE, 4)
    np.float64(0.4)
    >>> failure_probability(SAMPLE, 5.5) 
    np.float64(0.2)
    >>> failure_probability(SAMPLE, 7)
    np.float64(0.0)
    """
    sample = np.resize(sample, sample.getSize())
    return np.mean(sample >= threshold)


def _buffered_probability_min(sample, threshold):
    """ Compute the buffered probability of excedance for the given threshold
    using the minimisation formula.
    >>> _buffered_probability_min(SAMPLE, 0)
    np.float64(1.0)
    >>> _buffered_probability_min(SAMPLE, 2)
    np.float64(1.0)
    >>> _buffered_probability_min(SAMPLE, 4)
    np.float64(0.8400000000000001)
    >>> _buffered_probability_min(SAMPLE, 5.5)
    np.float64(0.4)
    >>> _buffered_probability_min(SAMPLE, 7)
    np.float64(0.0)
    """

    sample = np.resize(sample, sample.getSize())

    # Compute of the positive part of -1/(sample[i] - threshold)
    transform_sample = ((sample-threshold) < 0) * (-1/(sample-threshold))

    # Compute the minimum of psi_n(transform_sample) where
    #   psi_n(a) = mean([a*(sample-threshold+1]_+)
    return np.min(
        np.mean(
            (transform_sample[:, np.newaxis]*(sample - threshold)+1 >
             0) * (transform_sample[:, np.newaxis]*(sample-threshold)+1),
            axis=1))


def _buffered_probability_inverse(sample, threshold):
    """ Compute the buffered probability of excendance for the given threshold
    by inversing the superquantile.
    >>> _buffered_probability_inverse(SAMPLE, threshold=0)
    1
    >>> _buffered_probability_inverse(SAMPLE, threshold=2)
    1
    >>> _buffered_probability_inverse(SAMPLE, threshold=5.5)
    np.float64(0.5090909090910027)
    >>> _buffered_probability_inverse(SAMPLE, threshold=7)
    0
    """
    def f(alpha):
        return superquantile(sample, alpha, method='labopin') - threshold
    if f(0) > 0:
        return 1
    if f(1)<0:
        return 0
    res_intermediary = scipy.optimize.bisect(f, 0, 1, xtol=0.05, rtol=0.05)
    #res_final = scipy.optimize.newton(f, res_intermediary, maxiter=100)
    res_final = scipy.optimize.fsolve(f, res_intermediary)[0]
    return 1-res_final


def buffered_probability(sample, threshold, method='minimization'):
    """ Compute the buffered probability of excedance for the given threshold
    using the given method.
    The possible methodes are : minimization, inverse.
    >>> buffered_probability(SAMPLE_REAL, threshold=1.4)
    np.float64(0.19047389352929056)
    >>> buffered_probability(SAMPLE_REAL, threshold=1.4, method='inverse')
    np.float64(0.19253685724916692)
    """
    if method == 'minimization':
        return _buffered_probability_min(sample, threshold)
    if method == 'inverse':
        return _buffered_probability_inverse(sample, threshold)
    raise Exception(f"The methode '{method}' is unknowns. \n"
                    "The available methods are given in the documentation.")


if __name__ == "__main__":
    import doctest
    SAMPLE = ot.Sample.BuildFromPoint([5, 3, 6, -1, 3])
    SAMPLE_REAL = ot.Normal(0, 1).getSample(500)
    doctest.testmod()
