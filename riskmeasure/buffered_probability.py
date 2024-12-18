""" Numerical estimators of the buffered probability of excedance. """

import openturns as ot
import numpy as np


def failure_probability(sample, threshold):
    """ Compute the failure probability for the given threshold.
    >>> failure_probability(SAMPLE, 0)
    np.float64(0.8)
    >>> failure_probability(SAMPLE, 2)
    np.float64(0.8)
    >>> failure_probability(SAMPLE, 4)
    np.float64(0.4)
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


def buffered_probability(sample, threshold):
    """ Compute the buffered probability of excedance for the given threshold
    using the given method.
    """
    return _buffered_probability_min(sample, threshold)


if __name__ == "__main__":
    import doctest
    SAMPLE = ot.Sample.BuildFromPoint([5, 3, 6, -1, 3])
    doctest.testmod()
