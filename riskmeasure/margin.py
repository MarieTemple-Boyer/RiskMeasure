""" Numerical estimators of the margin. """

import math
import numpy as np
import openturns as ot


def margin(sample, lanbda=0):
    """ Compute the empirical margin of the sample using the estimator of the
    expectation and the unbiased estilator of the variance.
    >>> margin(SAMPLE)
    np.float64(3.2)
    >>> margin(SAMPLE_REAL, lanbda=1)
    np.float64(0.9958037516820143)
    """

    return (sample.computeMean()[0]
            + lanbda*np.sqrt(sample.computeCovariance()[0, 0])
            )


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


if __name__ == "__main__":
    import doctest
    SAMPLE = ot.Sample.BuildFromPoint([5, 3, 6, -1, 3])
    SAMPLE_REAL = ot.Normal(0, 1).getSample(500)
    doctest.testmod()
