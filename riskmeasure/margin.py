""" Numerical estimators of the margin. """

import numpy as np
import openturns as ot


def margin(sample, lanbda=0):
    """ Compute the empirical margin of the sample using the estimator of the
    expectation and the unbiased estilator of the variance.
    >>> margin(SAMPLE)
    np.float64(3.2)
    >>> margin(SAMPLE_REAL, lanbda=1)
    np.float64(0.9948186563368715)
    """
    sample = np.resize(sample, sample.getSize())
    avg = np.mean(sample)
    var = np.mean(np.square(sample-avg))
    return avg+lanbda*np.sqrt(var)
    #return (sample.computeMean()[0]
    #        + lanbda*np.sqrt(sample.computeCovariance()[0, 0])
    #        )

def dual_margin(sample, threshold=0):
    """ Compute the dual probability of the margin of the sample for
    lanbda: alpha -> -ln(1-alpha)
    >>> dual_margin(SAMPLE)
    1
    >>> dual_margin(SAMPLE_REAL, threshold=1)
    np.float64(0.3659466793513913)
    """

    sample = np.resize(sample, sample.getSize())
    avg = np.mean(sample)
    var = np.mean(np.square(sample-avg))
    if threshold<avg:
        return 1
    return np.exp(-(threshold-avg)/np.sqrt(var))

if __name__ == "__main__":
    import doctest
    SAMPLE = ot.Sample.BuildFromPoint([5, 3, 6, -1, 3])
    SAMPLE_REAL = ot.Normal(0, 1).getSample(500)
    doctest.testmod()
