""" Compute some standard risk measure of a normal distribution. """

import openturns as ot
import numpy as np


def pdf(x):
    """ Compute the pdf of a standard normal distribution. """
    return 1/np.sqrt(2*np.pi) * np.exp(-x**2/2)


def cdf(x):
    """ Compute the cdf of a standard normal distribution. """
    return ot.Normal(0, 1).computeCDF(x)


def cdf_inverse(alpha):
    """ Compute the inverse cdf (or quantile) of a standard normal
        distribution.
    """
    return ot.Normal(0, 1).computeScalarQuantile(alpha)


class Normal():
    """ Class for truncated normal distributions.
            - mu, sigma: average and standard-deviation of the normal
                         distribution (before truncation).
            - lower, upper: truncation.
    """

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def pdf(self, x):
        """ Compute the pdf of the distribution in x.
        >>> norm.pdf(0) == 1/np.sqrt(2*np.pi)
        np.True_
        """
        return 1/self.sigma * pdf((x-self.mu)/self.sigma)

    def quantile(self, alpha):
        """ Return the quantile of order alpha of the distribution.
        >>> norm.quantile(0.95)
        1.6448536269514726
        """
        return self.mu + self.sigma * cdf_inverse(alpha)

    def superquantile(self, alpha):
        """ Return the superquantile of order alpha of the distribution.
        >>> norm.superquantile(0.95)
        np.float64(2.0627128075074244)
        """
        return self.mu + self.sigma * pdf(cdf_inverse(alpha))/(1-alpha)


if __name__ == "__main__":
    import doctest
    norm = Normal(mu=0, sigma=1)
    doctest.testmod()
