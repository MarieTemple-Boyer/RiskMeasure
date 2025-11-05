""" Compute some standard risk measure of a normal distribution. """

import openturns as ot


def pdf(x):
    """ Compute the pdf of a standard normal distribution. """
    return ot.Normal(0, 1).computePDF(x)


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

    def __init__(self, mu, sigma, lower, upper):
        self.mu = mu
        self.sigma = sigma

    def quantile(self, alpha):
        """ Return the quantile of order alpha of the distribution.
        """
        return self.mu + self.sigma * cdf_inverse(alpha)

    def superquantile(self, alpha):
        """ Return the superquantile of order alpha of the distribution.
        """
        return self.mu + self.sigma * pdf(cdf_inverse(alpha))/(1-alpha)


if __name__ == "__main__":
    import doctest
    norm = Normal(mu=0, sigma=1)
    doctest.testmod()
