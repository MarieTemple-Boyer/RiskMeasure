""" Compute some standard risk measure of a truncated normal distribution. """

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


class TruncatedNormal():
    """ Class for truncated normal distributions.
            - mu, sigma: average and standard-deviation of the normal
                         distribution (before truncation).
            - lower, upper: truncation.
    """

    def __init__(self, mu, sigma, lower, upper):
        self.mu = mu
        self.sigma = sigma
        self.lower = lower
        self.upper = upper

    def _stand_distrib(self):
        """ Distribution of (self-self.mu)/self.sigma.
        >>> dist1 = trunc_norm_stand._stand_distrib()
        >>> dist1.mu == trunc_norm_stand.mu \
                and dist1.sigma == trunc_norm_stand.sigma \
                and dist1.lower == trunc_norm_stand.lower \
                and  dist1.upper == trunc_norm_stand.upper
        True
        """
        return TruncatedNormal(mu=0,
                               sigma=1,
                               lower=(self.lower-self.mu)/self.sigma,
                               upper=(self.upper-self.mu)/self.sigma)

    def _average_stand(self):
        assert self.mu == 0 and self.sigma == 1
        return (pdf(self.lower) - pdf(self.upper)) \
            / (cdf(self.upper) - cdf(self.lower))

    def _quantile_stand(self, alpha):
        assert self.mu == 0 and self.sigma == 1
        return cdf_inverse((1-alpha)*cdf(self.lower) + alpha*cdf(self.upper))
    
    def _superquantile_stand(self, alpha):
        assert self.mu == 0 and self.sigma == 1
        if alpha==1:
            return self.upper
        return 1/(1-alpha) \
                        * (pdf(self._quantile_stand(alpha)) - pdf(self.upper)) \
                        / (cdf(self.upper) - cdf(self.lower))

    def average(self):
        """ Return the average of the distribution.
        >>> trunc_norm.average()
        0.45927435818265794
        >>> trunc_norm_centered.average()
        0.0
        >>> almost_untrunc.average()
        0.9999680445609462
        """
        return self.mu + self.sigma*self._stand_distrib()._average_stand()

    def quantile(self, alpha):
        """ Return the quantile of order alpha of the distribution.
        >>> trunc_norm.quantile(1)
        4.0
        >>> trunc_norm.quantile(0)
        -2.0
        >>> almost_untrunc.quantile(0.95)
        4.289644663375504
        """
        return self.mu + self.sigma*self._stand_distrib()._quantile_stand(alpha)

    def superquantile(self, alpha):
        """ Return the superquantile of order alpha of the distribution.
        >>> trunc_norm.superquantile(1)
        4
        >>> trunc_norm_centered.superquantile(0)
        0.0
        >>> almost_untrunc.superquantile(0.95)
        5.125012649267585
        """
        if alpha==1:
            return self.upper
        return self.mu + self.sigma*self._stand_distrib()._superquantile_stand(alpha)



if __name__ == "__main__":
    import doctest
    trunc_norm_stand = TruncatedNormal(mu=0, sigma=1, lower=-1, upper=5)
    trunc_norm = TruncatedNormal(mu=0, sigma=2, lower=-2, upper=4)
    trunc_norm_centered = TruncatedNormal(mu=0, sigma=2, lower=-2, upper=2)
    almost_untrunc = TruncatedNormal(mu=1, sigma=2, lower=-11, upper=10)
    doctest.testmod()
