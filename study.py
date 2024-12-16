""" Study of the numerical convergence of different estimator """

import os
import openturns as ot
import openturns.viewer as viewer
import numpy as np
import matplotlib.pyplot as plt
# , superquantile_labopin, superquantile_min
from riskmeasure import superquantile


# NUMBER_OF_SAMPLES = 10000
NUMBER_OF_SAMPLES = 500
SAMPLE_SIZES = [10, 100, 500, 1000]

METHODS_NAME = ['standard',    'labopin',             'minimization']

SAMPLE_SIZE = 100
METHOD = superquantile

MU = 0
SIGMA = 1
ALPHA = 0.9

seed = os.getpid()
ot.RandomGenerator.SetSeed(seed)
print("------------------------------")
print(f"The random seed is {seed}")
print("------------------------------")

normal = ot.Normal(MU, SIGMA)


def superquantile_th(alpha, mu=0, sigma=1):
    """ Compute the alpha-superquantile of a normal distribution of mean mu and
        standart-deviation sigma. """
    normal_centered = ot.Normal(0, 1)
    return mu + sigma/(1-alpha) * (
        normal_centered.computePDF(normal_centered.computeQuantile(alpha)))


def get_superquantile_estimators(alpha,
                                 sample_size,
                                 number_of_samples,
                                 method):
    """ Get several realisation of a superquantile estimator using the given
        method.
        Returns an openturns sample.
    """
    estim = np.zeros(number_of_samples)
    for i in range(number_of_samples):
        sample = normal.getSample(sample_size)
        estim[i] = superquantile(sample, alpha, method)
    return ot.Sample.BuildFromPoint(estim)


def bias_variance(alpha, sample):
    """ Compute bias, variance and quadratique error of a sample that
        approximates the superquantile of order alpha. """
    bias1 = np.mean(sample) - superquantile_th(alpha, MU, SIGMA)
    variance1 = np.var(sample)
    err1 = bias1 ** 2 + variance1
    return bias1, variance1, err1


estimator_id = np.zeros((len(METHODS_NAME), len(SAMPLE_SIZES)))
estimator_value = []

bias = np.zeros((len(METHODS_NAME), len(SAMPLE_SIZES)))
err = np.zeros((len(METHODS_NAME), len(SAMPLE_SIZES)))
var = np.zeros((len(METHODS_NAME), len(SAMPLE_SIZES)))

""" FAIRE DES CLASSES ? """
""" IL FAUT METTRE L'ECART-TYPE DIVISÉ PAR L'ESPERANCE """
""" continuer avec la variance et le coeff de variation"""


for (i, method) in enumerate(METHODS_NAME):
    for (j, n) in enumerate(SAMPLE_SIZES):
        estim0 = get_superquantile_estimators(
            ALPHA, n, NUMBER_OF_SAMPLES, method=method)
        estimator_id[i, j] = len(estimator_value)
        estimator_value.append(estim0)
        bias0, var0, err0 = bias_variance(ALPHA, estim0)
        bias[i, j] = bias0
        err[i, j] = err0
        var[i, j] = var0

for (i, method_name) in enumerate(METHODS_NAME):
    plt.loglog(SAMPLE_SIZES, err[i], '-o', label=method_name)

plt.xlabel('Sample size')
plt.ylabel('Quadratic error of the estimator of the superquantile')
plt.legend()

plt.show()
