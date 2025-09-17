""" Plot for the exact decision criterion with the failure probability """

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import openturns as ot


BETA = 0.1

ALPHA = 0.95

SAMPLE_SIZES = [100, 1000, 10000]

NUMBER_FAIL = 10000
FAIL_MAX = 0.1

''' Graphical parameters '''
font = {'size': 25} 
matplotlib.rc('font', **font)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
LINEWIDTH = 3

fail_val = np.linspace(0, FAIL_MAX, NUMBER_FAIL, endpoint=False)

def proba_safe_exact(alpha, failure_probability, sample_size):
    """ Return the probability that a system is considered safe for the
        acceptable risk 1-alpha with the given sample_size.
    """
    binom_alpha = ot.Binomial(sample_size, 1-alpha)
    binom_fail = ot.Binomial(sample_size, failure_probability)
    return binom_fail.computeCDF(binom_alpha.computeScalarQuantile(BETA)-1)


cmap = plt.get_cmap("tab10")
linestyles = ['dotted', 'dashed', 'solid']
for (i, sample_size) in enumerate(SAMPLE_SIZES):
    prob = [proba_safe_exact(ALPHA, fail_prob, sample_size) for fail_prob in fail_val]
    plt.plot(fail_val,
             prob,
             color=cmap(i),
             linestyle=linestyles[i],
             label=sample_size,
             linewidth=LINEWIDTH)

plt.plot([0, FAIL_MAX], [BETA, BETA], color='black')

plt.plot([1-ALPHA, 1-ALPHA], [0,1], color='black')
plt.grid()

plt.annotate(r'$\beta$', xy=(0, BETA), ha='right')
plt.annotate(r'$1-\alpha$', xy=(1-ALPHA, -0.04), ha='center')

plt.legend(title=r'Sample size ($n$)')
plt.xlabel(r'$p_s(Y)$')
plt.ylabel(r'$\mathbb{P}\left (\widehat p_{s,n} < \frac{1}{n}q_\beta(\mathcal{B}(n, 1-\alpha))\right )$')


plt.show()
