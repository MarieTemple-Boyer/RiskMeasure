"""
    Construction of histograms of margin, quantile and superquantile estimator,
    in the case of a Pareto distribution.
"""

import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
from riskmeasure import margin, quantile, superquantile

# Value of alpha
ALPHA = 0.99

# Distribution parameters
A = 1
K = 3
distrib = ot.Pareto(A, K)


# Numerical parameters
NUMBER_OF_SAMPLES = 500
SAMPLE_SIZES = [100, 1000, 5000]
N_ASYMP = 10**7

# Definition of tools functions


def lanbda(alpha):
    ''' Function for the margin '''
    return -np.log(1-alpha)


def gaussian(x, mu=0, sigma=1):
    ''' PDF of a normal distribution '''
    return 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2 / (2*sigma**2))


normal = ot.Normal(0, 1)


# Computation of theoretical values
MARGIN_TH = K/(K-1)*A + lanbda(ALPHA)*np.sqrt(K)/((K-1)*np.sqrt(K-2)) * A
QUANTILE_TH = A*(1-ALPHA)**(-1/K)
SUPERQUANTILE_TH = (1-ALPHA)**(-1/K) / (1-1/K)


# Computation of the asymptotic variances
sd_margin = None
if K > 4:
    grad_h_trans = np.matrix([1-lanbda(ALPHA)*np.sqrt(K*(K-2)),
                              lanbda(ALPHA)/2 * (K-1)/A * np.sqrt((K-2)/K)])
    SIGMA11 = A**2 * K / ((K-1)**2 * (K-2))
    SIGMA12 = A**3 * 2*K / ((K-1)*(K-2)*(K-3))
    SIGMA22 = A**4 * 4*K / ((K-4)*(K-2)**2)

    SIGMA_MAT = np.matrix([[SIGMA11, SIGMA12],
                           [SIGMA12, SIGMA22]])
    sd_margin = grad_h_trans@SIGMA_MAT@grad_h_trans.transpose()
    sd_margin = np.sqrt(sd_margin[0, 0])

quant = distrib.computeQuantile(ALPHA)[0]
sd_quant = np.sqrt(ALPHA*(1-ALPHA))/distrib.computePDF(quant)

sample = distrib.getSample(N_ASYMP)
sd_superquantile = np.sqrt(
    np.var(quant+1/(1-ALPHA)*np.maximum(sample-quant, 0)))

sd_asymp = [sd_margin, sd_quant, sd_superquantile]


# ------------------------------
# Construction of the data set -
# ------------------------------

data = []
print('* Margin')
data_margin = []
for (j, sample_size) in enumerate(SAMPLE_SIZES):
    print(f'*--- {sample_size}')
    dat = np.zeros(NUMBER_OF_SAMPLES)
    for i in range(NUMBER_OF_SAMPLES):
        sample = distrib.getSample(sample_size)
        dat[i] = margin(sample, lanbda(ALPHA))
    data_margin.append(dat)
data.append(data_margin)

print('* Quantile')
data_quant = []
for (j, sample_size) in enumerate(SAMPLE_SIZES):
    print(f'*--- {sample_size}')
    dat = np.zeros(NUMBER_OF_SAMPLES)
    for i in range(NUMBER_OF_SAMPLES):
        sample = distrib.getSample(sample_size)
        dat[i] = quantile(sample, ALPHA)
    data_quant.append(dat)
data.append(data_quant)

print('* Superquantile')
data_sup = []
for (j, sample_size) in enumerate(SAMPLE_SIZES):
    print(f'*--- {sample_size}')
    dat = np.zeros(NUMBER_OF_SAMPLES)
    for i in range(NUMBER_OF_SAMPLES):
        sample = distrib.getSample(sample_size)
        dat[i] = superquantile(sample, ALPHA)
    data_sup.append(dat)
data.append(data_sup)

LABEL_SUPERQUANTILE = r'$\widehat{\overline{q}}_{\alpha,n}$'
LABEL_QUANTILE = r'$\widehat{q}_{\alpha,n}$'
LABEL_MARGIN = r'$\widehat{\text{mgn}}_{\alpha,n}$'


# ------------
# Graphics   -
# ------------

# Graphical parameters
TRANSPARENCY = 0.5
LINEWIDTH = 2
STEP_GRAPH = 0.1
N_GAUS = 1000
BINS = 30
cmap = plt.get_cmap('tab10')
color_list = [cmap(0), cmap(2), cmap(1)]

# Theoretical values (definition)
value_list = [MARGIN_TH, QUANTILE_TH, SUPERQUANTILE_TH]
style_list = ['dashdot', 'dotted', 'dashed']
LABEL_MARGIN_TH = r'$\text{mgn}_\alpha$'
LABEL_SUPERQUANTILE_TH = r'$\overline{q}_\alpha$'
LABEL_QUANTILE_TH = r'$q_\alpha$'
name_list_th = [LABEL_MARGIN_TH, LABEL_QUANTILE_TH, LABEL_SUPERQUANTILE_TH]
name_list = [LABEL_MARGIN, LABEL_QUANTILE, LABEL_SUPERQUANTILE]
val_th = [MARGIN_TH, QUANTILE_TH, SUPERQUANTILE_TH]


fig, axs = plt.subplots(3,
                        len(SAMPLE_SIZES),
                        sharex=True,
                        sharey=True)

mini = np.min(data) - STEP_GRAPH
maxi = np.max(data) + STEP_GRAPH

# Histograms
for i in range(3):
    for j, sample_size in enumerate(SAMPLE_SIZES):
        axs[i, j].set_xlim([mini-0.01, maxi+0.01])
        y, _, _ = axs[i, j].hist(data[i][j],
                                 bins=BINS,
                                 color=color_list[i],
                                 density=True,
                                 alpha=TRANSPARENCY)
        abscissa = np.linspace(mini, maxi,
                               N_GAUS)
        maxi_val_th = np.max(y)
        if not sd_asymp[i] is None:
            axs[i, j].plot(abscissa,
                           gaussian(
                               abscissa, mu=val_th[i], sigma=sd_asymp[i]/np.sqrt(sample_size)),
                           color=color_list[i])
            test = axs[i, j].get_ylim()
        axs[i, j].grid(True)
        axs[0, j].set_title(rf'$n=${sample_size}')

# Theoretical values
val_max_y = axs[0, 0].get_ylim()[1]
for i in range(3):
    for j, sample_size in enumerate(SAMPLE_SIZES):
        axs[i, j].plot([val_th[i],
                        val_th[i]],
                       [0, val_max_y],
                       color=color_list[i],
                       linewidth=LINEWIDTH,
                       linestyle='dotted',
                       label=name_list_th[i])
        axs[i, j].set_ylim([0, val_max_y])
        if j == len(SAMPLE_SIZES)-1:
            axs[i, j].legend()

for i in range(3):
    axs[i, 0].set_ylabel(name_list[i])


plt.show()
