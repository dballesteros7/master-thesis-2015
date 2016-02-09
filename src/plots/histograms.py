import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import constants


def len_histogram(samples):
    data = []
    for sample in samples:
        data.append(len(sample))
    weights = np.ones_like(data)/len(data)
    plt.hist(data, bins=50, weights=weights, normed=False,
             color='green', alpha=0.8)
    plt.xlabel('Set size')
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{Histogram\ of\ set\ size}$')
    plt.axis([0, 10, 0, 0.5])
    plt.xticks(np.arange(0, 11))
    plt.grid(True)
    plt.savefig(
        os.path.join(constants.IMAGE_PATH, 'length_histogram_gibbs_50.eps'),
        bbox_inches='tight')
    plt.clf()
    plt.close()


def pairs_histogram(samples):
    x_data = []
    y_data = []
    for sample in samples:
        sample = sorted(list(sample))
        if len(sample) == 2:
            x_data.append(sample[0])
            y_data.append(sample[1])

    weights = np.ones_like(x_data) / len(x_data)
    plt.hist2d(x_data, y_data, bins=40, weights=weights, cmap=cm.hot_r,
               alpha=0.8, vmin=0, vmax=0.1)
    plt.xlabel('Location 1')
    plt.ylabel('Location 2')
    plt.xticks(np.arange(0, 11))
    plt.yticks(np.arange(0, 11))
    plt.title(r'$\mathrm{Distribution\ of\ sets\ of\ size\ 2}$')
    plt.grid(True)
    plt.axis([0, 10, 0, 10])
    plt.colorbar()
    plt.savefig(
        os.path.join(constants.IMAGE_PATH, 'pairs_histogram_gibbs_50.eps'),
        bbox_inches='tight')
    plt.clf()
    plt.close()
