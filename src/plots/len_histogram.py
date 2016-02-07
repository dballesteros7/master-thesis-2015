import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def len_histogram(samples):
    data = []
    for sample in samples:
        data.append(len(sample))
    weights = np.ones_like(data)/len(data)
    plt.hist(data, bins=50, weights=weights, color="blue", alpha=0.5, normed=False)
    plt.show()


def pairs_histogram(samples):
    x_data = []
    y_data = []
    for sample in samples:
        sample = sorted(list(sample))
        if len(sample) == 2:
            x_data.append(sample[0])
            y_data.append(sample[1])

    weights = np.ones_like(x_data) / len(x_data)

    plt.hist2d(x_data, y_data, bins=40, weights=weights, cmap=cm.BuGn)
    plt.colorbar()
    plt.show()
