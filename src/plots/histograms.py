import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import constants
from models.features import IdentityFeatures
from models.general_features import GeneralFeatures
from sampling.gibbs_sampler import GibbsSampler
from utils import file


def len_histogram_strict(size_counts_1, size_counts_2, size_counts_3):
    x_values = np.arange(len(size_counts_1))
    fig, ax = plt.subplots()

    width = 0.3

    rects1 = ax.bar(x_values, size_counts_1, width=width, color='#1b9e77',
                    alpha=0.8)
    rects2 = ax.bar(x_values + width, size_counts_2, width=width, color='#d95f02',
                    alpha=0.8)
    rects3 = ax.bar(x_values + 2*width, size_counts_3, width=width, color='#7570b3',
                    alpha=0.8)
    ax.set_xlabel('Set size')
    ax.set_title('Histogram of set size')
    ax.set_xticks(x_values + 0.45)
    ticks = [str(x) for x in range(1, len(size_counts_1))]
    ticks.append('{}+'.format(len(size_counts_1)))
    ax.set_ylim([0, 1])
    ax.set_xticklabels(ticks)
    ax.legend((rects1[0], rects2[0], rects3[0]),
              ('N=10', 'N=50', 'N=100'),
              loc='upper right')
    plt.savefig(os.path.join(
       constants.IMAGE_PATH, 'data_size_histogram.eps'),
       bbox_inches='tight')
    plt.show()


def len_histogram_singleton(counts_sample, counts_real):
    x_values = np.arange(len(counts_sample))
    fig, ax = plt.subplots()

    width = 0.4

    rects1 = ax.bar(x_values, counts_sample, width=width, color='#1b9e77',
                    alpha=0.8)
    rects2 = ax.bar(x_values + width, counts_real, width=width, color='#d95f02',
                    alpha=0.8)
    ax.set_xlabel('Set size')
    ax.set_title('Histogram of set size (N=10)')
    ax.set_xticks(x_values + width)
    ticks = [str(x) for x in range(1, len(counts_sample))]
    ticks.append('{}+'.format(len(counts_sample)))
    ax.set_ylim([0, 1])
    ax.set_xticklabels(ticks)
    ax.legend((rects1[0], rects2[0]),
              ('Samples', 'Data'),
              loc='upper right')
    plt.savefig(os.path.join(
       constants.IMAGE_PATH, 'data_sampling_histogram_10_with_singletons.eps'),
       bbox_inches='tight')
    plt.show()


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
    plt.show()
    # plt.savefig(
    #     os.path.join(constants.IMAGE_PATH, 'length_histogram_gibbs_50.eps'),
    #     bbox_inches='tight')
    # plt.clf()
    # plt.close()


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


def main_1():
    upper_limit = 5
    counts_array = []
    for label in ['10', '50', '100']:
        data = file.load_csv_data(constants.ALL_DATA_PATH_TPL.format(
            dataset=constants.DATASET_NAME_TPL.format(label)))
        counts = np.zeros(upper_limit)
        for subset in data:
            if len(subset) < upper_limit:
                counts[len(subset) - 1] += 1
            else:
                counts[upper_limit - 1] += 1
        counts /= np.sum(counts)
        counts_array.append(counts)
    len_histogram_strict(*counts_array)


def main_2():
    np.random.seed(constants.SEED)
    n_items = 10
    upper_limit = 5
    dataset_name = constants.DATASET_NAME_TPL.format('10')
    l_dim = 5
    k_dim = 5
    features = IdentityFeatures(dataset_name, n_items, n_items)
    features.load_from_file()
    model = GeneralFeatures(n_items, features.as_array(),
                            l_dim, k_dim)
    model.load_from_file(constants.NCE_OUT_GENERAL_PATH_TPL.format(
        dataset=dataset_name, fold=1, l_dim=l_dim, k_dim=k_dim,
        index=features.index))
    n_iter = 1000000
    sampler = GibbsSampler(n_items, model)
    sampler.train(n_iter)
    counts_sample = np.zeros(upper_limit)
    for subset in sampler.counts:
        if len(subset) == 0:
            continue
        if len(subset) < upper_limit:
            counts_sample[len(subset) - 1] += sampler.counts[subset]
        else:
            counts_sample[upper_limit - 1] += sampler.counts[subset]
    counts_sample /= np.sum(counts_sample)

    data = file.load_csv_data(constants.ALL_DATA_PATH_TPL.format(
            dataset=constants.DATASET_NAME_TPL.format('10')))
    counts = np.zeros(upper_limit)
    for subset in data:
        if len(subset) < upper_limit:
            counts[len(subset) - 1] += 1
        else:
            counts[upper_limit - 1] += 1
    counts /= np.sum(counts)

    len_histogram_singleton(counts_sample, counts)


if __name__ == '__main__':
    main_2()
