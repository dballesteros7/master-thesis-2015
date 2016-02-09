import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import constants
from models.features import IdentityFeatures
from models.general_features import GeneralFeatures
from plots.histograms import len_histogram, pairs_histogram
from sampling.brute_force import BruteForceSampler

class GibbsSampler:
    def __init__(self, n_items: int, model):
        self.n_items = n_items
        self.model = model
        self.counts = defaultdict(int)
        self.samples = 0

    def train(self, n_iter: int = 1000000):
        initial_set_size = np.random.randint(
            low=1, high=self.n_items + 1, size=1)
        random_set = np.random.choice(self.n_items, size=initial_set_size,
                                      replace=False)
        for t in range(n_iter):
            element = np.random.randint(self.n_items, size=1)
            has_element = element in random_set
            if has_element:
                set_minus = np.array(list(set(random_set) - set(element)))
                delta_prob = self.model(random_set) - self.model(set_minus)
            else:
                set_plus = np.asarray(np.append(random_set, element),
                                      dtype=np.int64)
                delta_prob = self.model(set_plus) - self.model(random_set)
            p_add = np.exp(delta_prob) / (1 + np.exp(delta_prob))
            z = np.random.rand()
            if z <= p_add:
                if not has_element:
                    random_set = set_plus
            else:
                if has_element:
                    random_set = set_minus
            if t > n_iter / 2:
                self.samples += 1
                self.counts[frozenset(random_set)] += 1

    def sample(self, n_samples: int):
        ordered_sets = []
        probabilities = []
        for item_set in self.counts:
            ordered_sets.append(item_set)
            probabilities.append(self.counts[item_set] / self.samples)
        return np.random.choice(
            ordered_sets, size=n_samples, p=probabilities)


def plot_performance():
    np.random.seed(constants.SEED)
    l_dim = 5
    k_dim = 5
    features = IdentityFeatures(
        constants.DATASET_NAME, constants.N_ITEMS, constants.N_ITEMS)
    features.load_from_file()
    model = GeneralFeatures(constants.N_ITEMS, features.as_array(),
                            l_dim, k_dim)
    model.load_from_file(constants.NCE_OUT_GENERAL_PATH_TPL.format(
        dataset=constants.DATASET_NAME, fold=1, l_dim=l_dim, k_dim=k_dim,
        index=features.index))
    x = [100, 500, 1000, 2000, 5000, 10000, 50000, 100000, 500000, 1000000]
    #x = [100, 500, 1000, 2000, 5000, 10000]
    y = []
    for n_iter in x:
        sampler = GibbsSampler(constants.N_ITEMS, model)
        sampler.train(n_iter)
        exact_sampler = BruteForceSampler(constants.N_ITEMS, model)
        exact_sampler.train()
        error = 0
        for subset, prob in zip(exact_sampler.ordered_sets,
                                exact_sampler.probabilities):
            if subset in sampler.counts:
                error += abs(prob - (sampler.counts[subset] / sampler.samples))
            else:
                error += prob
        y.append(error)

    plt.plot(x, y, 'o')
    plt.plot(x, y, '-')
    plt.xlabel('Iterations')
    plt.ylabel('Absolute error')
    plt.title('Gibbs Sampling Error')
    plt.grid(True)
    plt.axis([min(x), max(x), 0, max(y) + 0.2])
    plt.xscale('log')
    plt.savefig(
        os.path.join(constants.IMAGE_PATH, 'gibbs_performance.eps'),
        bbox_inches='tight')


def main():
    #plot_performance()
    n_items = 50
    dataset_name = constants.DATASET_NAME_TPL.format(n_items)
    np.random.seed(constants.SEED)
    l_dim = 20
    k_dim = 20
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
    samples = sampler.sample(100000)
    len_histogram(samples)
    #pairs_histogram(samples)


if __name__ == '__main__':
    main()
