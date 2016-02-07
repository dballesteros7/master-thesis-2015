from itertools import combinations

import numpy as np

import constants
from models.features import BasicFeatures, IdentityFeatures
from models.general_features import GeneralFeatures
from plots.histograms import len_histogram, pairs_histogram


class BruteForceSampler:
    def __init__(self, n_items: int, model):
        self.n_items = n_items
        self.model = model
        self.ordered_sets = []
        self._probabilities = []
        self.probabilities = []

    def train(self):
        full_set = list(range(self.n_items))
        for set_size in range(0, self.n_items + 1):
            item_sets_for_size = combinations(full_set, set_size)
            for item_set in item_sets_for_size:
                self.ordered_sets.append(frozenset(item_set))
                self._probabilities.append(np.exp(self.model(
                    np.array(item_set))))
        self.probabilities = np.array(self._probabilities)
        total_sum = np.sum(self.probabilities)
        self.probabilities = self.probabilities / total_sum

    def sample(self, n_samples: int):
        return np.random.choice(
            self.ordered_sets, size=n_samples, p=self.probabilities)


def main():
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
    sampler = BruteForceSampler(constants.N_ITEMS, model)
    sampler.train()
    samples = sampler.sample(100000)
    len_histogram(samples)
    pairs_histogram(samples)


if __name__ == '__main__':
    main()
