from collections import defaultdict

import numpy as np

import constants
from models.features import IdentityFeatures
from models.general_features import GeneralFeatures
from plots.len_histogram import len_histogram


class GibbsSampler:
    def __init__(self, n_items: int, model):
        self.n_items = n_items
        self.model = model
        self.counts = defaultdict(int)
        self.samples = 0

    def train(self, n_iter: int = 100000):
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


def main():
    np.random.seed(constants.SEED)
    l_dim = 2
    k_dim = 2
    features = IdentityFeatures(
        constants.DATASET_NAME, constants.N_ITEMS, constants.N_ITEMS)
    features.load_from_file()
    model = GeneralFeatures(constants.N_ITEMS, features.as_array(),
                            l_dim, k_dim)
    model.load_from_file(constants.NCE_OUT_GENERAL_PATH_TPL.format(
        dataset=constants.DATASET_NAME, fold=1, l_dim=l_dim, k_dim=k_dim,
        index=features.index))
    sampler = GibbsSampler(constants.N_ITEMS, model)
    sampler.train()
    samples = sampler.sample(100000)
    len_histogram(samples)


if __name__ == '__main__':
    main()
