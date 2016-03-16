from itertools import combinations

import numpy as np
import time
from matplotlib import pyplot as plt

import constants
from models.features import Features, IdentityFeatures, BasicFeatures, \
    BasicFeaturesExtended, BasicFeaturesNoNormalized, GaussianFeatures
from utils import file


class GeneralFeatures:
    def __init__(self, n_items: int, features: np.ndarray, l_dims: int,
                 k_dims: int):
        assert features.shape[0] == n_items

        # Problem parameters
        self.n_items = n_items
        self.features = features
        self.l_dims = l_dims
        self.k_dims = k_dims
        self.m_feats = features.shape[1]

        # Model parameters
        self.a_weights = np.zeros(self.m_feats)
        self.b_weights = np.zeros((self.m_feats, self.l_dims))
        self.c_weights = np.zeros((self.m_feats, self.k_dims))
        self.n_logz = 0.

        # Composite model parameters
        self.utilities = np.dot(self.features, self.a_weights)
        self.diversity_weights = np.dot(self.features, self.b_weights)
        self.coherence_weights = np.dot(self.features, self.c_weights)

        # Statistics
        self.stats = {
            'params_time': [0.0, 0],
            'evaluation': [0.0, 0]
        }
        self.distribution = {}

    def update_composite_parameters(self):
        start = time.time()
        np.dot(self.features, self.a_weights, out=self.utilities)
        if self.l_dims:
            np.dot(self.features, self.b_weights, out=self.diversity_weights)
        if self.k_dims:
            np.dot(self.features, self.c_weights, out=self.coherence_weights)
        elapsed = time.time() - start
        self.stats['params_time'][0] += elapsed
        self.stats['params_time'][1] += 1

    def __call__(self, s: np.ndarray) -> np.ndarray:
        if len(s) == 0:
            return self.n_logz
        start = time.time()
        div_weights_view = self.diversity_weights[s, :]
        coh_weights_view = self.coherence_weights[s, :]
        logprob = self.n_logz
        logprob += np.sum(self.utilities[s])
        if self.l_dims:
            logprob += np.sum((np.max(div_weights_view, axis=0) -
                        np.sum(div_weights_view, axis=0)))
        if self.k_dims:
            logprob += np.sum((np.sum(coh_weights_view, axis=0) -
                        np.max(coh_weights_view, axis=0)))
        elapsed = time.time() - start
        self.stats['evaluation'][0] += elapsed
        self.stats['evaluation'][1] += 1
        return logprob

    def full_distribution(self):
        sum = 0
        for length in range(1, self.n_items + 1):
            for subset in combinations(range(self.n_items), length):
                prob = np.exp(self(list(subset)))
                sum += prob
                self.distribution[frozenset(subset)] = prob

        for subset in self.distribution:
            self.distribution[subset] /= sum
        return self.distribution

    def propose_set_item(self, to_complete: np.ndarray) -> np.ndarray:
        gains = np.zeros(self.n_items)

        gains += self.utilities
        if self.l_dims:
            gains -= np.sum(self.diversity_weights, axis=1)
        if self.k_dims:
            gains += np.sum(self.coherence_weights, axis=1)

        if self.l_dims:
            current_max_per_d = np.max(self.diversity_weights[to_complete, :], axis=0)
            max_per_dimension = np.repeat(
                    current_max_per_d.reshape((1, self.l_dims, 1)),
                    self.n_items, axis=0)
            gains += np.sum(
                np.max(
                    np.concatenate(
                        (self.diversity_weights.reshape((self.n_items, self.l_dims, 1)),
                         max_per_dimension),
                        axis=2),
                    axis=2),
                axis=1)

        if self.k_dims:
            current_max_per_d = np.max(self.coherence_weights[to_complete, :], axis=0)
            max_per_dimension = np.repeat(
                    current_max_per_d.reshape((1, self.k_dims, 1)),
                    self.n_items, axis=0)
            gains -= np.sum(
                np.max(
                    np.concatenate(
                            (self.coherence_weights.reshape((self.n_items, self.k_dims, 1)),
                             max_per_dimension),
                    axis=2),
                axis=2),
            axis=1)

        gains[to_complete] = -np.inf
        return np.argsort(gains)[to_complete.shape[0]:][::-1]

    def load_from_file(self, input_path: str):
        with open(input_path) as input_file:
            lines = list(input_file)

            n_logz = float(lines[0].strip())
            a_weights = [float(x) for x in lines[1].strip().split(',')]
            b_weights = []
            c_weights = []
            index = 2
            if self.l_dims > 0:
                for line in lines[index:index+self.m_feats]:
                    row = []
                    for item in line.strip().split(','):
                        row.append(float(item))
                    b_weights.append(row)
                index += self.m_feats
            if self.k_dims > 0:
                for line in lines[index:]:
                    row = []
                    for item in line.strip().split(','):
                        row.append(float(item))
                    c_weights.append(row)

            a_weights = np.array(a_weights)
            b_weights = np.array(b_weights)
            c_weights = np.array(c_weights)

            self.a_weights = a_weights
            self.b_weights = b_weights
            self.c_weights = c_weights
            self.n_logz = n_logz
            self.update_composite_parameters()


def load_and_evaluate(dataset_name: str, n_items: int, features: Features):
    for fold in range(1, constants.N_FOLDS + 1):
        for l_dim in range(5, 6):
            k_dim = l_dim
            model = GeneralFeatures(n_items, features.as_array(), l_dim, k_dim)
            model.load_from_file(constants.NCE_OUT_GENERAL_PATH_TPL.format(
                dataset=dataset_name, fold=fold, l_dim=l_dim, k_dim=k_dim,
                index=features.index))
            loaded_test_data = file.load_csv_test_data(
                constants.PARTIAL_DATA_PATH_TPL.format(
                    fold=fold, dataset=dataset_name))
            target_path = constants.RANKING_MODEL_PATH_TPL.format(
                dataset=dataset_name, fold=fold,
                model='submod_f_{}_l_{}_k_{}'.format(features.index, l_dim, k_dim))
            with open(target_path, 'w') as output_file:
                for subset in loaded_test_data:
                    subset.remove('?')
                    subset = np.array([int(item) for item in subset])
                    result = model.propose_set_item(subset)
                    output_file.write(','.join(str(item) for item in result))
                    output_file.write('\n')


def main():
    n_items = 100
    dataset_name = constants.DATASET_NAME_TPL.format('100_no_singles')
    features = GaussianFeatures(dataset_name, n_items=n_items,
                                m_features=n_items, sigma=0.2)
    features.load_from_file()
    load_and_evaluate(dataset_name, n_items, features)

if __name__ == '__main__':
    main()
