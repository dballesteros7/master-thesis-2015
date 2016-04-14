import os
from collections import defaultdict
from itertools import combinations
from typing import Iterable

import matplotlib as mpl
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


import numpy as np
from sklearn import linear_model
from scipy import linalg
from scipy import special

import constants
from models.features import BasicFeatures, IdentityFeatures, \
    BasicFeaturesNoNormalized, BasicFeaturesExtended, GaussianFeatures, \
    GaussianExtended
from utils import file


class ModularWithFeatures:
    def __init__(self, n_items: int, features: np.ndarray):
        assert n_items > 0
        assert features.shape[0] == n_items
        self.n_features = features.shape[1]
        self.n_items = n_items
        self.feature_weights = np.zeros(self.n_features)
        self.features = features
        self.logz = 0.
        self.exact_utilities = np.zeros(self.n_items)
        self.utilities = np.zeros(self.n_items)
        self.item_probs = np.ones(self.n_items)
        self.model_probs = np.zeros(self.n_items)
        self.distribution = {}
        self.is_identity_features = ((
            np.identity(n_items) == self.features).all()
                if self.features.shape == (n_items, n_items) else False)

    def train(self, set_samples: np.ndarray):
        for sample in set_samples:
            for element in sample:
                self.item_probs[element] += 1
        self.item_probs /= (len(set_samples) + self.n_items)

        np.log(1 / self.item_probs - 1, out=self.exact_utilities)
        self.exact_utilities *= -1
        regularized = linear_model.Ridge(alpha=0.001, fit_intercept=False,
                                         normalize=False, copy_X=True)
        regularized.fit(self.features, self.exact_utilities)
        self.feature_weights = regularized.coef_
        np.dot(self.features, self.feature_weights, out=self.utilities)
        special.expit(self.utilities, out=self.model_probs)
        special.expit(self.exact_utilities, out=self.item_probs)
        self.logz = np.sum(np.log1p(np.exp(self.utilities)))

    def sample(self, n: int, use_real_probs: bool = False) -> np.ndarray:
        probs = self.item_probs if use_real_probs else self.model_probs
        data = []
        for _ in range(n):
            s = np.nonzero(
                np.random.random_sample(self.n_items) <= probs)[0]
            data.append(s)
        return np.array(data)

    def __call__(self, item_set: Iterable[int]) -> float:
        return np.sum(self.utilities[item_set]) - self.logz

    def full_distribution(self):
        prob_sum = 0
        for length in range(0, self.n_items + 1):
            for subset in combinations(range(self.n_items), length):
                prob = np.exp(self(list(subset)))
                prob_sum += prob
                self.distribution[frozenset(subset)] = prob

        for subset in self.distribution:
            self.distribution[subset] /= prob_sum
        return self.distribution

    def propose_set_item(self, to_complete: Iterable[int]) -> Iterable[int]:
        utilities = np.copy(self.utilities)
        utilities[to_complete] = -np.inf
        sorted_indexes = np.argsort(utilities)[len(to_complete):]
        return sorted_indexes[::-1]


def learn_from_single_file():
    n_items = 10
    dataset_name = 'path_set_10'
    features = np.identity(n_items)
    loaded_data = file.load_csv_data(
        os.path.join(constants.DATA_PATH, dataset_name))
    modular_model = ModularWithFeatures(
            n_items=n_items, features=features)
    modular_model.train(loaded_data)


def main():
    np.random.seed(constants.SEED)
    dataset_name = constants.DATASET_NAME_TPL.format('100_no_singles')
    n_items = 100
    features = IdentityFeatures(dataset_name, n_items=n_items,
                                m_features=n_items)
    features.load_from_file()
    features_array = features.as_array()
    # set_probabilities = defaultdict(list)
    # fold_row = []
    # error_row.append(fold_row)
    for fold in range(1, constants.N_FOLDS + 1):
        loaded_data = file.load_csv_data(
            constants.TRAIN_DATA_PATH_TPL.format(
                fold=fold, dataset=dataset_name))
        loaded_test_data = file.load_csv_test_data(
            constants.RANKING_MODEL_PATH_TPL.format(
                fold=fold, dataset=dataset_name,
                model='partial'))
        modular_model = ModularWithFeatures(
            n_items=n_items, features=features_array)
        modular_model.train(loaded_data)

        # error = np.sum(np.power(modular_model.utilities - modular_model.exact_utilities, 2)) / n_items
        # fold_row.append(error)
        # modular_model.full_distribution()
        # for subset, prob in sorted(
        #         modular_model.distribution.items(), key=lambda x: x[1]):
        #     set_probabilities[subset].append(prob)

        target_path = constants.RANKING_MODEL_PATH_TPL.format(
            dataset=dataset_name, fold=fold,
            model='modular_features_{}'.format(features.index))
        with open(target_path, 'w') as output_file:
            for subset in loaded_test_data:
                subset.remove('?')
                subset = [int(x) for x in subset]
                result = modular_model.propose_set_item(np.array(subset))
                output_file.write(','.join(str(item) for item in result))
                output_file.write('\n')
                # if subset.index('?') > 0:
                #     short_subset = subset[:subset.index('?')]
                #     short_subset = [int(item) for item in short_subset]
                #     result = modular_model.propose_set_item(np.array(short_subset))
                #     output_file.write(','.join(str(item) for item in result))
                #     output_file.write('\n')
                # else:
                #     output_file.write('-\n')
    # mean_errors = []
    # for row in errors:
    #     mean_row_errors = []
    #     mean_errors.append(mean_row_errors)
    #     for cell in row:
    #         mean_row_errors.append(np.mean(cell))

    # error_dataset = pd.DataFrame(data=mean_errors, index=sigma_vals,
    #                              columns=m_features)
    # error_dataset.to_csv(constants.MODULAR_MODEL_ERROR_PATH_TPL.format(
    #     dataset=dataset_name, model='modular_features'))
    # means = {}
    # for subset, prob_list in set_probabilities.items():
    #     means[subset] = (np.mean(prob_list), np.std(prob_list))
    #
    # for subset, prob in sorted(means.items(), key=lambda x: x[1]):
    #     print('{}:{:.2f} +- {:.2f}%'.format(list(subset), prob[0] * 100, prob[1] * 100))

if __name__ == '__main__':
    main()
