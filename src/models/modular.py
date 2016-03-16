import os
from collections import defaultdict
from itertools import combinations
from typing import Iterable

import numpy as np
from sklearn import linear_model
from scipy import linalg
from scipy import special

import constants
from models.features import BasicFeatures, IdentityFeatures, \
    BasicFeaturesNoNormalized, BasicFeaturesExtended, GaussianFeatures
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
        self.utilities = np.zeros(self.n_features)
        self.item_probs = np.ones(self.n_items)
        self.model_probs = np.zeros_like(self.item_probs)
        self.distribution = {}
        self.is_identity_features = ((
            np.identity(n_items) == self.features).all()
                if self.features.shape == (n_items, n_items) else False)

    def train(self, set_samples: np.ndarray):
        for sample in set_samples:
            for element in sample:
                self.item_probs[element] += 1
        self.item_probs /= len(set_samples)

        y_values = -np.log(1 / self.item_probs - 1)
        regularized = linear_model.Ridge(alpha=0.001, fit_intercept=False,
                                         normalize=False, copy_X=True)
        regularized.fit(self.features, y_values)
        self.feature_weights = regularized.coef_
        self.utilities = np.dot(self.features, self.feature_weights)
        self.logz = np.sum(np.log1p(np.exp(self.utilities)))
        self.model_probs = special.expit(self.utilities)

    def sample(self, n: int) -> np.ndarray:
        data = []
        for _ in range(n):
            s = np.nonzero(
                np.random.random_sample(self.n_items) <= self.model_probs)[0]
            data.append(s)
        return np.array(data)

    def __call__(self, item_set: Iterable[int]) -> float:
        return np.sum(self.utilities[item_set]) - self.logz

    def full_distribution(self):
        prob_sum = 0
        for length in range(1, self.n_items + 1):
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
    # for n_feats in range(10, 110, 10):
    #     for sigma in range(1, 11):
    #         sigma /= 10
    #         sigma = round(sigma, 1)
    n_items = 100
    dataset_name = constants.DATASET_NAME_TPL.format('100_no_singles')
    features = IdentityFeatures(dataset_name, n_items=n_items,
                                m_features=n_items)
    features.load_from_file()
    features_array = features.as_array()
    set_probabilities = defaultdict(list)
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
                subset = [int(item) for item in subset]
                result = modular_model.propose_set_item(subset)
                output_file.write(','.join(str(item) for item in result))
                output_file.write('\n')
    means = {}
    for subset, prob_list in set_probabilities.items():
        means[subset] = (np.mean(prob_list), np.std(prob_list))

    for subset, prob in sorted(means.items(), key=lambda x: x[1]):
        print('{}:{:.2f} +- {:.2f}%'.format(list(subset), prob[0] * 100, prob[1] * 100))

if __name__ == '__main__':
    main()
