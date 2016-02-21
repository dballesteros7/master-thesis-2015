from typing import Iterable

import numpy as np
from scipy import linalg
from scipy import special

import constants
from models.features import BasicFeatures, IdentityFeatures, \
    BasicFeaturesNoNormalized, BasicFeaturesExtended
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

    def train(self, set_samples: np.ndarray):
        for sample in set_samples:
            for element in sample:
                self.item_probs[element] += 1
        self.item_probs /= len(set_samples)

        y_values = -np.log(1 / self.item_probs - 1)
        self.feature_weights = linalg.lstsq(self.features, y_values)[0]
        self.utilities = np.dot(self.features, self.feature_weights)
        self.logz = np.sum(np.log1p(np.exp(self.utilities)))
        self.model_probs = special.expit(self.utilities)

    def sample(self, n: int) -> np.ndarray:
        data = []
        for _ in range(n):
            s = np.nonzero(
                np.random.random_sample(self.n_items) <= self.model_probs)[0]
            if np.size(s) != 0:
                data.append(s)
        return np.array(data)

    def __call__(self, item_set: Iterable[int]) -> float:
        return np.sum(self.utilities[item_set]) - self.logz

    def propose_set_item(self, to_complete: Iterable[int]) -> Iterable[int]:
        utilities = np.copy(self.utilities)
        utilities[to_complete] = -np.inf
        sorted_indexes = np.argsort(utilities)[len(to_complete):]
        return sorted_indexes[::-1]


def main():
    n_items = 9141
    dataset_name = constants.DATASET_NAME_TPL.format('cluster_features_sample_10k')
    features = BasicFeaturesNoNormalized(dataset_name,
                                         n_items=n_items,
                                         m_features=10)
    features.load_from_file()
    features_array = features.as_array()
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

if __name__ == '__main__':
    main()
