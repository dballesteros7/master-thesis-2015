from typing import List

import numpy as np
from scipy import linalg
from scipy import special

import constants
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
        self.item_probs /= len(set_samples)  # P(i \in S)

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

    def __call__(self, item_set: List[int]) -> float:
        return np.sum(self.utilities[item_set]) - self.logz

    def propose_set_item(self, to_complete: List[int]) -> List[int]:
        utilities = np.copy(self.utilities)
        utilities[to_complete] = -np.inf
        sorted_indexes = np.argsort(utilities)[len(to_complete):]
        return sorted_indexes[::-1]


def main():
    features = np.identity(constants.N_ITEMS)

    for fold in range(1, constants.N_FOLDS + 1):
        loaded_data = file.load_csv_data(
            constants.TRAIN_DATA_PATH_TPL.format(
                fold=fold, dataset=constants.DATASET_NAME))
        loaded_test_data = file.load_csv_data(
            constants.RANKING_MODEL_PATH_TPL.format(
                fold=fold, dataset=constants.DATASET_NAME,
                model='partial'))

        modular_model = ModularWithFeatures(
            n_items=constants.N_ITEMS, features=features)
        modular_model.train(loaded_data)

        target_path = constants.RANKING_MODEL_PATH_TPL.format(
            dataset=constants.DATASET_NAME, fold=fold,
            model='modular_features')
        with open(target_path, 'w') as output_file:
            for subset in loaded_test_data:
                result = modular_model.propose_set_item(subset)
                output_file.write(','.join(str(item) for item in result))
                output_file.write('\n')

if __name__ == '__main__':
    main()
