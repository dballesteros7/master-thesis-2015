from typing import List
import numpy as np
from scipy import linalg

import constants
from utils import file


class ModularWithFeatures:
    """
    Model is:

    P(S) = \frac{1}{Z}\exp(\sum_{i \in S}u^{\intercal}x_{i})
    """

    def __init__(self, n_items: int, features: List[List[float]],
                 expand_features: bool = False):
        assert n_items > 0
        assert len(features) == n_items
        self.n_features = len(features[0])
        self.n_items = n_items
        self.feature_weights = np.zeros(self.n_features)
        self.features = np.array(features)
        self.logz = 1  # \log(Z)
        self.utilities = np.zeros(self.n_features)
        if expand_features:
            self.features = np.hstack(
                (self.features, np.identity(self.n_items)))
            self.n_features += self.n_items

    def train(self, set_samples: List[List[int]]):
        marginals = np.ones(self.n_items)  # Pseudo-count.
        for sample in set_samples:
            for element in sample:
                marginals[element] += 1
        marginals /= len(set_samples)  # P(i \in S)

        y_values = -np.log(1 / marginals - 1)
        self.feature_weights, _, _, _ = linalg.lstsq(self.features, y_values)
        self.logz = np.sum(np.log(1 + np.exp(
            np.dot(self.features, self.feature_weights))))
        self.utilities = np.dot(self.features, self.feature_weights)

    def __call__(self, item_set: List[int]) -> float:
        return np.sum(
            np.dot(self.features[item_set], self.feature_weights)) - self.logz

    def propose_set_item(self, to_complete: List[int]) -> List[int]:
        utilities = np.copy(self.utilities)
        utilities[to_complete] = -np.inf
        sorted_indexes = np.argsort(utilities)[len(to_complete):]
        return sorted_indexes[::-1]


def main():
    features = []
    with open(constants.ITEMS_DATA_PATH_TPL.format(
            dataset=constants.DATASET_NAME)) as input_items:
        for item in input_items:
            tokens = item.split(',')
            features.append([float(token) for token in tokens[3:]])
    if len(features) == 0:
        print('Could not load item features.')

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
        modular_model_expanded = ModularWithFeatures(
            n_items=constants.N_ITEMS, features=features, expand_features=True)
        modular_model.train(loaded_data)
        modular_model_expanded.train(loaded_data)

        target_path = constants.RANKING_MODEL_PATH_TPL.format(
            dataset=constants.DATASET_NAME, fold=fold,
            model='modular_features')
        with open(target_path, 'w') as output_file:
            for subset in loaded_test_data:
                result = modular_model.propose_set_item(subset)
                output_file.write(','.join(str(item) for item in result))
                output_file.write('\n')

        target_path = constants.RANKING_MODEL_PATH_TPL.format(
            dataset=constants.DATASET_NAME, fold=fold,
            model='modular_features_ex')
        with open(target_path, 'w') as output_file:
            for subset in loaded_test_data:
                result = modular_model_expanded.propose_set_item(subset)
                output_file.write(','.join(str(item) for item in result))
                output_file.write('\n')


if __name__ == '__main__':
    main()
