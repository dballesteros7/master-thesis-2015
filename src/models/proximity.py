from typing import Iterable, Dict

import numpy as np
from geopy.distance import great_circle
import constants
from utils import file


class Proximity:
    def __init__(self, n_items: int, use_rejection: bool):
        self.n_items = n_items
        self.distances = np.zeros(shape=(n_items, n_items))
        self.distance_calculator = great_circle()
        self.use_rejection = use_rejection

    def train(self, items: Iterable[Dict]):
        for index, item in enumerate(items):
            for other_index, other_item in enumerate(items[index + 1:]):
                distance = self.distance_calculator.measure(
                    (item['latitude'], item['longitude']),
                    (other_item['latitude'], other_item['longitude']))
                self.distances[index][other_index + index + 1] = distance
                self.distances[other_index + index + 1][index] = distance
        np.fill_diagonal(self.distances, np.inf)

    def propose_set_item(self, to_complete: np.ndarray) -> np.ndarray:
        min_distances = np.min(self.distances[to_complete, :], axis=0)
        if self.use_rejection:
            min_distances[to_complete] = np.inf
        sorted_indexes = np.argsort(min_distances)
        return sorted_indexes


def train_and_evaluate(dataset_name: str, n_items: int):
    for rejection in [False, True]:
        model = Proximity(n_items, use_rejection=rejection)
        items = []
        with open(constants.ITEMS_DATA_PATH_TPL.format(
                dataset=dataset_name), 'r') as items_file:
            for line in items_file:
                tokens = line.strip().split(',')
                items.append({
                    'latitude': float(tokens[0]),
                    'longitude': float(tokens[1])
                })
        model.train(items)
        for fold in range(1, constants.N_FOLDS + 1):
            loaded_test_data = file.load_set_data(
                    constants.PARTIAL_DATA_PATH_TPL.format(
                        fold=fold, dataset=dataset_name))
            model_name = 'proximity_r' if rejection else 'proximity'
            target_path = constants.RANKING_MODEL_PATH_TPL.format(
                dataset=dataset_name, fold=fold,
                model=model_name)
            with open(target_path, 'w') as output_file:
                for subset in loaded_test_data:
                    result = model.propose_set_item(subset)
                    output_file.write(','.join(str(item) for item in result))
                    output_file.write('\n')


if __name__ == '__main__':
    train_and_evaluate(constants.DATASET_NAME_TPL.format(50), 50)
