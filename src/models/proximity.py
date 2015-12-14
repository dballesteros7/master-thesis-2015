from typing import Iterable, Dict

import numpy as np
from geopy.distance import great_circle
import constants
from utils import file


class Proximity:
    def __init__(self, n_items: int):
        self.n_items = n_items
        self.distances = np.zeros(shape=(n_items, n_items))
        self.distance_calculator = great_circle()

    def train(self, items: Iterable[Dict]):
        for index, item in enumerate(items):
            for other_index, other_item in enumerate(items[index + 1:]):
                distance = self.distance_calculator.measure(
                    (item['latitude'], item['longitude']),
                    (other_item['latitude'], other_item['longitude']))
                self.distances[index][other_index + index + 1] = distance
                self.distances[other_index + index + 1][index] = distance

    def propose_set_item(self, to_complete: np.ndarray) -> np.ndarray:
        last_item = to_complete[-1]
        distances = np.copy(self.distances[last_item])
        distances[to_complete] = np.inf
        sorted_indexes = np.argsort(distances)
        return sorted_indexes


def train_and_evaluate(dataset_name: str, n_items: int):
    model = Proximity(n_items)
    items = []
    with open(constants.ITEMS_DATA_PATH_TPL.format(
            dataset=dataset_name), 'r') as items_file:
        for line in items_file:
            tokens = line.strip().split(',')
            items.append({
                'latitude': float(tokens[1]),
                'longitude': float(tokens[2])
            })
    model.train(items)
    for fold in range(1, constants.N_FOLDS + 1):
        loaded_test_data = file.load_set_data(
                constants.PARTIAL_DATA_PATH_TPL.format(
                    fold=fold, dataset=dataset_name))
        target_path = constants.RANKING_MODEL_PATH_TPL.format(
            dataset=dataset_name, fold=fold,
            model='proximity')
        with open(target_path, 'w') as output_file:
            for subset in loaded_test_data:
                result = model.propose_set_item(subset)
                output_file.write(','.join(str(item) for item in result))
                output_file.write('\n')


if __name__ == '__main__':
    train_and_evaluate(constants.DATASET_NAME, constants.N_ITEMS)
