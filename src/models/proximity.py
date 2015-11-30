import numpy as np
from geopy.distance import great_circle
import constants
from utils import file


class Proximity:
    def __init__(self, n_items):
        self.n_items = n_items
        self.distances = np.zeros((n_items, n_items))
        self.distance_calculator = great_circle()

    def calculate_distances(self, items):
        for index, item in enumerate(items):
            for other_index, other_item in enumerate(items[index + 1:]):
                distance = self.distance_calculator.measure(
                    (item['latitude'], item['longitude']),
                    (other_item['latitude'], other_item['longitude']))
                self.distances[index][other_index + index + 1] = distance
                self.distances[other_index + index + 1][index] = distance

    def propose_set_item(self, to_complete):
        last_item = to_complete[-1]
        distances = np.copy(self.distances[last_item])
        for item in to_complete:
            distances[item] = np.inf  # Excludes items already in the set.
        sorted_indexes = np.argsort(distances)  # Ascending order.
        # Returns a reversed view on the array.
        return sorted_indexes[:-len(to_complete)][::-1]


def main():
    model = Proximity(constants.N_ITEMS)
    items = []
    with open(constants.ITEMS_DATA_PATH_TPL.format(
            dataset='path_set'), 'r') as items_file:
        for line in items_file:
            tokens = line.strip().split(',')
            items.append({
                'latitude': float(tokens[1]),
                'longitude': float(tokens[2])
            })
    model.calculate_distances(items)
    for fold in range(1, constants.N_FOLDS + 1):
        loaded_test_data = file.load_csv_data(
            constants.RANKING_MODEL_PATH_TPL.format(
                fold=fold, dataset=constants.DATASET_NAME,
                model='partial'))
        target_path = constants.RANKING_MODEL_PATH_TPL.format(
            dataset=constants.DATASET_NAME, fold=fold,
            model='proximity')
        with open(target_path, 'w') as output_file:
            for subset in loaded_test_data:
                result = model.propose_set_item(subset)
                output_file.write(','.join(str(item) for item in result))
                output_file.write('\n')


if __name__ == '__main__':
    main()
