import itertools
import numpy as np

import constants
from utils import file


class MarkovChain:
    def __init__(self, n_items: int, pseudo_count: int = 1):
        self.n_items = n_items
        self.counts = np.empty(n_items)
        self.first_order_counts = np.empty((n_items, n_items))
        self.counts.fill(pseudo_count)
        self.first_order_counts.fill(pseudo_count)

    def train(self, ordered_sets: list):
        for ordered_set in ordered_sets:
            for item, next_item in itertools.zip_longest(
                    ordered_set, ordered_set[1:]):
                self.counts[item] += 1
                if next_item is not None:
                    self.first_order_counts[item][next_item] += 1

    def propose_set_item(self, to_complete):
        last_item_in_sequence = to_complete[-1]
        likelihood = self.first_order_counts[last_item_in_sequence] /\
            self.counts[last_item_in_sequence]
        for item in to_complete:
            likelihood[item] = -np.inf  # Excludes items already in the set.
        sorted_indexes = np.argsort(likelihood)  # Ascending order.
        # Returns a reversed view on the array.
        return sorted_indexes[len(to_complete):][::-1]

if __name__ == '__main__':
    for fold in range(1, constants.N_FOLDS + 1):
        chain = MarkovChain(constants.N_ITEMS)
        loaded_data = file.load_csv_data(constants.TRAIN_DATA_PATH_TPL.format(
            fold=fold, dataset=constants.DATASET_NAME))
        chain.train(loaded_data)
        loaded_test_data = file.load_csv_data(
            constants.RANKING_MODEL_PATH_TPL.format(
                fold=fold, dataset=constants.DATASET_NAME,
                model='partial'))
        target_path = constants.RANKING_MODEL_PATH_TPL.format(
            dataset=constants.DATASET_NAME, fold=fold,
            model='markov')
        with open(target_path, 'w') as output_file:
            for subset in loaded_test_data:
                result = chain.propose_set_item(subset)
                output_file.write(','.join(str(item) for item in result))
                output_file.write('\n')
