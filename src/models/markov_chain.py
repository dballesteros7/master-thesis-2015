import itertools
import numpy as np

import constants
from utils import file


class MarkovChain:
    def __init__(self, n_items: int, pseudo_count: int = 1,
                 use_rejection: bool = True):
        self.n_items = n_items
        self.counts = np.empty(n_items)
        self.first_order_counts = np.empty((n_items, n_items))
        self.counts.fill(pseudo_count)
        self.first_order_counts.fill(pseudo_count)
        self.use_rejection = use_rejection

        np.fill_diagonal(self.first_order_counts, 0)  # No self loops.

    def train(self, ordered_sets: np.ndarray):
        for ordered_set in ordered_sets:
            for item, next_item in itertools.zip_longest(
                    ordered_set, ordered_set[1:]):
                self.counts[item] += 1
                if next_item is not None:
                    self.first_order_counts[item][next_item] += 1

    def propose_set_item(self, to_complete: np.ndarray) -> np.ndarray:
        likelihood = self.first_order_counts[to_complete][-1]
        if self.use_rejection:
            likelihood[to_complete] = -np.inf
        sorted_indexes = np.argsort(likelihood)
        return sorted_indexes[::-1]


def train_and_evaluate(dataset_name: str, n_items: int):
    for fold in range(1, constants.N_FOLDS + 1):
        for use_rejection in (False, True):
            model = MarkovChain(n_items, use_rejection=use_rejection)
            loaded_data = file.load_set_data(
                    constants.TRAIN_DATA_PATH_TPL.format(
                            fold=fold, dataset=dataset_name))
            model.train(loaded_data)
            loaded_test_data = file.load_set_data(
                constants.PARTIAL_DATA_MARKOV_PATH_TPL.format(
                    fold=fold, dataset=dataset_name))
            model_name = 'pseudo_markov' if use_rejection else 'markov'
            target_path = constants.RANKING_MODEL_PATH_TPL.format(
                dataset=dataset_name, fold=fold, model=model_name)
            with open(target_path, 'w') as output_file:
                for subset in loaded_test_data:
                    result = model.propose_set_item(subset)
                    output_file.write(','.join(str(item) for item in result))
                    output_file.write('\n')

if __name__ == '__main__':
    train_and_evaluate(constants.DATASET_NAME_TPL.format(50), 50)
