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
        self.counts.fill((n_items - 1) * pseudo_count)
        self.first_order_counts.fill(pseudo_count)
        self.use_rejection = use_rejection

        np.fill_diagonal(self.first_order_counts, 0)  # No self loops.


    def train(self, ordered_sets: np.ndarray):
        for ordered_set in ordered_sets:
            for item, next_item in itertools.zip_longest(
                    ordered_set, ordered_set[1:]):
                if next_item is not None:
                    self.counts[item] += 1
                    self.first_order_counts[item][next_item] += 1

    def propose_set_item(self, to_complete):
        missing_pos = to_complete.index('?')
        probs = np.zeros_like(self.first_order_counts)
        for idx, row in enumerate(self.first_order_counts):
            probs[idx, :] = self.first_order_counts[idx, :] / self.counts[idx]
        if missing_pos == 0:
            column = probs[:, int(to_complete[missing_pos + 1])]
            row = np.ones_like(column)
        elif missing_pos == len(to_complete) - 1:
            row = probs[:, int(to_complete[missing_pos - 1])]
            column = np.ones_like(row)
        else:
            column = probs[:, int(to_complete[missing_pos + 1])]
            row = probs[:, int(to_complete[missing_pos - 1])]
        likelihood = column*row
        to_complete = [int(x) for x in to_complete if x != '?']

        if self.use_rejection:
            likelihood[to_complete] = 0.0
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
            loaded_test_data = file.load_csv_test_data(
                constants.PARTIAL_DATA_PATH_TPL.format(
                    fold=fold, dataset=dataset_name))
            model_name = 'pseudo_markov' if use_rejection else 'markov'
            target_path = constants.RANKING_MODEL_PATH_TPL.format(
                dataset=dataset_name, fold=fold, model=model_name)
            with open(target_path, 'w') as output_file:
                for subset in loaded_test_data:
                    model.propose_set_item(subset)
                    result = model.propose_set_item(subset)
                    # if subset.index('?') > 0:
                    #     short_subset = subset[:subset.index('?')]
                    #     short_subset = [int(item) for item in short_subset]
                    #
                    output_file.write(','.join(str(item) for item in result))
                    output_file.write('\n')
                    # else:
                    #     output_file.write('-\n')

if __name__ == '__main__':
    train_and_evaluate(constants.DATASET_NAME_TPL.format('100_no_singles'), 100)
    #train_and_evaluate(constants.DATASET_NAME_TPL.format('50_no_singles'), 50)
