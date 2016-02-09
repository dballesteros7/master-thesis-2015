from collections import defaultdict

import numpy as np

import constants
from utils import file


class Frequency:
    def __init__(self):
        self.counts = defaultdict(int)
        self.sorted_counts = []

    def train(self, set_samples: np.ndarray):
        for sample in set_samples:
            item_set = frozenset(sample)
            self.counts[item_set] += 1

        for item_set in self.counts:
            self.sorted_counts.append((item_set, self.counts[item_set]))

        self.sorted_counts.sort(key=lambda x: x[1], reverse=True)

    def print(self):
        printed = 0
        for entry in self.sorted_counts:
            if len(entry[0]) > 1:
                printed += 1
                print('${}$ & {}\\\\'.format(list(entry[0]), entry[1]))
            if printed > 10:
                return

def main():
    loaded_data = file.load_csv_data(
        constants.ALL_DATA_PATH_TPL.format(
            dataset=constants.DATASET_NAME_TPL.format(50)))
    modular_model = Frequency()
    modular_model.train(loaded_data)
    modular_model.print()


if __name__ == '__main__':
    main()
