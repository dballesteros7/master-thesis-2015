import numpy as np


def load_csv_test_data(filename):
    with open(filename, 'r') as input_file:
        return [[item for item in line.strip().split(',')]
                for line in input_file]


def load_csv_data(filename:str) -> np.ndarray:
    if not filename.endswith('.csv'):
        filename += '.csv'
    with open(filename, 'r') as input_file:
        return [[int(item) for item in line.strip().split(',')]
                for line in input_file]


def load_set_data(filename: str) -> np.ndarray:
    with open(filename, 'r') as input_file:
        loaded_set = []
        for line in input_file:
            if line.strip() != '':
                tokens = line.strip().split(',')
                set_items = np.array(list(map(int, tokens)))
            else:
                set_items = []
            loaded_set.append(set_items)
        return np.array(loaded_set)
