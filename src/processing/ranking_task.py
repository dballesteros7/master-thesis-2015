import copy
from itertools import combinations
from typing import Iterable

import itertools
import numpy as np

import constants
from utils import file


def save_to_csv(file_path: str, items: Iterable[Iterable[int]]):
    with open(file_path, 'w') as output_file:
        for subset in items:
            output_file.write(','.join([str(x) for x in subset]))
            output_file.write('\n')


def generate_ranking_task(dataset_name: str):
    for fold in range(1, constants.N_FOLDS + 1):
        test_data_path = constants.TEST_DATA_PATH_TPL.format(
                dataset=dataset_name, fold=fold)
        ground_truth_path = constants.GROUND_TRUTH_DATA_PATH_TPL.format(
                dataset=dataset_name, fold=fold)
        partial_path = constants.PARTIAL_DATA_PATH_TPL.format(
                dataset=dataset_name, fold=fold)

        test_data = file.load_set_data(test_data_path)

        list_gt = []
        list_partial = []
        for sample in test_data:
            if len(sample) < 2:
                continue  # Ignores single item sets.
            sample = list(sample)
            for i, item in enumerate(sample):
                sample_copy = list(sample)
                sample_copy[i] = '?'
                list_partial.append(sample_copy)
                list_gt.append([sample[i]])

        save_to_csv(ground_truth_path, list_gt)
        save_to_csv(partial_path, list_partial)


if __name__ == '__main__':
    #generate_ranking_task(dataset_name=constants.DATASET_NAME_TPL.format('cluster_features_sample_10k'))
    generate_ranking_task(dataset_name=constants.DATASET_NAME_TPL.format('synthetic_1'))
    generate_ranking_task(dataset_name=constants.DATASET_NAME_TPL.format('synthetic_2'))
    #generate_ranking_task(dataset_name=constants.DATASET_NAME_TPL.format('50_no_singles'))
