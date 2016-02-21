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


def generate_ranking_task_for_markov(dataset_name: str):
    for fold in range(1, constants.N_FOLDS + 1):
        test_data_path = constants.TEST_DATA_PATH_TPL.format(
                dataset=dataset_name, fold=fold)
        ground_truth_path = constants.GROUND_TRUTH_MARKOV_DATA_PATH_TPL.format(
                dataset=dataset_name, fold=fold)
        partial_path = constants.PARTIAL_DATA_MARKOV_PATH_TPL.format(
                dataset=dataset_name, fold=fold)

        test_data = file.load_set_data(test_data_path)

        list_gt = []
        list_partial = []
        for i, sample in enumerate(test_data):
            if len(sample) < 2:
                continue  # Ignores single item sets.

            for item, next_item in itertools.zip_longest(sample, sample[1:]):
                if next_item is not None:
                    list_gt.append([next_item])
                    partial = list(set(sample) - {item, next_item})
                    partial.append(item)
                    list_partial.append(partial)

        save_to_csv(ground_truth_path, list_gt)
        save_to_csv(partial_path, list_partial)


if __name__ == '__main__':
    generate_ranking_task(dataset_name=constants.DATASET_NAME_TPL.format('cluster_features_sample_10k'))
    generate_ranking_task(dataset_name=constants.DATASET_NAME_TPL.format('10'))
    generate_ranking_task(dataset_name=constants.DATASET_NAME_TPL.format('10_no_singles'))
