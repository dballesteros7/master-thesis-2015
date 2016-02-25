import os

import numpy as np

import constants
from models.general_features import GeneralFeatures
from processing.path_discovery import shuffle_train_and_test
from sampling.gibbs_sampler import GibbsSampler


def generate_three_elements():
    model = GeneralFeatures(n_items=3, features=np.identity(3),
                            l_dims=1, k_dims=0)

    model.a_weights = np.array([2, 1, 1])
    model.b_weights = np.array([0, 100, 100]).reshape((3, 1))

    model.update_composite_parameters()

    sampler = GibbsSampler(3, model)
    sampler.train(100000)
    for subset, count in sampler.counts.items():
        print('{}:{:.2f}%'.format(list(subset), count * 100 / sampler.samples))
    all_data = []

    for subset, count in sampler.counts.items():
        if len(subset) == 0:
            continue
        for _ in range(count):
            all_data.append(list(str(i) for i in subset))

    with open(os.path.join(
            constants.DATA_PATH, 'path_set_synthetic_1.csv'), 'w') as out_file:
        for sample in all_data:
            out_file.write('{}\n'.format(','.join(sample)))

    shuffle_train_and_test('synthetic_1', all_data)


def generate_four_elements():
    model = GeneralFeatures(n_items=4, features=np.identity(4),
                            l_dims=4, k_dims=0)

    model.a_weights = np.array([1, 1, 1, 1])
    model.b_weights = np.array([[100, 100, 0, 0], [0, 0, 100, 100], [100, 0, 100, 0], [0, 100, 0, 100]])
    model.c_weights = np.array([[5, 0], [5, 0], [0, 5], [0, 5]])

    model.update_composite_parameters()

    sampler = GibbsSampler(4, model)
    sampler.train(100000)
    for subset, count in sampler.counts.items():
        print('{}:{:.2f}%'.format(list(subset), count * 100 / sampler.samples))
    all_data = []

    for subset, count in sampler.counts.items():
        if len(subset) == 0:
            continue
        for _ in range(count):
            all_data.append(list(str(i) for i in subset))

    with open(os.path.join(
            constants.DATA_PATH, 'path_set_synthetic_2.csv'), 'w') as out_file:
        for sample in all_data:
            out_file.write('{}\n'.format(','.join(sample)))

    shuffle_train_and_test('synthetic_2', all_data)

def main():
    np.random.seed(constants.SEED)
    generate_four_elements()

if __name__ == '__main__':
    main()
