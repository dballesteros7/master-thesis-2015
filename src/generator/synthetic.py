import os

import itertools

import math
import numpy as np

import constants
from models.general_features import GeneralFeatures
from processing.path_discovery import shuffle_train_and_test
from sampling.gibbs_sampler import GibbsSampler


def generate_three_elements():
    # model = GeneralFeatures(n_items=3, features=np.identity(3),
    #                         l_dims=1, k_dims=0)
    #
    # model.a_weights = np.array([2, 2, 2])
    # model.b_weights = np.array([0, 20, 20]).reshape((3, 1))
    #
    # model.update_composite_parameters()
    # model.full_distribution()
    # for subset, prob in model.distribution.items():
    #     print('{}:{:.2f}%'.format(list(subset), prob * 100))
    keys = [['0', '1'], ['0', '2']]
    probs = [0.5, 0.5]
    # for key, prob in model.distribution.items():
    #     keys.append([str(x) for x in key])
    #     probs.append(prob)
    #
    all_data = [keys[0]]*500 + [keys[1]]*500

    with open(os.path.join(
            constants.DATA_PATH, 'path_set_synthetic_1.csv'), 'w') as out_file:
        for sample in all_data:
            out_file.write('{}\n'.format(','.join(sample)))

    shuffle_train_and_test('synthetic_1', all_data)


def generate_four_elements():
    model = GeneralFeatures(n_items=4, features=np.identity(4),
                            l_dims=2, k_dims=2)

    model.a_weights = np.array([0, 0, 0, 0])
    model.b_weights = np.array([[10, 0], [0, 10], [10, 0], [0, 10]])
    model.c_weights = np.array([[10, 0], [10, 0], [0, 10], [0, 10]])
    #
    model.update_composite_parameters()
    model.full_distribution()
    print(distribution_error_1(model.distribution))
    return

    # for subset, prob in sorted(model.distribution.items(), key=lambda x: x[1]):
    #     print('{}:{:.2f}%'.format(list(subset), prob * 100))
    # return
    keys = [['0', '1'], ['2', '3']]
    # probs = []
    # for key, prob in model.distribution.items():
    #     keys.append([str(x) for x in key])
    #     probs.append(prob)

    all_data = [keys[0]]*500 + [keys[1]]*500

    with open(os.path.join(
            constants.DATA_PATH, 'path_set_synthetic_2.csv'), 'w') as out_file:
        for sample in all_data:
            out_file.write('{}\n'.format(','.join(sample)))

    shuffle_train_and_test('synthetic_2', all_data)


def generate_features():
    n_items = 7
    features = np.array([
        [5., 0., 1.],
        [5., 1., 0.],
        [5., 1., 1.],
        [3., 0., 1.],
        [3., 0., 0.],
        [1., 1., 1.],
        [1., 1., 0.]
    ])

    model = GeneralFeatures(n_items=n_items, features=features,
                            l_dims=1, k_dims=1)
    model.a_weights = np.array([0, 0, 0])
    model.b_weights = np.array([[0], [20], [20]])
    model.c_weights = np.array([[2], [0], [0]])
    model.update_composite_parameters()
    model.full_distribution()
    for subset, prob in sorted(model.distribution.items(), key=lambda x: x[1]):
        print('{}:{:.2f}%'.format(list(subset), prob * 100))
    keys = []
    probs = []
    for key, prob in model.distribution.items():
        keys.append([str(x) for x in key])
        probs.append(prob)

    all_data = np.random.choice(keys, 10000, True, probs)

    with open(os.path.join(
            constants.DATA_PATH, 'path_set_synthetic_3.csv'), 'w') as out_file:
        for sample in all_data:
            out_file.write('{}\n'.format(','.join(sample)))

    with open(
        constants.ITEMS_DATA_PATH_TPL.format(dataset='path_set_synthetic_3'),
            'w') as out_file:
        for item in features:
            out_file.write('{}\n'.format(','.join([str(x) for x in item])))

    shuffle_train_and_test('synthetic_3', all_data)

def generate_features_2():
    features = np.array([
        [4., 1., 0.],
        [4., 1., 1.],
        [3., 0., 1.],
        [3., 1., 0.],
        [2., 1., 1.],
        [2., 1., 0.],
        # [5., 1., 0.],
    ])
    n_items = features.shape[0]

    model = GeneralFeatures(n_items=n_items, features=features,
                            l_dims=2, k_dims=1)
    model.a_weights = np.array([0.1, 0, 0])
    model.b_weights = np.array([[0, 0], [10, 0], [0, 10]])
    model.c_weights = np.array([[0.5], [0], [0]])
    model.update_composite_parameters()
    model.full_distribution()
    print(distribution_error(model.distribution))
    # for subset, prob in sorted(model.distribution.items(), key=lambda x: x[1]):
    #     print('{}:{:.2f}%'.format(list(subset), prob * 100))
    return
    keys = [['0', '2'], ['2', '3'], ['2', '5'], ['1'], ['0'], ['2'], ['3'], ['4'], ['5']]
    probs = [0.30, 0.25, 0.15, 0.10, 0.06, 0.04, 0.04, 0.03, 0.03]

    all_data = np.random.choice(keys, 1111, True, probs)

    with open(os.path.join(
            constants.DATA_PATH, 'path_set_synthetic_4.csv'), 'w') as out_file:
        for sample in all_data:
            out_file.write('{}\n'.format(','.join(sample)))

    with open(
        constants.ITEMS_DATA_PATH_TPL.format(dataset='path_set_synthetic_4'),
            'w') as out_file:
        for item in features:
            out_file.write('{}\n'.format(','.join([str(x) for x in item])))

    shuffle_train_and_test('synthetic_4', all_data)


def main():
    np.random.seed(constants.SEED)
    #generate_three_elements()
    generate_four_elements()
    generate_features_2()

if __name__ == '__main__':
    main()
