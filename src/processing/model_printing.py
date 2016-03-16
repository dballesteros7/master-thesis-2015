from collections import defaultdict

import numpy as np

import constants
from models.features import BasicFeaturesNoNormalized, IdentityFeatures
from models.general_features import GeneralFeatures


def model_print(dataset_name, n_items, features, l_dim, k_dim):
    a_weights = []
    b_weights = []
    c_weights = []
    set_probabilities = defaultdict(list)
    for fold in range(1, constants.N_FOLDS + 1):
        model = GeneralFeatures(n_items, features.as_array(), l_dim, k_dim)
        model.load_from_file(constants.NCE_OUT_GENERAL_PATH_TPL.format(
            dataset=dataset_name, fold=fold, l_dim=l_dim, k_dim=k_dim,
            index=features.index))
        a_weights.append(model.a_weights)
        b_weights.append(
            np.array(model.b_weights).reshape(features.m_features*l_dim))
        c_weights.append(
            np.array(model.c_weights).reshape(features.m_features*k_dim))
        model.full_distribution()
        for subset, prob in sorted(
                model.distribution.items(), key=lambda x: x[1]):
            set_probabilities[subset].append(prob)

    a_weights = np.array(a_weights)
    b_weights = np.array(b_weights)
    c_weights = np.array(c_weights)

    a_weights_mean = np.mean(a_weights, axis=0)
    b_weights_mean = np.mean(b_weights, axis=0).reshape(
        (features.m_features, l_dim))
    c_weights_mean = np.mean(c_weights, axis=0).reshape(
        (features.m_features, k_dim))

    print(a_weights_mean)
    print(b_weights_mean)
    print(c_weights_mean)

    a_weights_std = np.std(a_weights, axis=0)
    b_weights_std = np.std(b_weights, axis=0).reshape(
        (features.m_features, l_dim))
    c_weights_std = np.std(c_weights, axis=0).reshape(
        (features.m_features, k_dim))

    print(a_weights_std)
    print(b_weights_std)
    print(c_weights_std)

    means = {}
    for subset, prob_list in set_probabilities.items():
        means[subset] = (np.mean(prob_list), np.std(prob_list))

    for subset, prob in sorted(means.items(), key=lambda x: x[1]):
        print('{}:{:.2f} +- {:.2f}%'.format(list(subset), prob[0] * 100, prob[1] * 100))


def main():
    dataset_name = 'path_set_synthetic_4'
    n_items = 7
    m_features = 3
    features = BasicFeaturesNoNormalized(dataset_name, n_items=n_items,
                                         m_features=3)
    features.load_from_file()
    model_print(dataset_name, n_items, features, 2, 1)

if __name__ == '__main__':
    main()
