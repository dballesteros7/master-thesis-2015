import numpy as np
import matplotlib.pyplot as plt

import constants
from models.features import IdentityFeatures
from models.general_features import GeneralFeatures
from models.modular import ModularWithFeatures
from utils import file


def load_submodular_model(name):
    n_items = 50
    l_dim = 20
    k_dim = 20
    fold = 1
    dataset_name = constants.DATASET_NAME_TPL.format(name)
    features = IdentityFeatures(dataset_name,
                                n_items=n_items,
                                m_features=n_items)
    features.load_from_file()
    model = GeneralFeatures(n_items, features.as_array(), l_dim, k_dim)
    model.load_from_file(constants.NCE_OUT_GENERAL_PATH_TPL.format(
        dataset=dataset_name, fold=fold, l_dim=l_dim, k_dim=k_dim,
        index=features.index))
    return model


def load_modular(name):
    n_items = 50
    fold = 1
    dataset_name = constants.DATASET_NAME_TPL.format(name)
    features = IdentityFeatures(dataset_name,
                                n_items=n_items,
                                m_features=n_items)
    features.load_from_file()
    features_array = features.as_array()
    loaded_data = file.load_csv_data(
        constants.TRAIN_DATA_PATH_TPL.format(
            fold=fold, dataset=dataset_name))
    modular_model = ModularWithFeatures(
        n_items=n_items, features=features_array)
    modular_model.train(loaded_data)
    return modular_model


def plot_models(n_items, modular_model, submodular_model):
    width = 0.3
    x_values = np.arange(n_items)
    fig, ax = plt.subplots()
    exp_modular_utilities = (modular_model.utilities)
    exp_submodular_utilities = (submodular_model.utilities)
    max_utility = np.max(
        [np.max(exp_modular_utilities), np.max(exp_submodular_utilities)])
    min_utility = np.min(
        [np.min(exp_modular_utilities), np.min(exp_submodular_utilities)])
    normalized_modular_utilities = (exp_modular_utilities - min_utility) / (max_utility - min_utility)
    normalized_submodular_utilities = (exp_submodular_utilities - min_utility) / (max_utility - min_utility)
    rects1 = ax.bar(x_values, normalized_modular_utilities, color='b',
                    width=width)
    rects2 = ax.bar(x_values + width, normalized_submodular_utilities,
                    color='g',
                    width=width)
    ax.set_xlabel('$i$')
    ax.set_ylabel('$u_{i}$')
    ax.set_title('Item utility weights')
    ax.set_xticks(x_values + (3 * width / 2))
    ax.set_xticklabels(x_values)
    ax.legend((rects1[0], rects2[0]),
              ('Modular', 'FLDC (L=20, K=20)'),
              loc='upper right')
    plt.show()


def main():
    name = '50_no_singles'
    modular_model = load_modular(name)
    submodular_model = load_submodular_model(name)
    plot_models(50, modular_model, submodular_model)


if __name__ == '__main__':
    main()
