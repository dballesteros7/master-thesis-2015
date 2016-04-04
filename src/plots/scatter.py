import numpy as np
import pandas as pd

import constants
from models.features import GaussianExtended
from models.modular import ModularWithFeatures
from utils import file

import matplotlib.pyplot as plt
import seaborn as sns


def plot_scatter():
    n_items = 100
    dataset_name = constants.DATASET_NAME_TPL.format('100_no_singles')
    f = GaussianExtended('path_set_100_no_singles', 100, 100, 0.05)
    f.load_from_file()
    loaded_data = file.load_csv_data(
            constants.TRAIN_DATA_PATH_TPL.format(
                fold=1, dataset=dataset_name))
    modular_model = ModularWithFeatures(
        n_items=n_items, features=np.identity(n_items))
    modular_model.train(loaded_data)
    data = []
    for photo_count, item_prob in zip(f.as_array()[:,-5], modular_model.item_probs):
        data.append((photo_count, item_prob))
    dataset = pd.DataFrame(data, columns=['users_per_photo', 'item_marginal'])
    sns.lmplot(x='users_per_photo', y='item_marginal', data=dataset)
    plt.show()

if __name__ == '__main__':
    plot_scatter()
