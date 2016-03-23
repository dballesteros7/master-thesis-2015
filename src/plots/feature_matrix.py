import os

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

import constants
from models.features import GaussianFeatures, Features, GaussianExtended


def plot_matrix(features: Features):
    cmap = mpl.colors.ListedColormap(sns.color_palette('RdBu_r', 10))
    fig, ax = plt.subplots()
    im = ax.matshow(features.as_array(),
                aspect=features.m_features / features.n_items, cmap=cmap)
    ax.grid(False)
    ax.set_xticks(np.arange(features.m_features, step=10))
    ax.set_yticks(np.arange(features.n_items, step=10))
    ax.set_xlabel('Feature')
    ax.set_ylabel('Item')
    ax.set_title('Gaussian features')
    plt.colorbar(mappable=im)
    plt.savefig(os.path.join(
      constants.IMAGE_PATH, 'gaussian_features_n_100_0dot15.eps'),
      bbox_inches='tight')
    plt.show()


def main():
    features = GaussianExtended('path_set_100_no_singles',
                                n_items=100, m_features=10, sigma=0.1)
    features.load_from_file()
    plot_matrix(features)


if __name__ == '__main__':
    main()
