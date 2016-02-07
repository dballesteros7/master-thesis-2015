import os

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

import constants
from models.diversity_features import DiversityFeatures
from models.features import IdentityFeatures, BasicFeatures, \
    BasicFeaturesExtended, Features
from models.general_features import GeneralFeatures
from processing import ranking


def plot_weights(div_model: GeneralFeatures, features: Features):
    fig = plt.figure()
    if div_model.l_dims > 0:
        if div_model.k_dims > 0:
            ax = fig.add_subplot(211)
        else:
            ax = fig.add_subplot(111)
        weights = (div_model.diversity_weights - np.min(div_model.diversity_weights)) / (np.max(div_model.diversity_weights) - np.min(div_model.diversity_weights))
        color_set = ax.matshow(weights, cmap=cm.GnBu, interpolation='none',
                               aspect='auto')
        plt.colorbar(color_set)
        ax.set_title('Diversity')
        ax.set_xticks(np.arange(div_model.l_dims))
        ax.set_yticks(np.arange(10))
        ax.set_xticklabels([], visible=False)
        ax.set_yticklabels(('Grossmünster', 'Paradeplatz',
                            'Fraumünster', 'Bellevueplatz',
                            'Zoo', 'Rathaus',
                            'Lindenhof', 'Hallenstadion',
                            'HB', 'Bürkliterrasse'))
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Locations')

    if div_model.k_dims > 0:
        if div_model.l_dims > 0:
            ax = fig.add_subplot(212)
        else:
            ax = fig.add_subplot(111)
        weights = (div_model.coherence_weights - np.min(div_model.coherence_weights)) / (np.max(div_model.coherence_weights) - np.min(div_model.coherence_weights))
        color_set = ax.matshow(weights, cmap=cm.GnBu, interpolation='none',
                               aspect='auto')
        plt.colorbar(color_set)
        ax.set_title('Coherence')
        ax.set_xticks(np.arange(div_model.k_dims))
        ax.set_yticks(np.arange(10))
        ax.set_xticklabels([], visible=False)
        ax.set_yticklabels(('Grossmünster', 'Paradeplatz',
                            'Fraumünster', 'Bellevueplatz',
                            'Zoo', 'Rathaus',
                            'Lindenhof', 'Hallenstadion',
                            'HB', 'Bürkliterrasse'))
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Locations')
    plt.tight_layout()
    plt.savefig(os.path.join(
        constants.IMAGE_PATH,
        'submodular_weights_f_{}_l_dim_{}_k_dim_{}.png').format(
            features.index, div_model.l_dims, div_model.k_dims),
        bbox_inches='tight')
    plt.savefig(os.path.join(
        constants.IMAGE_PATH,
        'submodular_weights_f_{}_l_dim_{}_k_dim_{}.eps').format(
            features.index, div_model.l_dims, div_model.k_dims),
        bbox_inches='tight')


def plot_scores():
    dims = np.arange(1, 51)
    acc_avg = np.zeros_like(dims)
    acc_std = np.zeros_like(dims)
    mrr_avg = np.zeros_like(dims)
    mrr_std = np.zeros_like(dims)
    for dim in range(1, 51):
        results = ranking.rank_results('submod_d_{}'.format(dim))
        acc_avg[dim - 1] = results[0]
        acc_std[dim - 1] = results[1]
        mrr_avg[dim - 1] = results[2]
        mrr_std[dim - 1] = results[3]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(dims, acc_avg, label='Average accuracy', color='blue')
    ax.plot(dims, mrr_avg, label='MRR', color='green')
    ax.fill_between(dims, acc_avg + acc_std, acc_avg - acc_std,
                    color='lightblue',
                    facecolor='lightblue', alpha=0.5)
    ax.fill_between(dims, mrr_avg + mrr_std, mrr_avg - mrr_std,
                    color='lightgreen',
                    facecolor='lightgreen', alpha=0.5)
    ax.set_title('FLID model results')
    ax.set_xlabel('Latent dimensions ($L$)')
    ax.set_ylabel('Score %')
    ax.legend(loc='upper left')
    ax.grid()
    plt.savefig(os.path.join(constants.IMAGE_PATH, 'submodular_score.eps'),
                bbox_inches='tight')


def main():
    l_dim = 2
    k_dim = 2
    features = BasicFeatures(constants.DATASET_NAME,
                             constants.N_ITEMS, 4)
    input_model_path = constants.NCE_OUT_GENERAL_PATH_TPL.format(
        dataset='path_set', l_dim=l_dim, k_dim=k_dim, fold=1,
        index=features.index)
    features.load_from_file()
    model = GeneralFeatures(n_items=constants.N_ITEMS,
                            features=features.as_array(),
                            l_dims=l_dim,
                            k_dims=k_dim)
    model.load_from_file(input_model_path)
    model.update_composite_parameters()
    plot_weights(model, features)

if __name__ == '__main__':
    main()
