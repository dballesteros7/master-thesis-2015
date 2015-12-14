import os
import pickle

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

import constants
from models.diversity_features import DiversityFeatures
from models.features import IdentityFeatures
from processing import ranking


def plot_weights(div_model: DiversityFeatures, d: int):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    weights = div_model.weights
    color_set = ax.matshow(weights, cmap=cm.GnBu, interpolation='none',
                           aspect='auto')
    plt.colorbar(color_set)
    ax.set_xticks(np.arange(d))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels([], visiable=False)
    ax.set_yticklabels(('Grossmünster', 'Paradeplatz',
                        'Fraumünster', 'Bellevueplatz',
                        'Zoo', 'Rathaus',
                        'Lindenhof', 'Hallenstadion',
                        'HB', 'Bürkliterrasse'))
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Locations')
    plt.savefig(os.path.join(
        constants.IMAGE_PATH, 'submodular_weights_d_{}.eps').format(d),
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
    input_model_path = constants.NCE_OUT_PATH_TPL.format(
        dataset='path_set', dim=5, fold=1, index=0)
    features = IdentityFeatures(constants.DATASET_NAME,
                                constants.N_ITEMS, constants.N_ITEMS)
    features.load_from_file()
    model = DiversityFeatures(n_items=constants.N_ITEMS,
                              features=features.as_array(), l_dims=5)
    model.load_from_file(input_model_path)
    model.update_composite_parameters()
    plot_weights(model, d=5)

if __name__ == '__main__':
    #sys.exit(plot_scores())
    main()
