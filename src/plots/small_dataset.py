import os
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import constants
import plots
from models.features import IdentityFeatures
from models.general_features import GeneralFeatures
from models.modular import ModularWithFeatures
from processing.ranking import rank_results_pandas
from utils import file


def scores_baselines():
    modular_results = rank_results_pandas('path_set_10_no_singles', 'modular_features_0', 0)
    markov_results = rank_results_pandas('path_set_10_no_singles', 'markov', 0)
    pseudo_markov_results = rank_results_pandas('path_set_10_no_singles', 'pseudo_markov', 0)
    proximity_results = rank_results_pandas('path_set_10_no_singles', 'proximity_r', 0)
    #flid_results = rank_results_pandas('path_set_10_no_singles', 'submod_f_0_l_10_k_10_iter_1000_noise_5_eta_1_adagrad_1', 0)

    results_column = np.concatenate((modular_results, pseudo_markov_results, pseudo_markov_results, proximity_results))
    model_column = np.repeat(['Log-modular', 'Markov', 'Heuristic Markov', 'Proximity'], constants.N_FOLDS)

    dataset = pd.DataFrame({
        'scores': 100 * results_column,
        'model': model_column
    })

    ax = sns.barplot(x='model', y='scores', data=dataset, ci=95,
                     palette=sns.color_palette('Set1'))
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy (\%)')
    ax.set_title('Accuracy of baseline models')

    plt.savefig(os.path.join(
       constants.IMAGE_PATH, 'baseline_models_10.eps'),
       bbox_inches='tight')
    plt.show()

def score_flids_varying_l():
    l_range = np.arange(1, 11, 1)

    l_column = np.repeat(l_range, constants.N_FOLDS)

    dataset_name = 'path_set_10_no_singles'
    model_template = 'submod_f_0_l_{}_k_0_iter_1000_noise_5_eta_1_adagrad_1'

    results_column = []

    for l_dim in l_range:
        results = rank_results_pandas(dataset_name, model_template.format(l_dim), 0)
        results_column.extend(results)

    dataset = pd.DataFrame({
        'scores': 100 *np.array(results_column),
        'l': l_column
    })

    ax = sns.pointplot(x='l', y='scores', data=dataset, ci=95)
    ax.set_xlabel('$L$')
    ax.set_ylabel(r'Accuracy (\%)')
    ax.set_title('Accuracy vs number of diversity dimensions')
    ax.set_ylim([0, 45])

    modular_mean = 100*np.mean(rank_results_pandas(dataset_name, 'modular_features_0', 0))
    plt.plot(ax.get_xlim(), [modular_mean, modular_mean], linestyle='dotted')

    plt.savefig(os.path.join(
        constants.IMAGE_PATH, 'flid_10_l_dims.eps'),
        bbox_inches='tight')

    plt.show()


def score_flids_varying_l_k():
    l_range = np.arange(0, 11, 2)
    k_range = np.arange(2, 11, 2)

    l_column = []
    k_column = []

    dataset_name = 'path_set_10_no_singles'
    model_template = 'submod_f_0_l_{}_k_{}_iter_1000_noise_5_eta_1_adagrad_1'

    results_column = []

    for l_dim in l_range:
        for k_dim in k_range:
            l_column += [l_dim]*constants.N_FOLDS
            k_column += [k_dim]*constants.N_FOLDS
            results = rank_results_pandas(
                dataset_name, model_template.format(l_dim, k_dim), 0)
            results_column.extend(results)

    dataset = pd.DataFrame({
        'scores': 100 *np.array(results_column),
        'l': l_column,
        'k': k_column
    })

    ax = sns.pointplot(x='l', y='scores', hue='k', data=dataset, ci=95,
                       palette=sns.color_palette('Set1'))
    ax.set_xlabel('$L$')
    ax.set_ylabel(r'Accuracy (\%)')
    ax.set_title('Accuracy vs number of dimensions')
    ax.set_ylim([0, 40])

    legend = ax.get_legend()
    legend.set_title('$K$')

    modular_mean = 100*np.mean(rank_results_pandas(dataset_name, 'modular_features_0', 0))
    plt.plot(ax.get_xlim(), [modular_mean, modular_mean], linestyle='dotted')

    plt.savefig(os.path.join(
        constants.IMAGE_PATH, 'fldc_10_l_k_dims.eps'),
        bbox_inches='tight')

    plt.show()


def compare_dists(dist_a, dist_b):
    max_sum = 0
    for subset in dist_a:
        other_prob = dist_a[subset]
        abs_err = abs(other_prob - dist_b[subset])
        max_sum += abs_err
    return max_sum/2


def total_distance():
    n_items = 10
    dataset_name = constants.DATASET_NAME_TPL.format('10_no_singles')

    main_distribution = defaultdict(int)
    all_data = file.load_set_data(constants.ALL_DATA_PATH_TPL.format(dataset=dataset_name))
    for subset in all_data:
        main_distribution[frozenset(subset)] += 1
    for subset in main_distribution:
        main_distribution[subset] /= len(all_data)

    features = IdentityFeatures(dataset_name, n_items=n_items,
                                m_features=n_items)
    features.load_from_file()
    loaded_data = file.load_csv_data(
            constants.TRAIN_DATA_PATH_TPL.format(
                fold=1, dataset=dataset_name))
    modular = ModularWithFeatures(n_items, features.as_array())
    modular.train(loaded_data)
    modular.full_distribution()
    score_column = []
    l_dim_column = []
    k_dim_column = []
    for l_dim in np.arange(0, 11, 5):
        for k_dim in np.arange(0, 11, 5):
            for fold in range(1, constants.N_FOLDS + 1):
                model = GeneralFeatures(n_items,
                                        features.as_array(), l_dim, k_dim)
                model.load_from_file(constants.NCE_OUT_GENERAL_PATH_TPL.format(
                    dataset=dataset_name, fold=fold, l_dim=l_dim, k_dim=k_dim,
                    index=features.index, iter=1000,
                    noise=5, eta_0=1, adagrad=1))
                model.full_distribution()
                score_column.append(compare_dists(
                    model.distribution, main_distribution))
                l_dim_column.append(l_dim)
                k_dim_column.append(k_dim)
    baseline = compare_dists(modular.distribution, main_distribution)

    dataset = pd.DataFrame({
        'scores': np.array(score_column),
        'l': l_dim_column,
        'k': k_dim_column
    })

    ax = sns.barplot(x='l', y='scores', hue='k', data=dataset, ci=95,
                     palette=sns.color_palette('Set1'))
    ax.set_xlabel('$L$')
    ax.set_ylabel(r'$\|P_{d} - \hat{P}_{d}\|_{TV}$')
    ax.set_title('Total variation distance for FLDC models.')

    legend = ax.get_legend()
    legend.set_title('$K$')

    plt.plot(ax.get_xlim(), [baseline, baseline], linestyle='dotted')
    plt.savefig(os.path.join(
        constants.IMAGE_PATH, 'fldc_10_l_k_dims_tv.eps'),
        bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    plots.setup()
    sns.set_palette(sns.color_palette('Set1'))
    #scores_baselines()
    score_flids_varying_l()
    #score_flids_varying_l_k()
    #total_distance()
