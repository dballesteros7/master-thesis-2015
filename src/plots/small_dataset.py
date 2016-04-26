import os
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import constants
import plots
from models.features import IdentityFeatures, DescriptiveFeatures
from models.general_features import GeneralFeatures
from models.modular import ModularWithFeatures
from processing.ranking import rank_results_pandas
from utils import file


def scores_baselines():
    dataset_name = 'path_set_10_no_singles'
    modular_results = rank_results_pandas(dataset_name, 'modular_features_0', 0)
    markov_results = rank_results_pandas(dataset_name, 'markov', 0)
    pseudo_markov_results = rank_results_pandas(dataset_name, 'pseudo_markov', 0)
    proximity_results = rank_results_pandas(dataset_name, 'proximity', 0)
    proximity_ordered_results = rank_results_pandas(dataset_name, 'proximity_ordered', 0)

    results_column = np.concatenate((modular_results, markov_results, pseudo_markov_results, proximity_results, proximity_ordered_results))
    model_column = np.repeat(['Log-modular', 'Markov', 'Heuristic Markov', 'Proximity', 'Proximity Ordered'], constants.N_FOLDS)

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

def log_likelihood_flid():
    l_range = [0]
    k_range = [10]

    l_column = []
    k_column = []
    ll_column = []
    ll_2_column = []
    dataset_name = 'path_set_10_no_singles'

    features = IdentityFeatures(dataset_name, 10, 10)
    features.load_from_file()
    for fold in range(1, constants.N_FOLDS + 1):
        loaded_test_data = file.load_csv_test_data(
                    constants.TEST_DATA_PATH_TPL.format(
                        fold=fold, dataset=dataset_name))
        modular = ModularWithFeatures(10, features.as_array())
        modular.train(file.load_csv_test_data(
            constants.TEST_DATA_PATH_TPL.format(
                fold=fold, dataset=dataset_name)))
        ll_modular = modular.log_likelihood(loaded_test_data)
        other_modular = ModularWithFeatures(10, np.identity(10))
        other_modular.train(file.load_csv_test_data(
            constants.TEST_DATA_PATH_TPL.format(
                fold=fold, dataset=dataset_name)))
        other_ll_modular = modular.log_likelihood(loaded_test_data)
        for l_dim in l_range:
            for k_dim in k_range:
                model = GeneralFeatures(10, features.as_array(), l_dim, k_dim)
                model.load_from_file(constants.NCE_OUT_GENERAL_PATH_TPL.format(
                    dataset=dataset_name, fold=fold, l_dim=l_dim, k_dim=k_dim,
                    index=features.index, iter=1000,
                    noise=5, eta_0=1, adagrad=1))

                ll = model.log_likelihood(loaded_test_data)
                l_column.append(l_dim)
                k_column.append(k_dim)
                llri = 100*(ll - ll_modular) / abs(ll_modular)
                llri_2 = 100*(ll - other_ll_modular) / abs(other_ll_modular)
                ll_column.append(llri)
                ll_2_column.append(llri_2)

    dataset = pd.DataFrame({
        'll': ll_column,
        'll2': ll_2_column,
        'l': l_column,
        'k': k_column
    })
    mean = dataset.groupby(['l', 'k'])['ll2'].mean().unstack(1)
    std = dataset.groupby(['l', 'k'])['ll2'].std().unstack(1)
    print(mean)
    print(std)
    return
    ax = sns.heatmap(mean, vmin=0, vmax=5, annot=True, fmt='.2f',
                     linewidths=.5)
    ax.set_xlabel('$K$')
    ax.set_ylabel('$L$')
    #ax.set_ylabel(r'Accuracy (\%)')
    ax.set_title('LLRI')
    #ax.set_ylim([0, 45])

    # modular_mean = 100*np.mean(rank_results_pandas(dataset_name, 'modular_features_0', 0))
    # plt.plot(ax.get_xlim(), [modular_mean, modular_mean], linestyle='dotted')
    #
    # plt.savefig(os.path.join(
    #     constants.IMAGE_PATH, 'flid_10_llri.eps'),
    #     bbox_inches='tight')

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
    result = dataset.groupby(['l', 'k'])['scores'].mean()
    print(dataset.groupby(['l', 'k'])['scores'].std())
    dataset = result.unstack(1)

    cmap = sns.cubehelix_palette(8, start=1.8, light=0.8, as_cmap=True)
    ax = sns.heatmap(dataset, cmap=cmap, vmin=0, vmax=0.5, linewidths=.5,
                     annot=True, fmt='.2f')
    # ax = sns.barplot(x='l', y='scores', hue='k', data=dataset, ci=95,
    #                  palette=sns.color_palette('Set1'))
    ax.set_xlabel('$K$')
    ax.set_ylabel(r'$L$')
    ax.set_title(r'$\|P_{d} - \hat{P}_{d}\|_{TV}$')

    # legend = ax.get_legend()
    # legend.set_title('$K$')
    #
    # plt.plot(ax.get_xlim(), [baseline, baseline], linestyle='dotted')
    plt.savefig(os.path.join(
        constants.IMAGE_PATH, 'fldc_10_l_k_dims_tv.eps'),
        bbox_inches='tight')
    plt.show()

def log_likelihood_features():
    ll_column = []
    type_column = []
    dataset_name = 'path_set_10_no_singles'

    features = DescriptiveFeatures(dataset_name)
    features.load_from_file()
    for fold in range(1, constants.N_FOLDS + 1):
        loaded_test_data = file.load_csv_test_data(
                    constants.TEST_DATA_PATH_TPL.format(
                        fold=fold, dataset=dataset_name))
        modular_no_features = ModularWithFeatures(10, np.identity(10))
        modular_no_features.train(file.load_csv_test_data(
            constants.TRAIN_DATA_PATH_TPL.format(
                fold=fold, dataset=dataset_name)))
        ll_column.append(modular_no_features.log_likelihood(loaded_test_data))
        type_column.append('Modular')
        modular = ModularWithFeatures(10, features.as_array())
        modular.train(file.load_csv_test_data(
            constants.TRAIN_DATA_PATH_TPL.format(
                fold=fold, dataset=dataset_name)))
        ll_modular = modular.log_likelihood(loaded_test_data)
        ll_column.append(ll_modular)
        type_column.append('Modular with features')
        model = GeneralFeatures(10, features.as_array(), 5, 5)
        model.load_from_file(constants.NCE_OUT_GENERAL_PATH_TPL.format(
            dataset=dataset_name, fold=fold, l_dim=5, k_dim=5,
            index=features.index, iter=1000,
            noise=5, eta_0=1, adagrad=1))

        ll = model.log_likelihood(loaded_test_data)
        llri = 100*(ll - ll_modular) / abs(ll_modular)
        llri = ll
        ll_column.append(llri)
        type_column.append('FFLDC (Features)')
        model = GeneralFeatures(10, np.identity(10), 5, 5)
        model.load_from_file(constants.NCE_OUT_GENERAL_PATH_TPL.format(
            dataset=dataset_name, fold=fold, l_dim=5, k_dim=5,
            index='0', iter=1000,
            noise=5, eta_0=1, adagrad=1))

        ll = model.log_likelihood(loaded_test_data)
        llri = 100*(ll - ll_modular) / abs(ll_modular)
        llri = ll
        ll_column.append(llri)
        type_column.append('FLDC (No features)')

    dataset = pd.DataFrame({
        'll': -np.array(ll_column),
        'type': type_column
    })
    ax = sns.barplot(x='type', y='ll', data=dataset)
    ax.set_xlabel('Model')
    ax.set_ylabel('')
    #ax.set_ylabel(r'Accuracy (\%)')
    ax.set_title('Negative log-likelihood for Different Models')

    # modular_mean = 100*np.mean(rank_results_pandas(dataset_name, 'modular_features_0', 0))
    # plt.plot(ax.get_xlim(), [modular_mean, modular_mean], linestyle='dotted')
    #
    plt.savefig(os.path.join(
        constants.IMAGE_PATH, 'ffldc_10_nlog.eps'),
        bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    plots.setup()
    sns.set_palette(sns.color_palette('Set1'))
    #scores_baselines()
    #score_flids_varying_l()
    #score_flids_varying_l_k()
    #total_distance()
    #log_likelihood_flid()
    log_likelihood_features()
