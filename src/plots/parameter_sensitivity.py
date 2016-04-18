import itertools

import math
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import constants
import plots
from models.features import BasicFeaturesNoNormalized, IdentityFeatures
from models.general_features import GeneralFeatures
from models.modular import ModularWithFeatures
from utils import file


def distribution_error_1(other_distribution):
    probabilities = {}
    for size in range(5):
        for subset in itertools.combinations(range(4), size):
            probabilities[frozenset(subset)] = 0.0

    probabilities[frozenset({0, 1})] = 0.5
    probabilities[frozenset({2, 3})] = 0.5

    acc_err = 0.0
    max_err = -1
    max_sum = 0.0
    for subset in probabilities:
        other_prob = other_distribution[subset]
        abs_err = abs(other_prob - probabilities[subset])
        max_sum += abs_err
        if abs_err > max_err:
            max_err = abs_err
        error = math.pow(other_prob - probabilities[subset], 2)
        acc_err += error
    rmse = 100 * math.sqrt(acc_err / len(probabilities))
    return rmse, max_sum/2


def distribution_error(other_distribution):
    probabilities = {}
    for size in range(7):
        for subset in itertools.combinations(range(6), size):
            probabilities[frozenset(subset)] = 0.0

    probabilities[frozenset({0, 2})] = 0.3
    probabilities[frozenset({2, 3})] = 0.25
    probabilities[frozenset({2, 5})] = 0.15
    probabilities[frozenset({1})] = 0.1
    probabilities[frozenset({0})] = 0.06
    probabilities[frozenset({2})] = 0.04
    probabilities[frozenset({3})] = 0.04
    probabilities[frozenset({4})] = 0.03
    probabilities[frozenset({5})] = 0.03

    acc_err = 0.0
    max_err = -1
    max_sum = 0
    for subset in probabilities:
        other_prob = other_distribution[subset]
        abs_err = abs(other_prob - probabilities[subset])
        if abs_err > max_err:
            max_err = abs_err
        max_sum += abs_err
        error = math.pow(other_prob - probabilities[subset], 2)
        acc_err += error
    rmse = 100 * math.sqrt(acc_err / len(probabilities))
    return rmse, max_sum/2


def noise_factor_plot():
    n_items = 6
    dataset_name = constants.DATASET_NAME_TPL.format('synthetic_4')
    features = BasicFeaturesNoNormalized(dataset_name, n_items=n_items,
                                         m_features=3)
    features.load_from_file()

    noise_range = np.arange(0.5, 10.5, .5)
    #noise_range = [10]
    noise_column = np.repeat(noise_range, constants.N_FOLDS)
    errors = []
    max_errors = []

    modular_errors = []
    for fold in range(1, constants.N_FOLDS + 1):
        modular = ModularWithFeatures(n_items, features.as_array())
        loaded_data = file.load_set_data(
                    constants.TRAIN_DATA_PATH_TPL.format(
                        fold=fold, dataset=dataset_name))
        modular.train(loaded_data)
        modular.full_distribution()
        modular_errors.append(distribution_error(modular.distribution)[0])
    for noise in noise_range:
        if noise == int(noise):
            noise = int(noise)
        eta_0 = 0.05
        for fold in range(1, constants.N_FOLDS + 1):
            model = GeneralFeatures(n_items, features.as_array(), 2, 1)
            model.load_from_file(constants.NCE_OUT_GENERAL_PATH_TPL.format(
                dataset=dataset_name, fold=fold, l_dim=2, k_dim=1,
                index=features.index, iter=100,
                eta_0=eta_0, adagrad=1, noise=noise))
            model.full_distribution()
            scores = distribution_error(model.distribution)
            errors.append(scores[0])
            max_errors.append(scores[1])

    dataset = pd.DataFrame({
        'noise': noise_column,
        'error': errors,
        'maxerror': max_errors
    })
    ax = sns.pointplot(x='noise', y='maxerror', data=dataset, ci=95)
    ax.set_xlabel(r'$\nu$')
    ax.set_ylabel(r'$\|P_{d} - \hat{P}_{d}\|$')
    ax.set_title(r'Effect of noise-to-data ratio $(\nu)$')
    ax.set_ylim([0, 0.25])


    #ax.plot(ax.get_xlim(), [np.mean(modular_errors), np.mean(modular_errors)],
    #        linestyle='dashed')

    plt.savefig(os.path.join(
        constants.IMAGE_PATH, 'effects_noise_ffldc.eps'),
        bbox_inches='tight')
    plt.show()


def iterations_plot():
    n_items = 6
    dataset_name = constants.DATASET_NAME_TPL.format('synthetic_4')
    features = BasicFeaturesNoNormalized(dataset_name, n_items=n_items,
                                         m_features=3)
    features.load_from_file()

    iter_range = np.arange(100, 1600, 100)
    iter_column = np.repeat(iter_range, constants.N_FOLDS)
    errors = []
    max_errors = []

    modular_errors = []
    for fold in range(1, constants.N_FOLDS + 1):
        modular = ModularWithFeatures(n_items, features.as_array())
        loaded_data = file.load_set_data(
                    constants.TRAIN_DATA_PATH_TPL.format(
                        fold=fold, dataset=dataset_name))
        modular.train(loaded_data)
        modular.full_distribution()
        modular_errors.append(distribution_error(modular.distribution)[0])
    for iter in iter_range:
        eta_0 = 0.05
        for fold in range(1, constants.N_FOLDS + 1):
            model = GeneralFeatures(n_items, features.as_array(), 2, 1)
            model.load_from_file(constants.NCE_OUT_GENERAL_PATH_TPL.format(
                dataset=dataset_name, fold=fold, l_dim=2, k_dim=1,
                index=features.index, iter=iter,
                eta_0=eta_0, adagrad=1, noise=10))
            model.full_distribution()
            scores = distribution_error(model.distribution)
            errors.append(scores[0])
            max_errors.append(scores[1])

    dataset = pd.DataFrame({
        'iter': iter_column,
        'error': errors,
        'maxerror': max_errors
    })
    ax = sns.pointplot(x='iter', y='maxerror', data=dataset, ci=95)
    ax.set_xlabel(r'Epochs')
    ax.set_ylabel(r'$\|P_{d} - \hat{P}_{d}\|$')
    ax.set_title(r'Effect of number of epochs')
    ax.set_ylim([0, 0.25])

    #ax.plot(ax.get_xlim(), [np.mean(modular_errors), np.mean(modular_errors)],
    #        linestyle='dashed')

    plt.savefig(os.path.join(
        constants.IMAGE_PATH, 'effects_epochs_ffldc.eps'),
        bbox_inches='tight')
    plt.show()


def eta_0_no_adagrad():
    eta_0_range = [5e-4, 1e-3, 5e-3, 1e-2, 2e-2]
    iterations = 100
    eta_0_column = np.repeat(eta_0_range, constants.N_FOLDS * iterations / 5)
    iterations_column = np.tile(range(0, iterations, 5), constants.N_FOLDS * len(eta_0_range))
    all_objectives = []
    for eta_0 in eta_0_range:
        objectives = []
        for fold in range(1, constants.N_FOLDS + 1):
            with open(constants.NCE_OUT_OBJECTIVE_PATH_TPL.format(
                    dataset='path_set_synthetic_4', index='1',
                    l_dim=2, k_dim=1,
                    fold=fold, iter=iterations, noise=2,
                    eta_0=eta_0, adagrad=0)) as input_data:
                for idx, objective in enumerate(input_data):
                    if idx % 5 == 0:
                        all_objectives.append(float(objective.strip()))

    dataset = pd.DataFrame(np.c_[eta_0_column, iterations_column, all_objectives],
                           columns=['eta0', 'iterations', 'objective'])
    dataset['iterations'] = dataset['iterations'].astype(np.int32)
    ax = sns.pointplot(x='iterations', data=dataset, y='objective', hue='eta0',
                       ci=95, palette='Set1', markers=['o', 's', 'x', '^', '+'])
    ax.set_title(r'Learning performance without AdaGrad')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'$g(\theta)$')
    ax.set_ylim([-2000, -1300])
    legend = ax.get_legend()
    legend.set_title(r'$\eta_{0}$')
    plt.savefig(os.path.join(
        constants.IMAGE_PATH, 'effects_eta_0_ffldc.eps'),
        bbox_inches='tight')
    plt.show()


def eta_0():
    eta_0_range = [5e-3, 1e-2, 2e-2, 5e-2, 1e-1]
    iterations = 100
    eta_0_column = np.repeat(eta_0_range, constants.N_FOLDS * iterations/5)
    iterations_column = np.tile(range(0, iterations, 5), constants.N_FOLDS * len(eta_0_range))
    all_objectives = []
    for eta_0 in eta_0_range:
        for fold in range(1, constants.N_FOLDS + 1):
            with open(constants.NCE_OUT_OBJECTIVE_PATH_TPL.format(
                    dataset='path_set_synthetic_4', index='1',
                    l_dim=2, k_dim=1,
                    fold=fold, iter=iterations, noise=2,
                    eta_0=eta_0, adagrad=1)) as input_data:
                for idx, objective in enumerate(input_data):
                    if idx % 5 == 0:
                        all_objectives.append(float(objective.strip()))
    dataset = pd.DataFrame(np.c_[eta_0_column, iterations_column, all_objectives],
                           columns=['eta0', 'iterations', 'objective'])
    dataset['iterations'] = dataset['iterations'].astype(np.int32)
    ax = sns.pointplot(x='iterations', y='objective', hue='eta0', ci=95,
                       data=dataset, palette='Set1',
                       markers=['o', 's', 'x', '^', '+'])
    #grid.set(ylim=(-3500, -2000))
    ax.set_title(r'Learning performance with AdaGrad')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'$g(\theta)$')
    ax.set_ylim([-2000, -1300])
    legend = ax.get_legend()
    legend.set_title(r'$\eta_{0}$')
    plt.savefig(os.path.join(
        constants.IMAGE_PATH, 'effects_eta_0_ffldc_adagrad.eps'),
        bbox_inches='tight')
    plt.show()


def adagrad_comparison():
    dataset_name = 'path_set_synthetic_4'
    features_name = '1'
    l_dim = 2
    k_dim = 1
    objective_column = []
    iterations_column = []
    type_column = np.repeat([r'No $\eta_{0} = 0.005$', r'Yes $\eta_{0} = 0.1$'],
                            constants.N_FOLDS * 20)
    for fold in range(1, constants.N_FOLDS + 1):
        with open(constants.NCE_OUT_OBJECTIVE_PATH_TPL.format(
                dataset=dataset_name, index=features_name,
                l_dim=l_dim, k_dim=k_dim,
                fold=fold, iter=100, noise=2,
                eta_0=0.005, adagrad=0)) as input_data:
            for idx, objective in enumerate(input_data):
                if idx % 5 == 0:
                    iterations_column.append(idx)
                    objective_column.append(float(objective.strip()))
    for fold in range(1, constants.N_FOLDS + 1):
        with open(constants.NCE_OUT_OBJECTIVE_PATH_TPL.format(
                dataset=dataset_name, index=features_name,
                l_dim=l_dim, k_dim=k_dim,
                fold=fold, iter=100, noise=2,
                eta_0=0.1, adagrad=1)) as input_data:
            for idx, objective in enumerate(input_data):
                if idx % 5 == 0:
                    iterations_column.append(idx)
                    objective_column.append(float(objective.strip()))

    dataset = pd.DataFrame(data={
        'AdaGrad': type_column,
        'Epoch': iterations_column,
        'objective': objective_column})
    ax = sns.pointplot(x='Epoch', y='objective', hue='AdaGrad', ci=95,
                       data=dataset, palette='Set1',
                       markers=['o', 's'])

    # fig, ax = plt.subplots()
    # line_1 = plt.errorbar(np.arange(len(avg_objectives_1)), avg_objectives_1, marker='o',
    #                       yerr=std_objectives_1)
    # line_2 = plt.errorbar(np.arange(len(avg_objectives_2)), avg_objectives_2, marker='^',
    #                       alpha=0.8, yerr=std_objectives_2)
    # ax.set_xlabel(r'Iterations')
    ax.set_ylabel(r'$g(\theta)$')
    ax.set_title('Learning performance with/without AdaGrad')
    # #ax.set_ylim([-3500, -2000])
    # #ax.set_ylim([-11000, -5000])
    # # ax.set_xlim([0, 49])
    # ax.legend((line_1, line_2), ('Without AdaGrad', 'With AdaGrad'), loc='lower right')
    plt.savefig(os.path.join(
       constants.IMAGE_PATH, 'ffldc_adagrad_comparison.eps'),
       bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    plots.setup()
    sns.set_palette(sns.color_palette('Set1', 2))
    #noise_factor_plot()
    #eta_0_no_adagrad()
    #eta_0()
    #adagrad_comparison()
    iterations_plot()
