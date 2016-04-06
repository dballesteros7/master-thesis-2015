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


def distribution_error_1(other_distribution):
    probabilities = {}
    for size in range(5):
        for subset in itertools.combinations(range(4), size):
            probabilities[frozenset(subset)] = 0.0

    probabilities[frozenset({0, 1})] = 0.5
    probabilities[frozenset({2, 3})] = 0.5

    acc_err = 0.0
    max_err = -1
    for subset in probabilities:
        other_prob = other_distribution[subset]
        abs_err = abs(other_prob - probabilities[subset])
        if abs_err > max_err:
            max_err = abs_err
        error = math.pow(other_prob - probabilities[subset], 2)
        acc_err += error
    print(sorted(other_distribution.items(), key=lambda x: -x[1]))
    rmse = 100 * math.sqrt(acc_err / len(probabilities))
    return rmse, 100 * max_err


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
    for subset in probabilities:
        other_prob = other_distribution[subset]
        abs_err = abs(other_prob - probabilities[subset])
        if abs_err > max_err:
            max_err = abs_err
        error = math.pow(other_prob - probabilities[subset], 2)
        acc_err += error
    rmse = 100 * math.sqrt(acc_err / len(probabilities))
    return rmse, 100 * max_err


def noise_factor_plot():
    n_items = 4
    dataset_name = constants.DATASET_NAME_TPL.format('synthetic_2')
    features = IdentityFeatures(dataset_name, n_items=n_items,
                                m_features=n_items)
    features.load_from_file()

    noise_range = list(range(1, 11))
    mean_errs = []
    std_errs = []
    max_errs = []
    std_max_errs = []

    for noise in noise_range:
        errors = []
        max_errors = []
        for fold in range(1, constants.N_FOLDS + 1):
            model = GeneralFeatures(n_items, features.as_array(), 2, 2)
            model.load_from_file(constants.NCE_OUT_GENERAL_PATH_TPL.format(
                dataset=dataset_name, fold=fold, l_dim=2, k_dim=2,
                index=features.index, iter=10,
                eta_0=0.05, adagrad=1, noise=noise))
            print(model.diversity_weights)
            print(model.coherence_weights)
            model.full_distribution()
            # for key, prob in sorted(model.distribution.items(), key=lambda x:x[1]):
            #     print('{}: {}'.format(key, prob))
            scores = distribution_error_1(model.distribution)
            errors.append(scores[0])
            max_errors.append(scores[1])

        mean_err = np.mean(errors)
        std_err = np.std(errors)
        mean_errs.append(mean_err)
        std_errs.append(std_err)
        max_errs.append(np.mean(max_errors))
        std_max_errs.append(2*np.std(max_errors))

    fig, ax = plt.subplots()

    line1 = ax.errorbar(noise_range, mean_errs, yerr=std_errs)
    #line2 = ax.errorbar(noise_range, max_errs, yerr=std_max_errs)


    ax.set_xlabel(r'$\nu$')
    ax.set_ylabel(r'$e$(\%)')
    ax.set_title(r'Effect of noise-to-data ratio $(\nu)$')

    #ax.legend([line1, line2], [r'$e_{rms}$', r'$e_{\max}$'])

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

    iteration_range = list(range(10, 110, 10))
    mean_errs = []
    std_errs = []
    max_errs = []
    std_max_errs = []

    for iteration in iteration_range:
        errors = []
        max_errors = []
        for fold in range(1, constants.N_FOLDS + 1):
            model = GeneralFeatures(n_items, features.as_array(), 2, 1)
            model.load_from_file(constants.NCE_OUT_GENERAL_PATH_TPL.format(
                dataset=dataset_name, fold=fold, l_dim=2, k_dim=1,
                index=features.index, iter=iteration,
                eta_0=0.01, adagrad=0, noise=10))
            model.full_distribution()
            # for key, prob in sorted(model.distribution.items(), key=lambda x:x[1]):
            #     print('{}: {}'.format(key, prob))
            scores = distribution_error(model.distribution)
            errors.append(scores[0])
            max_errors.append(scores[1])

        mean_err = np.mean(errors)
        std_err = np.std(errors)
        mean_errs.append(mean_err)
        std_errs.append(std_err)
        max_errs.append(np.mean(max_errors))
        std_max_errs.append(np.std(max_errors))

    fig, ax = plt.subplots()

    line1 = ax.errorbar(iteration_range, mean_errs, yerr=std_errs)
    line2 = ax.errorbar(iteration_range, max_errs, yerr=std_max_errs)


    ax.set_xlabel(r'$\nu$')
    ax.set_ylabel(r'$e$(\%)')
    ax.set_title(r'Effect of noise-to-data ratio $(\nu)$')

    ax.legend([line1, line2], [r'$e_{rms}$', r'$e_{\max}$'])

    # plt.savefig(os.path.join(
    #     constants.IMAGE_PATH, 'effects_noise_ffldc.eps'),
    #     bbox_inches='tight')
    plt.show()


def eta_0_no_adagrad():
    eta_0_range = [5e-4, 1e-3, 5e-3, 1e-2, 2e-2, 3e-2]
    iterations = 100
    eta_0_column = np.repeat(eta_0_range, iterations / 5)
    iterations_column = np.tile(range(0, iterations, 5), len(eta_0_range))
    all_objectives = []
    std_objectives = []
    for eta_0 in eta_0_range:
        objectives = []
        for fold in range(1, constants.N_FOLDS + 1):
            fold_objectives = []
            objectives.append(fold_objectives)
            with open(constants.NCE_OUT_OBJECTIVE_PATH_TPL.format(
                    dataset='path_set_synthetic_4', index='1',
                    l_dim=2, k_dim=1,
                    fold=fold, iter=iterations, noise=2,
                    eta_0=eta_0, adagrad=0)) as input_data:
                for idx, objective in enumerate(input_data):
                    if idx % 5 == 0:
                        fold_objectives.append(float(objective.strip()))
        all_objectives.extend(np.mean(objectives, axis=0))
        std_objectives.extend(2 * np.std(objectives, axis=0))

    dataset = pd.DataFrame(np.c_[eta_0_column, iterations_column, all_objectives, std_objectives],
                           columns=['eta0', 'iterations', 'objective', 'std_objective'])
    grid = sns.FacetGrid(dataset, col='eta0', hue='eta0', col_wrap=2)
    grid.map(plt.errorbar, 'iterations', 'objective', 'std_objective', marker='o')
    #grid.set(ylim=(-3500, -2000))
    grid.set_titles(r'$\eta_0 = {col_name}$')
    grid.set_xlabels('Iterations')
    grid.set_ylabels(r'$g(\theta)$')
    plt.savefig(os.path.join(
        constants.IMAGE_PATH, 'effects_eta_0_ffldc.eps'),
        bbox_inches='tight')
    plt.show()


def eta_0():
    eta_0_range = [5e-3, 1e-2, 2e-2, 3e-2, 5e-2, 1e-1]
    iterations = 100
    eta_0_column = np.repeat(eta_0_range, iterations/5)
    iterations_column = np.tile(range(0, iterations, 5), len(eta_0_range))
    all_objectives = []
    std_objectives = []
    for eta_0 in eta_0_range:
        objectives = []
        for fold in range(1, constants.N_FOLDS + 1):
            fold_objectives = []
            objectives.append(fold_objectives)
            with open(constants.NCE_OUT_OBJECTIVE_PATH_TPL.format(
                    dataset='path_set_synthetic_4', index='1',
                    l_dim=2, k_dim=1,
                    fold=fold, iter=iterations, noise=2,
                    eta_0=eta_0, adagrad=1)) as input_data:
                for idx, objective in enumerate(input_data):
                    if idx % 5 == 0:
                        fold_objectives.append(float(objective.strip()))
        all_objectives.extend(np.mean(objectives, axis=0))
        std_objectives.extend(2 * np.std(objectives, axis=0))
    dataset = pd.DataFrame(np.c_[eta_0_column, iterations_column, all_objectives, std_objectives],
                           columns=['eta0', 'iterations', 'objective', 'std_objective'])
    grid = sns.FacetGrid(dataset, col='eta0', hue='eta0', col_wrap=2)
    grid.map(plt.errorbar, 'iterations', 'objective', 'std_objective', marker='o')
    #grid.set(ylim=(-3500, -2000))
    grid.set_titles(r'$\eta_0 = {col_name}$')
    grid.set_xlabels('Iterations')
    grid.set_ylabels(r'$g(\theta)$')
    plt.savefig(os.path.join(
        constants.IMAGE_PATH, 'effects_eta_0_ffldc_adagrad.eps'),
        bbox_inches='tight')
    plt.show()


def adagrad_comparison():
    dataset_name = 'path_set_synthetic_4'
    features_name = '1'
    l_dim = 2
    k_dim = 1
    objectives = []
    for fold in range(1, constants.N_FOLDS + 1):
        fold_objectives = []
        objectives.append(fold_objectives)
        with open(constants.NCE_OUT_OBJECTIVE_PATH_TPL.format(
                dataset=dataset_name, index=features_name,
                l_dim=l_dim, k_dim=k_dim,
                fold=fold, iter=100, noise=2,
                eta_0=0.005, adagrad=0)) as input_data:
            for idx, objective in enumerate(input_data):
                if idx % 2 == 0:
                    fold_objectives.append(float(objective.strip()))
    objectives = np.array(objectives)
    avg_objectives_1 = np.mean(objectives, axis=0)
    std_objectives_1 = np.std(objectives, axis=0)
    objectives = []
    for fold in range(1, constants.N_FOLDS + 1):
        fold_objectives = []
        objectives.append(fold_objectives)
        with open(constants.NCE_OUT_OBJECTIVE_PATH_TPL.format(
                dataset=dataset_name, index=features_name,
                l_dim=l_dim, k_dim=k_dim,
                fold=fold, iter=100, noise=2,
                eta_0=0.05, adagrad=1)) as input_data:
            for idx, objective in enumerate(input_data):
                if idx % 2 == 0:
                    fold_objectives.append(float(objective.strip()))
    objectives = np.array(objectives)
    avg_objectives_2 = np.mean(objectives, axis=0)
    std_objectives_2 = np.std(objectives, axis=0)

    fig, ax = plt.subplots()
    line_1 = plt.errorbar(np.arange(len(avg_objectives_1)), avg_objectives_1, marker='o',
                          yerr=std_objectives_1)
    line_2 = plt.errorbar(np.arange(len(avg_objectives_2)), avg_objectives_2, marker='^',
                          alpha=0.8, yerr=std_objectives_2)
    ax.set_xlabel(r'Iterations')
    ax.set_ylabel(r'$g(\theta)$')
    #ax.set_title('AdaGrad\'s effect on NCE learning')
    #ax.set_ylim([-3500, -2000])
    #ax.set_ylim([-11000, -5000])
    # ax.set_xlim([0, 49])
    ax.legend((line_1, line_2), ('Without AdaGrad', 'With AdaGrad'), loc='lower right')
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
    adagrad_comparison()
