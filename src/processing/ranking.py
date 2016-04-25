from collections import defaultdict

import numpy as np
import pandas as pd

import constants


def compute_measures(true_item, ranked_suggestions):
    accuracy = 1 if ranked_suggestions[0] == true_item else 0
    rank = 1 + np.argwhere(np.array(ranked_suggestions) == true_item)[0][0]
    return accuracy, rank


def _rank_results(dataset_name, model_name, eval_size):
    cross_accuracies = [[] for _ in range(eval_size + 1)]
    cross_ranks = [[] for _ in range(eval_size + 1)]
    for fold in range(1, constants.N_FOLDS + 1):
        accuracies = [[] for _ in range(eval_size + 1)]
        ranks = [[] for _ in range(eval_size + 1)]

        partial_results = constants.PARTIAL_DATA_PATH_TPL.format(
            dataset=dataset_name, fold=fold)
        ground_truth_results =\
            constants.GROUND_TRUTH_DATA_PATH_TPL.format(
                dataset=dataset_name, fold=fold)
        model_results = constants.RANKING_MODEL_PATH_TPL.format(
            dataset=dataset_name, fold=fold, model=model_name)

        with open(model_results, 'r') as model_file, \
                open(ground_truth_results, 'r') as ground_truth_file, \
                open(partial_results, 'r') as partial_set_file:
            for model_line, truth_line, partial_line in zip(
                    model_file, ground_truth_file, partial_set_file):
                model_line = model_line.strip()
                if model_line == '-':
                    continue
                truth_line = truth_line.strip()
                partial_line = partial_line.strip()
                suggested_set = [int(item) for item in model_line.split(',')]
                true_item = int(truth_line)
                acc, rank = compute_measures(true_item, suggested_set)
                partial_size = len(partial_line.split(',')) - 1
                if partial_size >= eval_size:
                    accuracies[eval_size].append(acc)
                    ranks[eval_size].append(rank)
                else:
                    accuracies[partial_size].append(acc)
                    ranks[partial_size].append(rank)
                accuracies[0].append(acc)
                ranks[0].append(rank)
            for cross_accuracy, accuracy_list in zip(cross_accuracies, accuracies):
                if len(accuracy_list):
                    cross_accuracy.append(np.mean(accuracy_list))
            for cross_rank, rank_list in zip(cross_ranks, ranks):
                if len(rank_list):
                    cross_rank.append(np.mean([1 / rank for rank in rank_list]))
    return cross_accuracies, cross_ranks


def rank_results(dataset_name, model_name, eval_size):
    cross_accuracies, cross_ranks = _rank_results(
        dataset_name, model_name, eval_size)
    mean_accuracies = []
    std_accuracies = []
    for accuracies in cross_accuracies:
        if len(accuracies):
            mean_accuracies.append(100 * np.mean(accuracies))
            std_accuracies.append(100 * np.std(accuracies))
        else:
            mean_accuracies.append(0.)
            std_accuracies.append(0.)
    mean_ranks = []
    std_ranks = []
    for ranks in cross_ranks:
        if len(ranks):
            mean_ranks.append(100 * np.mean(ranks))
            std_ranks.append(100 * np.std(ranks))
        else:
            mean_ranks.append(0.)
            std_ranks.append(0.)
    return mean_accuracies, std_accuracies, mean_ranks, std_ranks


def rank_results_pandas(dataset_name, model_name, eval_size):
    cross_accuracies, cross_ranks = _rank_results(
        dataset_name, model_name, eval_size)

    return np.array(cross_accuracies[0])


def main():
    dataset_name = constants.DATASET_NAME_TPL.format('10_no_singles')
    models = ['modular_features_0', 'markov', 'pseudo_markov', 'proximity',
              'proximity_ordered',
              'submod_f_0_l_5_k_5_iter_1000_noise_5_eta_0.1_adagrad_1',
              'modular_features_special_gen',
              'submod_f_special_gen_l_5_k_5_iter_5000_noise_5_eta_1_adagrad_1',
              #'submod_f_0_l_15_k_15_iter_1000_noise_5_eta_0.1_adagrad_1',
              # 'submod_f_0_l_20_k_5_iter_1000_noise_5_eta_0.1_adagrad_1',
              # 'submod_f_0_l_20_k_20_iter_1000_noise_5_eta_0.1_adagrad_1',
              # 'submod_f_0_l_20_k_25_iter_1000_noise_5_eta_0.1_adagrad_1',
              # 'submod_f_0_l_20_k_30_iter_1000_noise_5_eta_0.1_adagrad_1',
              # 'submod_f_0_l_15_k_10_iter_1000_noise_5_eta_0.1_adagrad_1'
              #'submod_f_0_l_10_k_10_iter_100_noise_5_eta_1_adagrad_1',
              #'submod_f_0_l_10_k_10_iter_500_noise_5_eta_0.1_adagrad_1',
              #'submod_f_0_l_10_k_10_iter_1000_noise_5_eta_0.1_adagrad_1',
              #'submod_f_gauss_ext_0_k_100_l_10_k_10_iter_500_noise_5_eta_0.1_adagrad_1',
              ]
    #models = ['modular_features_0', 'submod_f_0_l_2_k_2']
    # models = ['modular_features_0', 'submod_f_0_l_20_k_20']
    # sigma = 0.16
    # n_feats = 100
    # for sigma in [0.1, 0.2, 0.3, 0.4]:
    #     for n_feats in [100, 80, 60, 50]:
    #         models.append('modular_features_gauss_{}_k_{}'.format(sigma, n_feats))
    # sigma = 0.16
    # n_feats = 100
    # for dim in range(5, 35, 5):
    #     models.append('submod_f_gauss_{0}_k_{1}_l_{2}_k_{2}'.format(sigma, n_feats, dim))
    for model_name in models:
        results = rank_results(dataset_name, model_name, 5)
        print('{}: {} +- {}'.format(model_name, results[0][0], results[1][0]))

if __name__ == '__main__':
    main()
