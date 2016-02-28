from collections import defaultdict

import numpy as np
import constants


def compute_measures(true_item, ranked_suggestions):
    accuracy = 1 if ranked_suggestions[0] == true_item else 0
    rank = 1 + np.argwhere(np.array(ranked_suggestions) == true_item)[0][0]
    return accuracy, rank


def rank_results(dataset_name, model_name, eval_size):
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


def main():
    dataset_name = constants.DATASET_NAME_TPL.format('10')
    models = [
        #'pseudo_markov',
        'modular_features_0', 'modular_features_3', 'submod_f_0_l_2_k_2', 'submod_f_3_l_2_k_2', #'submod_f_0_l_2_k_0',
        #'submod_f_0_l_3_k_0', 'submod_f_0_l_4_k_0', 'submod_f_0_l_4_k_2'
        #'modular_features_0', 'submod_f_0_l_10_k_10', 'markov', 'pseudo_markov',
        #'proximity', 'proximity_r', 'submod_f_1_l_10_k_10'
        #'modular_features_0', 'markov', 'pseudo_markov', 'proximity', 'proximity_r',
        #'submod_f_0_l_20_k_20'
    ]
    for model_name in models:
        results = rank_results(dataset_name, model_name, 5)
        print(results[0])
        print(results[1])

if __name__ == '__main__':
    main()
