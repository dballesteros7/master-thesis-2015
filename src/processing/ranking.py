import numpy as np
import constants


def compute_measures(true_item, ranked_suggestions):
    accuracy = 1 if ranked_suggestions[0] == true_item else 0
    rank = 1 + np.argwhere(np.array(ranked_suggestions) == true_item)[0][0]
    return accuracy, rank


def rank_results(model_name):
    cross_accuracies = np.zeros(constants.N_FOLDS)
    cross_ranks = np.zeros(constants.N_FOLDS)
    for fold in range(1, constants.N_FOLDS + 1):
        accuracies = []
        ranks = []
        model_results = constants.RANKING_MODEL_PATH_TPL.format(
            dataset=constants.DATASET_NAME, fold=fold, model=model_name)
        ground_truth_results = constants.RANKING_MODEL_PATH_TPL.format(
            dataset=constants.DATASET_NAME, fold=fold, model='gt')
        with open(model_results, 'r') as model_file, \
                open(ground_truth_results, 'r') as ground_truth_file:
            for model_line, truth_line in zip(model_file, ground_truth_file):
                model_line = model_line.strip()
                truth_line = truth_line.strip()
                suggested_set = [int(item) for item in model_line.split(',')]
                true_item = int(truth_line)
                acc, rank = compute_measures(true_item, suggested_set)
                accuracies.append(acc)
                ranks.append(rank)
            cross_accuracies[fold - 1] = np.mean(accuracies)
            cross_ranks[fold - 1] = np.mean([1 / rank for rank in ranks])
    mean_accuracy = 100 * np.mean(cross_accuracies)
    std_accuracy = 100 * np.std(cross_accuracies)
    mean_rank = 100 * np.mean(cross_ranks)
    std_rank = 100 * np.std(cross_ranks)
    return mean_accuracy, std_accuracy, mean_rank, std_rank


def main():
    models = [
        'modular_features', 'markov', 'proximity',
        'submod_f_0_d_2', 'submod_f_0_d_5', 'submod_f_0_d_10',
        'submod_f_2_l_0_k_0', 'submod_f_2_l_0_k_2', 'submod_f_2_l_0_k_5', 'submod_f_2_l_0_k_10'
    ]
    for model_name in models:
        results = rank_results(model_name)
        print('{:2.2f} & {:2.2f} & {:2.2f} & {:2.2f}'.format(*results))

if __name__ == '__main__':
    main()
