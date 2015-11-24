from __future__ import division, print_function
import numpy as np
from itertools import product
import os

DATA_PATH = '/local/workspace/master-thesis-2015/data'
MODEL_PATH = os.path.join(DATA_PATH, 'models')
RANKING_PATH = os.path.join(DATA_PATH, 'ranking_test')

n_folds = 10
folds_range = range(1, n_folds + 1)
plot = False

datasets = ['path_set']
dim_assignment = {}
for dataset in datasets:
    dim_assignment[dataset] = 20



# Adopted for our needs
def compute_precision_curve(true_set, suggested):
    assert len(true_set) == 1

    accuracy = 1. if suggested[0] in true_set else 0.
    rank = 1. + np.argwhere(np.array(suggested) == true_set[0])[0][0]

    return (accuracy, rank)


# def compute_precision_curve(true_set, suggested):
#     # print(true_set)
#     scores = []
#     so_far = 0.
#     for suggestion in suggested:
#         if suggestion in true_set:
#             so_far += 1.
#         scores.append(so_far / len(true_set))
#     # print('given true:', true_set)
#     # print('given pred:', suggested)
#     # print('scores    :', scores)
#     return scores


if __name__ == '__main__':
    results_all_submod = dict()
    results_all_mod = dict()
    results_all_dpp = dict()
    results_all_random = dict()
    results_all_markov = dict()
    rank_submod = dict()
    rank_mod = dict()
    rank_random = dict()
    rank_dpp = dict()
    rank_markov = dict()
    for dataset in datasets:
        results_all_submod[dataset] = []
        results_all_mod[dataset] = []
        results_all_dpp[dataset] = []
        results_all_random[dataset] = []
        results_all_markov[dataset] = []

        rank_submod[dataset] = []
        rank_mod[dataset] = []
        rank_dpp[dataset] = []
        rank_random[dataset] = []
        rank_markov[dataset] = []

    for dataset, fold in product(datasets, folds_range):
        np.random.seed(20150820)

        print('-' * 30)
        print('dataset: %s (fold %d)' % (dataset, fold))
        print('dim=%d' % dim_assignment[dataset])

        result_ranking_submod_f = os.path.join(RANKING_PATH, '{}_submod_d_{}_fold_{}.pkl'.format(dataset, dim_assignment[dataset], fold))
        result_ranking_mod_f = os.path.join(RANKING_PATH, '{}_mod_fold_{}.pkl'.format(dataset, fold))
        result_ranking_dpp_f = os.path.join(RANKING_PATH, '{}_dpp_fold_{}.pkl'.format(dataset, fold))
        result_ranking_random_f = os.path.join(RANKING_PATH, '{}_random_fold_{}.pkl'.format(dataset, fold))
        result_ranking_markov_f = os.path.join(RANKING_PATH, '{}_markov_fold_{}.pkl'.format(dataset, fold))
        result_ranking_gt_f = os.path.join(RANKING_PATH, '{}_gt_fold_{}.pkl'.format(dataset, fold))

        GROUND_TRUTH = result_ranking_gt_f
        METHODS = {
                'submod': result_ranking_submod_f,  # 'ranking_test/prop1.csv',
                'mod': result_ranking_mod_f,  # 'ranking_test/prop2.csv',
                'random': result_ranking_random_f,
                'markov': result_ranking_markov_f
            }


        results = dict()
        with open(GROUND_TRUTH) as f_gt:
            lines_gt = list(f_gt)
            for method, filename in METHODS.items():
                print('processing', method)
                result = None
                avg_score = []
                avg_rank = []
                with open(filename, 'r') as f_sc:
                    for line_gt, line_sc in zip(lines_gt, f_sc):
                        if len(line_gt.strip()) == 0 or len(line_sc.strip()) == 0:
                            continue
                        true_set = list(map(int, line_gt.strip().split(',')))
                        suggested_orig = list(map(int, line_sc.strip().split(',')))

                        suggested = suggested_orig
                        accuracy, rank = compute_precision_curve(true_set, suggested)
                        # print("acc: ", accuracy)
                        # print("rank: ", rank)
                        avg_score.append(accuracy)
                        avg_rank.append(rank)

                if method == 'submod':
                    results_all_submod[dataset].append(np.mean(avg_score))
                    rank_submod[dataset].append(np.mean([1./rank for rank in avg_rank]))
                elif method == 'mod':
                    results_all_mod[dataset].append(np.mean(avg_score))
                    rank_mod[dataset].append(np.mean([1./rank for rank in avg_rank]))
                elif method == 'dpp':
                    results_all_dpp[dataset].append(np.mean(avg_score))
                    rank_dpp[dataset].append(np.mean([1./rank for rank in avg_rank]))
                elif method == 'random':
                    results_all_random[dataset].append(np.mean(avg_score))
                    rank_random[dataset].append(np.mean([1./rank for rank in avg_rank]))
                elif method == 'markov':
                    results_all_markov[dataset].append(np.mean(avg_score))
                    rank_markov[dataset].append(np.mean([1./rank for rank in avg_rank]))
                else:
                    assert False
                # results[method] = np.mean(result, axis=0)

        # break  # TODO remove

    # generate latex table data
    for dataset in datasets:
        # print("DATASET %s (dim %d)" % (dataset, dim_assignment[dataset]))
        # IPython.embed()
        mean_modular = 100 * np.mean(results_all_mod[dataset])
        std_modular = 100 * np.std(results_all_mod[dataset])
        mean_rank_modular = 100 * np.mean(rank_mod[dataset])
        std_rank_modular = 100 * np.std(rank_mod[dataset])
        print('\\multirow{3}{*}{%s} & modular & $%2.2f \pm %2.2f$ & $%2.2f \pm %2.2f$ \\\\' % (dataset, mean_modular, std_modular, mean_rank_modular, std_rank_modular)) # np.mean(results_all_mod[dataset][5]), np.mean(results_all_mod[dataset][10])))

        #print(' & random                     & $%f$ & $%f$ & $%f$ \\' % (np.mean(results_all_random[dataset][-1]), 0, 0)) # np.mean(results_all_random[dataset][5]), np.mean(results_all_random[dataset][10])))

        mean_submod = 100 * np.mean(results_all_submod[dataset])
        std_submod = 100 * np.std(results_all_submod[dataset])
        mean_rank_submod = 100 * np.mean(rank_submod[dataset])
        std_rank_submod = 100 * np.std(rank_submod[dataset])
        print(' & \\FLID       & $%2.2f \pm %2.2f$ & $%2.2f \pm %2.2f$ \\\\' % (mean_submod, std_submod, mean_rank_submod, std_rank_submod)) # np.mean(results_all_submod[dataset][5]), np.mean(results_all_submod[dataset][10])))

        mean_markov = 100 * np.mean(results_all_markov[dataset])
        std_markov = 100 * np.std(results_all_markov[dataset])
        mean_rank_markov = 100 * np.mean(rank_markov[dataset])
        std_rank_markov = 100 * np.std(rank_markov[dataset])
        print(' & \\Markov       & $%2.2f \pm %2.2f$ & $%2.2f \pm %2.2f$ \\\\' % (mean_markov, std_markov, mean_rank_markov, std_rank_markov))
