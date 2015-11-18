from __future__ import division, print_function
from matplotlib import pyplot as plt
import numpy as np
import IPython
from itertools import product
from itertools import chain, combinations


RANKING_PATH = 'ranking_test'

n_folds = 10
folds_range = range(0, n_folds)
#folds_range = [8]
plot = False

datasets = ['safety']
dim_assignment = {}
for dataset in datasets:
    dim_assignment[dataset] = 2



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
    rank_submod = dict()
    rank_mod = dict()
    rank_random = dict()
    rank_dpp = dict()
    for dataset in datasets:
        results_all_submod[dataset] = []
        results_all_mod[dataset] = []
        results_all_dpp[dataset] = []
        results_all_random[dataset] = []

        rank_submod[dataset] = []
        rank_mod[dataset] = []
        rank_dpp[dataset] = []
        rank_random[dataset] = []

    for dataset, fold in product(datasets, folds_range):
        np.random.seed(20150820)

        print('-' * 30)
        print('dataset: %s (fold %d)' % (dataset, fold + 1))
        print('dim=%d' % dim_assignment[dataset])

        result_ranking_submod_f = '{0}/{1}_submod_d_{2}_fold_{3}.pkl'.format(RANKING_PATH, dataset, dim_assignment[dataset], fold + 1)
        result_ranking_mod_f = '{0}/{1}_mod_fold_{2}.pkl'.format(RANKING_PATH, dataset, fold + 1)
        result_ranking_dpp_f = '{0}/{1}_dpp_fold_{2}.pkl'.format(RANKING_PATH, dataset, fold + 1)
        result_ranking_random_f = '{0}/{1}_random_fold_{2}.pkl'.format(RANKING_PATH, dataset, fold + 1)
 # ranking_test/safety_dpp_fold_1.pkl
        result_ranking_gt_f = '{0}/{1}_gt_fold_{2}.pkl'.format(RANKING_PATH, dataset, fold + 1)

        GROUND_TRUTH = result_ranking_gt_f
        METHODS = {
                'submod': result_ranking_submod_f,  # 'ranking_test/prop1.csv',
                'mod': result_ranking_mod_f,  # 'ranking_test/prop2.csv',
                'random': result_ranking_random_f
            }


        results = dict()
        with open(GROUND_TRUTH) as f_gt:
            lines_gt = list(f_gt)
            for method, filename in METHODS.items():
                # print('processing', method)
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


