import os

from matplotlib import pyplot as plt
import numpy as np

import constants


def plot_objective_progress(dataset_name, features_name, l_dim, k_dim):
    objectives = []
    for fold in range(1, constants.N_FOLDS + 1):
        fold_objectives = []
        objectives.append(fold_objectives)
        with open(constants.NCE_OUT_OBJECTIVE_PATH_TPL.format(
                dataset=dataset_name, index=features_name,
                l_dim=l_dim, k_dim=k_dim,
                fold=fold)) as input_data:
            for objective in input_data:
                fold_objectives.append(float(objective.strip()))
    objectives = np.array(objectives)
    avg_objectives_1 = np.mean(objectives, axis=0)
    std_objectives_1 = np.std(objectives, axis=0)

    # objectives = []
    # for fold in range(1, constants.N_FOLDS + 1):
    #     fold_objectives = []
    #     objectives.append(fold_objectives)
    #     with open(constants.NCE_OUT_OBJECTIVE_PATH_TPL.format(
    #             dataset=dataset_name, index='1',
    #             l_dim=l_dim, k_dim=k_dim,
    #             fold=fold)) as input_data:
    #         for objective in input_data:
    #             fold_objectives.append(float(objective.strip()))
    # objectives = np.array(objectives)
    # avg_objectives_2 = np.mean(objectives, axis=0)
    # std_objectives_2 = np.std(objectives, axis=0)

    fig, ax = plt.subplots()
    line_1 = plt.errorbar(np.arange(10), avg_objectives_1, color='#396AB1',
                          linestyle='-', marker='o', alpha=0.8,
                          yerr=std_objectives_1, ecolor='#CC2529')
    # line_2 = plt.errorbar(np.arange(100), avg_objectives_2, color='#3E9651',
    #              linestyle='-', marker='^', alpha=0.8,
    #              yerr=std_objectives_2, ecolor='#922428')
    ax.set_xlabel(r'$i$')
    ax.set_ylabel(r'$g(\theta)$')
    ax.set_title(r'$\mathrm{NCE\ objective}$')
    #ax.set_xticks([0, 9, 19, 29, 39, 49])
    #ax.set_xticklabels([1, 10, 20, 30, 40])
    #ax.set_xlim([0, 49])
    #ax.set_ylim([-50000, 0])
    # ax.legend((line_1[0], line_2[0]),
    #           ('FLDC', 'FFLDC'),
    #           loc='upper left')
    # plt.savefig(os.path.join(
    #    constants.IMAGE_PATH, 'synthetic_4_objective_comparison.eps'),
    #    bbox_inches='tight')
    plt.show()


def main():
    dataset_name = 'path_set_100_no_singles'
    features_name = 'gauss_0.2_k_100'
    l_dim = 5
    k_dim = 5
    plot_objective_progress(dataset_name, features_name, l_dim, k_dim)

if __name__ == '__main__':
    main()
