import os

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

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
    line_1 = plt.errorbar(np.arange(50), avg_objectives_1, marker='o',
                          yerr=std_objectives_1)
    # line_2 = plt.errorbar(np.arange(100), avg_objectives_2, color='#3E9651',
    #              linestyle='-', marker='^', alpha=0.8,
    #              yerr=std_objectives_2, ecolor='#922428')
    ax.set_xlabel(r'$i$')
    ax.set_ylabel(r'$g(\theta)$')
    ax.set_title('NCE objective')
    ax.set_xlim([0, 49])
    ax.set_ylim([-200000, -100000])
    plt.savefig(os.path.join(
       constants.IMAGE_PATH, '100_no_singles_features_objective.eps'),
       bbox_inches='tight')
    plt.show()


def main():
    dataset_name = 'path_set_100_no_singles'
    features_name = 'gauss_0.15_k_10'
    l_dim = 0
    k_dim = 30
    plot_objective_progress(dataset_name, features_name, l_dim, k_dim)

if __name__ == '__main__':
    main()
