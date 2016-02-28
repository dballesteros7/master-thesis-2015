import os

import numpy as np

import constants
from processing.ranking import rank_results

import matplotlib.pyplot as plt


def do_plot():
    dataset_name = constants.DATASET_NAME_TPL.format('10')
    x_values = np.arange(0.1, 1.1, 0.1)

    y_values = []
    std_values = []
    for g in x_values:
        results = rank_results(dataset_name,
                     'submod_f_gauss_{}_l_2_k_2'.format(g),
                     5)
        y_values.append(results[0][0])
        std_values.append(results[1][0])

    modular_result = rank_results(dataset_name, 'modular_features_0', 5)
    submodular_result = rank_results(dataset_name, 'submod_f_0_l_2_k_2', 5)

    fig, ax = plt.subplots()
    line1 = ax.errorbar(x_values, y_values, yerr=std_values, color='#4daf4a')
    line2 = plt.plot([0, 1.1], [modular_result[0][0], modular_result[0][0]], color='#377eb8', linestyle='--')
    line3 = plt.plot([0, 1.1], [submodular_result[0][0], submodular_result[0][0]], color='#e41a1c', linestyle='--')

    ax.set_xlabel('$\sigma$')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(r'$\mathrm{Gaussian\ Features\ Score}$')
    ax.set_xlim([0, 1.1])
    ax.set_ylim([0, 50])
    ax.legend((line1, line2[0], line3[0]), ('Gaussian FLDC', 'Modular', 'FLDC (K=2, L=2)'), loc='upper right')
    plt.savefig(os.path.join(
       constants.IMAGE_PATH, 'gaussian_features_score.eps'),
       bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    do_plot()
