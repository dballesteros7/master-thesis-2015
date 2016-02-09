import os

import matplotlib.pyplot as plt
import numpy as np

import constants
from processing.ranking import rank_results


def plot_scores():
    N_ITEMS = 10

    markov, markov_std, _, __ = rank_results('markov', N_ITEMS)
    pseudo_markov, pseudo_markov_std, _, __ = rank_results('pseudo_markov',
                                                           N_ITEMS)
    modular, modular_std, _, __ = rank_results('modular', N_ITEMS)
    submod, submod_std, _, __ = rank_results('submod_f_0_l_5_k_5', N_ITEMS)

    n_bars = 8
    x_values = np.arange(n_bars)
    width = 0.2
    fig, ax = plt.subplots()
    rects1 = ax.bar(x_values, markov[:n_bars], width, color='b',
                    yerr=markov_std[:n_bars],
                    alpha=0.8, ecolor='r')
    rects2 = ax.bar(x_values + width, pseudo_markov[:n_bars], width, color='g',
                    yerr=pseudo_markov_std[:n_bars], alpha=0.8, ecolor='r')
    rects3 = ax.bar(x_values + 2 * width, modular[:n_bars], width, color='y',
                    yerr=modular_std[:n_bars], alpha=0.8, ecolor='r')
    rects4 = ax.bar(x_values + 3 * width, submod[:n_bars], width, color='c',
                    yerr=submod_std[:n_bars], alpha=0.8, ecolor='r')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy by partial set size (N = 10)')
    ax.set_xticks(x_values + 2 * width)
    ticks = ['Total']
    ticks.extend([str(x) for x in range(1, n_bars + 1)])
    ax.set_xticklabels(ticks)
    ax.set_ylim([0, 100])
    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]),
              ('Markov', 'Markov Heuristic', 'Modular', 'FLDC (L=5, K=5)'),
              loc='upper left')
    plt.savefig(os.path.join(constants.IMAGE_PATH, 'set_size_score_10.eps'),
                bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    plot_scores()
