import os

import matplotlib.pyplot as plt
import numpy as np

import constants
from processing.ranking import rank_results


def plot_scores_total():
    x_values = np.arange(5)

    values = [13.580926666750148, 7.1600711284758578, 0.0047103155911446069, 8.993418055849526, 0.0047103155911446069]
    error = [0.79855330773358579, 0.67832228203036116, 0.014130946773433821, 0.73251292320851713, 0.014130946773433821]
    fig, ax = plt.subplots()
    ax.bar(x_values, values, color='#e41a1c',
           yerr=error, width=1,
           alpha=0.8, ecolor='#000000')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Model')
    ax.set_title('Model accuracy')
    ax.set_xticks(x_values + 0.5)
    ticks = ['Markov (H)', 'Modular', 'Modular (F)', 'FLDC', 'FFLDC']
    ax.set_xticklabels(ticks)
    ax.set_ylim([0, 15])
    plt.savefig(os.path.join(
       constants.IMAGE_PATH, 'score_all.eps'),
       bbox_inches='tight')
    plt.show()

def plot_scores():
    dataset_name = constants.DATASET_NAME_TPL.format('all_no_singles')
    set_sizes = 5

    markov, markov_std, _, __ = rank_results(dataset_name,
                                             'pseudo_markov',
                                                set_sizes)
    proximity, proximity_std, _, __ = rank_results(dataset_name,
                                             'proximity_r',
                                                set_sizes)
    modular, modular_std, _, __ = rank_results(dataset_name,
                                               'modular_features_0',
                                               set_sizes)
    modular_f, modular_f_std, _, __ = rank_results(dataset_name,
                                               'modular_features_1',
                                               set_sizes)
    submod, submod_std, _, __ = rank_results(dataset_name,
                                             'submod_f_0_l_20_k_20',
                                             set_sizes)
    submod_f, submod_f_std, _, __ = rank_results(dataset_name,
                                             'submod_f_1_l_20_k_20',
                                             set_sizes)
    # dataset_name = constants.DATASET_NAME_TPL.format('100_no_singles')
    # modular_no_singles, modular_no_singles_std, _, __ = rank_results(dataset_name,
    #                                            'modular_features_0',
    #                                            set_sizes)
    # submod_no_singles, submod_no_singles_std, _, __ = rank_results(dataset_name,
    #                                          'submod_f_0_l_20_k_20',
    #                                          set_sizes)
    # proximity, proximity_std, _, __ = rank_results(dataset_name,
    #                                                'proximity_r',
    #                                                set_sizes)
    # proximity_bad, proximity_bad_std, _, __ = rank_results(dataset_name,
    #                                                'proximity',
    #                                                set_sizes)

    x_values = np.arange(set_sizes + 1)
    width = 0.2
    fig, ax = plt.subplots()
    rects1 = ax.bar(x_values, modular, width, color='#a6cee3',
                    yerr=modular_std,
                    alpha=0.8, ecolor='#000000')
    rects2 = ax.bar(x_values + width, modular_no_singles, width, color='#1f78b4',
                    yerr=modular_no_singles_std, alpha=0.8, ecolor='#000000')
    rects3 = ax.bar(x_values + 2 * width, submod, width, color='#b2df8a',
                    yerr=submod_std, alpha=0.8, ecolor='#000000')
    rects4 = ax.bar(x_values + 3 * width, submod_no_singles, width, color='#33a02c',
                    yerr=submod_no_singles_std, alpha=0.8, ecolor='#000000')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy by partial set size (N = 100)')
    ax.set_xticks(x_values + 2 * width)
    ticks = ['Total']
    ticks.extend([str(x) for x in range(1, set_sizes)])
    ticks.append('{}+'.format(set_sizes))
    ax.set_xticklabels(ticks)
    ax.set_ylim([0, 50])
    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]),
              ('Modular', 'Modular - No singletons', 'FLDC (L=20,K=20)',
               'FLDC (L=20,K=20) - No singletons'),
              loc='upper left')
    plt.savefig(os.path.join(
       constants.IMAGE_PATH, 'set_size_score_100_no_singletons.eps'),
       bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    plot_scores_total()
