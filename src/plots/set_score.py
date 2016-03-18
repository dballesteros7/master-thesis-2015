import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import constants
from processing.ranking import rank_results, rank_results_pandas

def plot_score():
    accuracies = []
    accuracies_error = []
    model_template = 'submod_f_0_l_{0}_k_{0}'
    k_values = [10, 20, 30, 40, 50, 60, 70]
    for k in k_values:
        model = model_template.format(k)
        acc, acc_std, _, _ = rank_results('path_set_100_no_singles', model, 0)
        accuracies.append(acc[0])
        accuracies_error.append(acc_std)
    fig, ax = plt.subplots()
    plt.errorbar(k_values, accuracies, yerr=accuracies_error, marker='o')
    ax.set_title('Model accuracy')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('L,K')
    plt.savefig(os.path.join(
       constants.IMAGE_PATH, 'score_100_no_singles_no_features_dim.eps'),
       bbox_inches='tight')
    plt.show()

def plot_scores_total():
    sns.set_palette(sns.color_palette('Set2', 4))
    accuracy_data = {}
    models = ['pseudo_markov', 'submod_f_0_l_50_k_50', 'modular_features_0', 'submod_f_gauss_0.15_k_10_l_0_k_30']
    for model in models:
        accuracy_data[model] = 100*rank_results_pandas('path_set_100_no_singles', model, 0)
    dataset = pd.DataFrame(accuracy_data)
    ax = sns.barplot(data=dataset, order=models, ci=95)
    ax.set_title('Model accuracy scores')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Model')
    ticks = ['Markov (H)', 'FLDC (L=50, K=50)', 'Modular', 'FFLDC (L=0, K=30)']
    ax.set_xticklabels(ticks)
    plt.savefig(os.path.join(
       constants.IMAGE_PATH, 'score_100_no_singles_with_features.eps'),
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
