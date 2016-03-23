import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import constants
from processing.ranking import rank_results, rank_results_pandas


def plot_heatmap():
    if os.path.exists(os.path.join(constants.DATA_PATH, '.cache_plot_4.csv')):
        dataset = pd.read_csv(os.path.join(constants.DATA_PATH, '.cache_plot_4.csv'), index_col=0)
    else:
        model_template = 'modular_features_gauss_{}_k_{}'
        index_data = [1e-4, 1e-3, 1e-2, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
        columns = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
        accuracies = []
        for m in columns:
            sigma_acc = []
            for sigma in index_data:
                model_name = model_template.format(sigma, m)
                accuracy = rank_results('path_set_100_no_singles', model_name, 0)[0][0]
                sigma_acc.append(accuracy)
            accuracies.append(sigma_acc)
        accuracies = np.array(accuracies)
        dataset = pd.DataFrame(accuracies, index=columns, columns=index_data)
        dataset.to_csv(os.path.join(constants.DATA_PATH, '.cache_plot_4.csv'), index_label='Sigma')
    ax = sns.heatmap(dataset, annot=True, fmt='.2f')
    ax.set_ylabel('$M$')
    ax.set_xlabel('$\sigma$')
    ax.set_title('Accuracy with Gaussian Features')
    # plt.savefig(os.path.join(
    #    constants.IMAGE_PATH, 'score_100_no_singles_modular_ext_sigma_features_effect.png'),
    #    bbox_inches='tight')
    plt.show()


def plot_heatmap_2():
    dataset = pd.read_csv(constants.MODULAR_MODEL_ERROR_PATH_TPL.format(
        dataset='path_set_100_no_singles', model='modular_features'),
        index_col=0).transpose()
    ax = sns.heatmap(dataset, annot=True, fmt='.2f')
    ax.set_ylabel('$M$')
    ax.set_xlabel('$\sigma$')
    ax.set_title('MSE on item utilities')
    plt.savefig(os.path.join(
       constants.IMAGE_PATH, 'score_100_no_singles_modular_mse.png'),
       bbox_inches='tight')
    plt.show()


def plot_score_progress():
    sns.set(context='poster')
    modular_acc = []
    modular_acc_err = []
    submod_acc = []
    submod_acc_err = []
    m_values = [10, 20, 40, 60, 80, 100]
    dataset_name = constants.DATASET_NAME_TPL.format('100_no_singles')
    modular_template = 'modular_features_gauss_0.05_k_{}'
    submod_template = 'submod_f_gauss_0.05_k_{}_l_20_k_20'
    for m in m_values:
        model_name = modular_template.format(m)
        acc, acc_err, _, _ = rank_results(dataset_name, model_name, 0)
        modular_acc.append(acc[0])
        modular_acc_err.append(acc_err[0])
        model_name = submod_template.format(m)
        acc, acc_err, _, _ = rank_results(dataset_name, model_name, 0)
        submod_acc.append(acc[0])
        submod_acc_err.append(acc_err[0])

    fig, ax = plt.subplots()
    mod_line = plt.errorbar(m_values, modular_acc, yerr=modular_acc_err, marker='o')
    submod_line = plt.errorbar(m_values, submod_acc, yerr=submod_acc_err, marker='^')
    ax.set_xlabel('$M$')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy with Gaussian Features ($\sigma = 0.05$)')
    ax.set_ylim([0, 20])
    ax.set_xlim([0, 105])
    ax.legend([mod_line, submod_line], ['Modular', 'FLDC ($L=20$, $K=20$)'],
              loc='upper left')
    plt.savefig(os.path.join(
       constants.IMAGE_PATH, 'score_100_no_singles_m_effect.png'),
       bbox_inches='tight')
    plt.show()


def plot_score_progress_2():
    sns.set(context='poster')
    modular_acc = []
    modular_acc_err = []
    submod_acc = []
    submod_acc_err = []
    sigma_values = [1e-3, 1e-2, 2e-2, 5e-2, 1e-1, 15e-2, 2e-1]
    dataset_name = constants.DATASET_NAME_TPL.format('100_no_singles')
    modular_template = 'modular_features_gauss_{}_k_100'
    submod_template = 'submod_f_gauss_{}_k_100_l_20_k_20'
    acc, acc_err, _, _ = rank_results(dataset_name, 'modular_features_0', 0)
    modular_acc.append(acc[0])
    modular_acc_err.append(acc_err[0])
    acc, acc_err, _, _ = rank_results(dataset_name, 'submod_f_0_l_20_k_20', 0)
    submod_acc.append(acc[0])
    submod_acc_err.append(acc_err[0])
    for s in sigma_values:
        model_name = modular_template.format(s)
        acc, acc_err, _, _ = rank_results(dataset_name, model_name, 0)
        modular_acc.append(acc[0])
        modular_acc_err.append(acc_err[0])
        model_name = submod_template.format(s)
        acc, acc_err, _, _ = rank_results(dataset_name, model_name, 0)
        submod_acc.append(acc[0])
        submod_acc_err.append(acc_err[0])
    sigma_values.insert(0, 0)
    fig, ax = plt.subplots()
    mod_line = plt.errorbar(sigma_values, modular_acc, yerr=modular_acc_err, marker='o')
    submod_line = plt.errorbar(sigma_values, submod_acc, yerr=submod_acc_err, marker='^')
    ax.set_xlabel('$\sigma$')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy with Gaussian Features ($M=100$)')
    ax.set_ylim([0, 20])
    ax.set_xlim([0, 0.22])
    ax.legend([mod_line, submod_line], ['Modular', 'FLDC ($L=20$, $K=20$)'],
              loc='upper left')
    plt.savefig(os.path.join(
       constants.IMAGE_PATH, 'score_100_no_singles_sigma_effect.png'),
       bbox_inches='tight')
    plt.show()



def plot_score():
    accuracies = []
    accuracies_error = []
    model_template = 'submod_f_gauss_{}_k_100_l_20_k_20'
    s_values = [0.05, 0.1, 0.15, 0.2, 0.25]
    for s in s_values:
        model = model_template.format(s)
        acc, acc_std, _, _ = rank_results('path_set_100_no_singles', model, 0)
        accuracies.append(acc[0])
        accuracies_error.append(acc_std)
    s_values.insert(0, 0)
    acc, acc_std, _, _ = rank_results('path_set_100_no_singles', 'submod_f_0_l_20_k_20', 0)
    accuracies.insert(0, acc[0])
    accuracies_error.insert(0, acc_std[0])
    fig, ax = plt.subplots()
    plt.errorbar(s_values, accuracies, yerr=accuracies_error, marker='o')
    ax.set_title('Model accuracy ($L = 20$, $K = 20$, $M = 100$)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('$\sigma$')
    ax.set_ylim([0, 20])
    # plt.savefig(os.path.join(
    #    constants.IMAGE_PATH, 'score_100_no_singles_sigma_effect.png'),
    #    bbox_inches='tight')
    plt.show()

def plot_score_2():
    accuracies = []
    accuracies_error = []
    model_template = 'submod_f_gauss_0.05_k_{}_l_20_k_20'
    k_values = [10, 20, 40, 60, 80, 85, 90, 95, 98, 100]
    for k in k_values:
        model = model_template.format(k)
        acc, acc_std, _, _ = rank_results('path_set_100_no_singles', model, 0)
        accuracies.append(acc[0])
        accuracies_error.append(acc_std)
    fig, ax = plt.subplots()
    plt.errorbar(k_values, accuracies, yerr=accuracies_error, marker='o')
    ax.set_title('Model accuracy ($L = 20$, $K = 20$, $\sigma = 0.05$)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('$M$')
    ax.set_ylim([0, 20])
    plt.savefig(os.path.join(
       constants.IMAGE_PATH, 'score_100_no_singles_m_effect.png'),
       bbox_inches='tight')
    plt.show()

def plot_scores_total():
    sns.set_palette(sns.color_palette('Set2', 4))
    accuracy_data = {}
    models = ['modular_features_0', 'modular_features_gauss_ext_0.05_k_100',
              'modular_features_gauss_ext_0.05_k_10',
              'submod_f_0_l_20_k_20', 'submod_f_gauss_ext_0.05_k_100_l_20_k_20',
              'submod_f_gauss_ext_0.05_k_10_l_20_k_20']
    for model in models:
        accuracy_data[model] = 100*rank_results_pandas('path_set_100_no_singles', model, 0)
    dataset = pd.DataFrame(accuracy_data)
    ax = sns.barplot(data=dataset, order=models, ci=95)
    ax.set_title('Model accuracy scores')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Model')
    #ticks = ['Markov (H)', 'FLDC (L=50, K=50)', 'Modular', 'FFLDC (L=0, K=30)']
    #ax.set_xticklabels(ticks)
    # plt.savefig(os.path.join(
    #    constants.IMAGE_PATH, 'score_100_no_singles_with_features.eps'),
    #    bbox_inches='tight')
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
    #plot_score_progress()
    plot_scores_total()
    #plot_heatmap()
