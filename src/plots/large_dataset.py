import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import constants
import plots
from processing.ranking import rank_results_pandas


def scores_baselines():
    dataset_name = 'path_set_100_no_singles'
    modular_results = rank_results_pandas(dataset_name, 'modular_features_0', 0)
    markov_results = rank_results_pandas(dataset_name, 'markov', 0)
    pseudo_markov_results = rank_results_pandas(dataset_name, 'pseudo_markov', 0)
    proximity_results = rank_results_pandas(dataset_name, 'proximity', 0)
    proximity_ordered_results = rank_results_pandas(dataset_name, 'proximity_ordered', 0)

    results_column = np.concatenate((modular_results, markov_results, pseudo_markov_results, proximity_results, proximity_ordered_results))
    model_column = np.repeat(['Log-modular', 'Markov', 'Heuristic Markov', 'Proximity', 'Proximity Ordered'], constants.N_FOLDS)

    dataset = pd.DataFrame({
        'scores': 100 * results_column,
        'model': model_column
    })

    ax = sns.barplot(x='model', y='scores', data=dataset, ci=95,
                     palette=sns.color_palette('Set1'))
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy (\%)')
    ax.set_title('Accuracy of baseline models')

    plt.savefig(os.path.join(
       constants.IMAGE_PATH, 'baseline_models_100.eps'),
       bbox_inches='tight')
    plt.show()

def accuracy_fldc():
    l_range = np.arange(5, 35, 5)
    k_range = np.arange(5, 35, 5)

    l_column = []
    k_column = []
    score_column = []
    dataset_name = 'path_set_100_no_singles'
    model_tpl = 'submod_f_0_l_{}_k_{}_iter_1000_noise_5_eta_0.1_adagrad_1'
    for l_dim in l_range:
        for k_dim in k_range:
            model_name = model_tpl.format(l_dim, k_dim)
            results = rank_results_pandas(dataset_name, model_name, 0)
            score_column.extend(results)
            l_column.extend([l_dim] * len(results))
            k_column.extend([k_dim] * len(results))
    dataset = pd.DataFrame({
        'score': 100 * np.array(score_column),
        'l': l_column,
        'k': k_column
    })
    dataset = dataset.groupby(['l', 'k'])['score'].mean().unstack(1)

    ax = sns.heatmap(dataset, vmin=10, vmax=20, annot=True, fmt='.1f',
                     linewidths=.5)
    ax.set_xlabel('$K$')
    ax.set_ylabel('$L$')
    #ax.set_ylabel(r'Accuracy (\%)')
    ax.set_title(r'Accuracy (\%)')
    #ax.set_ylim([0, 45])

    # modular_mean = 100*np.mean(rank_results_pandas(dataset_name, 'modular_features_0', 0))
    # plt.plot(ax.get_xlim(), [modular_mean, modular_mean], linestyle='dotted')
    #
    plt.savefig(os.path.join(
        constants.IMAGE_PATH, 'large_fldc_dims.eps'),
        bbox_inches='tight')

    plt.show()

def comparison_large():
    dataset_name = 'path_set_100_no_singles'
    modular_results = rank_results_pandas(dataset_name, 'modular_features_0', 0)
    markov_results = rank_results_pandas(dataset_name, 'markov', 0)
    pseudo_markov_results = rank_results_pandas(dataset_name, 'pseudo_markov', 0)
    proximity_results = rank_results_pandas(dataset_name, 'proximity', 0)
    proximity_ordered_results = rank_results_pandas(dataset_name, 'proximity_ordered', 0)
    fldc_results = rank_results_pandas(dataset_name, 'submod_f_0_l_5_k_15_iter_1000_noise_5_eta_0.1_adagrad_1', 0)

    results_column = np.concatenate((markov_results, proximity_ordered_results, pseudo_markov_results, modular_results, proximity_results, fldc_results))
    model_column = np.repeat(['Markov', 'Prox. Ordered', 'Heuristic Markov', 'Log-modular', 'Proximity', 'FLDC ($L=5,K=15$)'], constants.N_FOLDS)
    type_column = np.repeat(['Set', 'Path', 'Path', 'Set', 'Path', 'Set'], constants.N_FOLDS)

    dataset = pd.DataFrame({
        'scores': 100 * results_column,
        'model': model_column,
        'type': type_column
    })

    ax = sns.barplot(x='model', y='scores', data=dataset, ci=95,
                     palette=sns.color_palette('Set1'))
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy (\%)')
    ax.set_title('Accuracy comparison for the large dataset.')
    #
    plt.savefig(os.path.join(
       constants.IMAGE_PATH, 'all_models_100.eps'),
       bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    plots.setup()
    sns.set_palette(sns.color_palette('Set1', 4))
    comparison_large()
