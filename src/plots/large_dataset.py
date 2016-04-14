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
    pseudo_markov_results = rank_results_pandas(dataset_name, 'pseudo_markov', 0)
    proximity_results = rank_results_pandas(dataset_name, 'proximity_r', 0)

    results_column = np.concatenate(
        (modular_results, proximity_results, pseudo_markov_results))
    model_column = np.repeat(
        ['Modular', 'Heuristic Markov', 'Proximity'], constants.N_FOLDS)

    dataset = pd.DataFrame({
        'scores': 100 * results_column,
        'model': model_column
    })

    ax = sns.barplot(x='model', y='scores', data=dataset, ci=95)
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy (\%)')
    ax.set_title('Accuracy of baseline models')

    plt.savefig(os.path.join(
       constants.IMAGE_PATH, 'baseline_models_100.eps'),
       bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    plots.setup()
    sns.set_palette(sns.color_palette('Set1', 4))
    scores_baselines()
