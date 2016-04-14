import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import constants
import plots
from processing.ranking import rank_results, rank_results_pandas


def scores_modular():
    singles_results = rank_results_pandas('path_set_10', 'modular_features_0', 0)
    no_singles_results = rank_results_pandas('path_set_10_no_singles', 'modular_features_0', 0)

    singles_results_100 = rank_results_pandas('path_set_100', 'modular_features_0', 0)
    no_singles_results_100 = rank_results_pandas('path_set_100_no_singles', 'modular_features_0', 0)

    results_column = np.concatenate((singles_results, singles_results_100, no_singles_results, no_singles_results_100))
    type_column = ['With singletons']*(constants.N_FOLDS*2) + ['Without singletons']*(2*constants.N_FOLDS)
    dataset_column = np.tile(np.repeat(['$|V| = 10$', '$|V|=100$'], constants.N_FOLDS), 2)

    dataset = pd.DataFrame({
        'type': type_column,
        'scores': 100 * results_column,
        'size': dataset_column
    })

    ax = sns.barplot(x='size', y='scores', hue='type', data=dataset, ci=95,
                     palette=sns.color_palette('Set1'))
    ax.set_xlabel('Dataset size')
    ax.set_ylabel('Accuracy (\%)')
    ax.set_title('Effect of singletons in accuracy')

    legend = ax.get_legend()
    legend.set_title('Dataset type')

    plt.savefig(os.path.join(
        constants.IMAGE_PATH, 'singletons_modular_comparison.eps'),
        bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    plots.setup()

    scores_modular()
