import pickle

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

import constants
from nips.ml_novel_nonexp_nce import DiversityFun


def plot_weights(div_model: DiversityFun):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    weights = div_model.W
    color_set = ax.matshow(weights, cmap=cm.GnBu_r, interpolation='none')
    plt.colorbar(color_set)
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(10))
    ax.set_yticklabels(('HB', 'Bürkliplatz',
                        'Bellevueplatz', 'Paradeplatz',
                        'Grossmünster', 'Zoo',
                        'Lindenhof', 'Rathaus',
                        'Fraumünster', 'Hallenstadion'))
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Locations')
    plt.show()


def main():
    input_model_path = constants.MODEL_PATH_TPL.format(
        dataset='path_set', model='submod_d_5', fold='1')
    with open(input_model_path, 'rb') as input_model_file:
        input_model = pickle.load(input_model_file)
        plot_weights(input_model['model'])

if __name__ == '__main__':
    import sys
    sys.exit(main())
