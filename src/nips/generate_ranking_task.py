from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
from itertools import product


# from submod.functions.facility_location import FacilityLocation
# from submod.maximization.greedy import greedy_order
# from submod.maximization.randomized_2 import optimize as maximize
# from submod.functions.mutators import add_modular

from ml_novel_nonexp_nce import *
from amazon_utils import *

## Generates the data for the ranking task

try:
    import cPickle as pickle
except ImportError:
    import pickle

DATA_PATH = '/local/workspace/master-thesis-2015/data'
MODEL_PATH = os.path.join(DATA_PATH, 'models')
RANKING_PATH = os.path.join(DATA_PATH, 'ranking_test')

n_folds = 10
max_len = 6 # maxium length of subsets to consider

datasets = ['path_set']

def get_proposal(f, Sorig):
    """
    Get the proposal for an additional item from f, given S
    """

    if isinstance(f, ModularFun):
        # TODO Fast computation of proposal
        utils = np.copy(f.s)
        utils[Sorig] = - float('inf')
        t = np.argsort(utils)
        t = t[len(Sorig):]
        return t[::-1]
    else:
        assert False


def save_to_csv(filename, lst):
    with open(filename, "wt") as f:
        for sample in lst:
            for i, k in enumerate(sample):
                f.write("%d" % k)
                if i < len(sample) - 1:
                    f.write(",")
            f.write("\n")


if __name__ == '__main__':
    for dataset, fold in product(datasets, range(1, n_folds + 1)):
        np.random.seed(20150820)

        print('-' * 30)
        print('dataset: %s (fold %d)' % (dataset, fold))

        sets_test_f = os.path.join(DATA_PATH, '{}_test_fold_{}.csv'.format(dataset, fold))

        result_mod_f = os.path.join(MODEL_PATH, '{}_mod_fold_{}.pkl'.format(dataset, fold))
        result_ranking_gt_f = os.path.join(RANKING_PATH, '{}_gt_fold_{}.pkl'.format(dataset, fold))
        result_ranking_partial_f = os.path.join(RANKING_PATH, '{}_partial_fold_{}.pkl'.format(dataset, fold))
        result_ranking_random_f = os.path.join(RANKING_PATH, '{}_random_fold_{}.pkl'.format(dataset, fold))
        result_ranking_mod_f = os.path.join(RANKING_PATH, '{}_mod_fold_{}.pkl'.format(dataset, fold))

        data_test = load_amazon_data(sets_test_f)

        print('# of data items (test) : ', len(data_test))

        # Load models
        results_modular = pickle.load(open(result_mod_f, 'rb'))

        f_noise = results_modular['model']
        V = f_noise.V

        list_prop_modular = []
        list_gt = []
        list_partial = []  # keep track of the partial shown sets (as interface for Matlab)
        list_random = []
        for i, sample in enumerate(data_test):
            if len(sample) < 2:  # only consider sets with at least two items
                continue

            if i % 100 == 0:
                print("%d/%d" % (i, len(data_test)))

            for n_show in [len(sample) - 1]: #[int(np.ceil(len(sample) / 2))]:
                if len(sample) > max_len:
                    print("skipping sample (too large)")
                    continue
                # range(1, len(sample)):
                #n_show = int(np.ceil(len(sample) / 2))
                for A in combinations(list(range(len(sample))), n_show):
                    sample_new = np.array(sample)[list(A)]
                    sample_new = sample_new.tolist()
                    gt = np.array(sample)[list(set(range(len(sample))).difference(list(A)))]
                    gt = gt.tolist()

                    # get proposal from modular distribution
                    prop_modular = get_proposal(f_noise, sample_new[:])

                    # propose random set of given size
                    t = list(set(range(len(V))).difference(sample_new[:]))
                    random.shuffle(t)
                    prop_random = t

                    # bookkeeping
                    list_prop_modular.append(prop_modular)
                    list_gt.append(gt)
                    list_partial.append(sample_new)
                    list_random.append(prop_random)

        save_to_csv(result_ranking_mod_f, list_prop_modular)
        save_to_csv(result_ranking_gt_f, list_gt)
        save_to_csv(result_ranking_partial_f, list_partial)
        save_to_csv(result_ranking_random_f, list_random)
