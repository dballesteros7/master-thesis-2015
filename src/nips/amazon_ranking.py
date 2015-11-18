from __future__ import division
from __future__ import print_function

import codecs
import random
import numpy as np
import IPython
from itertools import product
from matplotlib import pyplot as plt
import scipy.io
import time

# from submod.functions.facility_location import FacilityLocation
# from submod.maximization.greedy import greedy_order
# from submod.maximization.randomized_2 import optimize as maximize
# from submod.functions.mutators import add_modular

from fast_train import Trainer
from ml_novel_nonexp_nce import *
from amazon_utils import *
from multiprocessing import Pool

## Load pre-trained models for the amazon data and perform product
## recommendation

try:
    import cPickle as pickle
except ImportError:
    import pickle

DATA_PATH = 'amazon_data'
MODEL_PATH = 'models'
RANKING_PATH = 'ranking_test'
N_CPUS = 4

n_folds = 10
folds_range = range(0, n_folds)
# folds_range = [9]
n_propose = 1
f_model = None

datasets = ['safety']
dim_range = [2]

def get_proposal(Sorig, sample='topN'):
    """
    Get the proposal for an additional item from f, given S
    """
    f = f_model  # use global model
    # print("SANITY... MODEL TYPE: ", f.tpe)

    if isinstance(f, ModularFun):
        assert False  # This should happen somewhere else

    assert n_propose == 1  # modified to work for this case only

    V = f.V
    N = len(V)

    S = Sorig[:]
    Vred = np.array(list(set(V).difference(set(S))))
    f_S = f(S)

    probs = []
    gains = []
    vals = f.all_singleton_adds(S)
    vals = np.delete(vals, S)
    gains = vals - f_S
    # for i in Vred:
    #     f_Si = f(list(set(S).union(set([i]))))
    #     probs.append(f_Si)
    #     gains.append(f_Si - f_S)
    probs = np.exp(vals - lse(vals))

    order = Vred[np.argsort(probs)[::-1]]
    probs = np.sort(probs)[::-1]

    return {'order': order, 'probs': probs}

def save_to_csv(filename, lst):
    with open(filename, "wt") as f:
        for sample in lst:
            for i, k in enumerate(sample):
                f.write("%d" % k)
                if i < len(sample) - 1:
                    f.write(",")
            f.write("\n")


if __name__ == '__main__':
    for dataset, fold in product(datasets, folds_range):
        np.random.seed(20150820)

        print('-' * 30)
        print('dataset: %s (fold %d)' % (dataset, fold + 1))

        result_ranking_partial_f = '{0}/{1}_partial_fold_{2}.pkl'.format(RANKING_PATH, dataset, fold + 1)

        # load all models
        f_models = dict()
        for dim in dim_range:
            result_submod_f = '{0}/{1}_submod_d_{2}_fold_{3}.pkl'.format(MODEL_PATH, dataset, dim, fold + 1)
            results_model = pickle.load(open(result_submod_f, 'rb'))
            f_tmp_model = results_model['model']
            f_models[dim] = f_tmp_model

        result_ranking_gt_f = '{0}/{1}_gt_fold_{2}.pkl'.format(RANKING_PATH, dataset, fold + 1)
        result_ranking_partial_f = '{0}/{1}_partial_fold_{2}.pkl'.format(RANKING_PATH, dataset, fold + 1)

        data_ranking = load_amazon_ranking_data(result_ranking_partial_f)

        print("Performing ranking.")

        list_prop_model = dict()
        for dim in dim_range:
            list_prop_model[dim] = []
        V = f_tmp_model.V

        for dim in dim_range:
            print("... dim=%d" % dim)
            f_model = f_models[dim]
            pool = Pool(N_CPUS)
            prop_model_dicts = pool.map(get_proposal, data_ranking) # get_proposal(f_models[dim], s_partial[:], n_propose=n_propose)
            pool.close()
            pool.join()

            # extract proposals from list of dictionaries
            prop_model = []
            for d in prop_model_dicts:
                prop_model.append(d['order'])

            # bookkeeping
            list_prop_model[dim] = prop_model

        # for i, s_partial in enumerate(data_ranking):
        #     if i % 100 == 0:
        #         print("%d/%d ..." % (i, len(data_ranking)))

        #     for dim in dim_range:
        #         prop_model = get_proposal(f_models[dim], s_partial[:], n_propose=n_propose)
        #         list_prop_model[dim].append(prop_model)

        for dim in dim_range:
            result_ranking_submod_f = '{0}/{1}_submod_d_{2}_fold_{3}.pkl'.format(RANKING_PATH, dataset, dim, fold + 1)
            save_to_csv(result_ranking_submod_f, list_prop_model[dim])


    # IPython.embed()
