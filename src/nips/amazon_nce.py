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

try:
    import cPickle as pickle
except ImportError:
    import pickle

DATA_PATH = 'amazon_data'
RESULT_PATH = 'models'
N_SAMPLES = 5

n_folds = 10
dim = 2

datasets = ['safety']

nce_logsubmod = {}
em_dpp = {}
ll_dpp_picard = {}
ml_modular = {}

time_logsubmod = {}
time_dpp = {}
time_dpp_picard = {}
time_modular = {}

n_items = {}


if __name__ == '__main__':
    for dataset, fold in product(datasets, range(n_folds)):
        np.random.seed(20150820)

        print('-' * 30)
        print('dataset: %s (fold %d)' % (dataset, fold + 1))

        sets_train_f = '{0}/1_100_100_100_{1}_regs_train_fold_{2}.csv'.format(
            DATA_PATH, dataset, fold + 1)
        sets_test_f = '{0}/1_100_100_100_{1}_regs_test_fold_{2}.csv'.format(
            DATA_PATH, dataset, fold + 1)
        names_f = '{0}/1_100_100_100_{1}_item_names.txt'.format(
            DATA_PATH, dataset)
        result_submod_f = '{0}/{1}_submod_d_{2}_fold_{3}.pkl'.format(RESULT_PATH, dataset, dim, fold + 1)
        result_mod_f = '{0}/{1}_mod_fold_{2}.pkl'.format(RESULT_PATH, dataset, fold + 1)

        # names = load_amazon_names(names_f)
        data = load_amazon_data(sets_train_f)
        data_test = load_amazon_data(sets_test_f)

        random.shuffle(data)

        # print('names', len(names))
        print('# of data items (train): ', len(data))
        print('# of data items (test) : ', len(data_test))

        results_model = pickle.load(open(result_submod_f, 'rb'))
        results_modular = pickle.load(open(result_mod_f, 'rb'))

        f_model = results_model['model']
        time_logsubmod_ = results_model['time']

        f_noise = results_modular['model']
        time_modular_ = results_modular['time']

        n_items[dataset] = len(f_model.V)

        # TODO Add switch for computing the LL exactly/approximately

        # Compute LLs
        #print("TRAIN: Estimated nLL: ", -f_
        #model._estimate_LL(data)

        test = 0
        #test = -f_model._estimate_LL(data_test)
        #print("TEST:  Estimated nLL: ", test)

        test = -f_model._estimate_LL_exact(data_test)
        print("TEST:      Exact nLL: ", test)
        # import sys
        # sys.exit(1)



