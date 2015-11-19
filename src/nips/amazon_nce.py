from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
from itertools import product

from ml_novel_nonexp_nce import *
from amazon_utils import *

try:
    import cPickle as pickle
except ImportError:
    import pickle

DATA_PATH = '/local/workspace/master-thesis-2015/data/'
RESULT_PATH = os.path.join(DATA_PATH, 'models')
N_SAMPLES = 5

n_folds = 10
dim = 2

datasets = ['path_set']

nce_logsubmod = {}
em_dpp = {}
ll_dpp_picard = {}
ml_modular = {}

time_logsubmod = {}
time_dpp = {}
time_dpp_picard = {}
time_modular = {}


if __name__ == '__main__':
    for dataset, fold in product(datasets, range(1, n_folds + 1)):
        np.random.seed(20150820)

        print('-' * 30)
        print('dataset: %s (fold %d)' % (dataset, fold))

        sets_train_f = os.path.join(DATA_PATH, '{}_train_fold_{}.csv'.format(dataset, fold))
        sets_test_f = os.path.join(DATA_PATH, '{}_test_fold_{}.csv'.format(dataset, fold))
        result_submod_f = os.path.join(RESULT_PATH, '{}_submod_d_{}_fold_{}.pkl'.format(dataset, dim, fold))
        result_mod_f = os.path.join(RESULT_PATH, '{}_mod_fold_{}.pkl'.format(dataset, fold))

        data = load_amazon_data(sets_train_f)
        data_test = load_amazon_data(sets_test_f)

        random.shuffle(data)

        print('# of data items (train): ', len(data))
        print('# of data items (test) : ', len(data_test))

        results_model = pickle.load(open(result_submod_f, 'rb'))
        results_modular = pickle.load(open(result_mod_f, 'rb'))

        f_model = results_model['model']
        time_logsubmod_ = results_model['time']

        f_noise = results_modular['model']
        time_modular_ = results_modular['time']

        test = -f_model._estimate_LL_exact(data_test)
        print("TEST:      Exact nLL: ", test)
