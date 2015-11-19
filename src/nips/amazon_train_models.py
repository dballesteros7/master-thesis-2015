from __future__ import division
from __future__ import print_function

import codecs
import os
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

DATA_PATH = '/local/workspace/master-thesis-2015/data'
RESULT_PATH = os.path.join(DATA_PATH, 'models')
N_SAMPLES = 5
F_NOISE = 20  # Number of noise samples = F_NOISE * (number of training samples)
N_CPUS = 4  # How many processors to parallelize the sampling on.

n_folds = 10
dim_range = [2, 5, 10, 20]

datasets = ['path_set']

def sample_once(f_noise):
    print("N_SAMPLES = ", N_SAMPLES)
    return f_noise.sample(N_SAMPLES // N_CPUS)


def train_model(data, n_items=None):
    global N_SAMPLES

    if n_items is None:
        n_items = max(max(x) for x in data) + 1

    print('total items:', n_items)
    #print('    in test:', (max(max(x) for x in data_test) + 1))

    start = time.time()
    # Count item frequencies.
    item_marg = np.zeros(n_items) + 1  # pseudo count
    for subset in data:
        for element in subset:
            item_marg[element] += 1
    item_marg /= len(data)
    print('<item_margs>')
    print(item_marg)
    print('</item_margs>')
    s = -np.log(1. / item_marg - 1.)  # The modular term.

    f_noise = ModularFun(list(range(n_items)), np.copy(s))
    end = time.time()
    time_modular = end - start

    N_SAMPLES = F_NOISE * len(data)

    print('sampling')
    if True:
        pool = Pool(N_CPUS)

        samples_all = pool.map(sample_once, [f_noise] * N_CPUS)
        data_noise = []
        for samples in samples_all:
            data_noise.extend(samples)

        pool.close()
        pool.join()

        print('saving samples')
        pickle.dump(data_noise, open('all_samples_million.pkl', 'wb'))
        print('sampling done')
    else:
        print('unpickling samples')
        data_noise = pickle.load(open('all_samples_million.pkl', 'rb'))
        print('done')
    # Remove empty sets.
    data_noise = list(filter(lambda x: len(x) > 0, data_noise))

    # print("--- FITTED MODULAR MODEL ---")
    # print("TRAIN: Estimated nLL: ", - f_noise._estimate_LL(data))
    # print("TEST:  Estimated nLL: ", - f_noise._estimate_LL(data_test))

    f_model = DiversityFun(list(range(n_items)), dim)
    f_model.utilities = np.copy(f_noise.s)

    useCPP = True
    if not useCPP:
        nce = NCE(f_model, f_noise)
        nce.learn_sgd(data, data_noise, n_iter=10, compute_LL=True)
    else:
        start = time.time()
        trainer = Trainer(data, data_noise, f_noise.s, n_items=n_items, dim=dim)
        trainer.train(10, 0.01, 0.1, plot=False)
        f_model.W = trainer.weights
        f_model.utilities = trainer.unaries
        f_model.n_logz = trainer.n_logz
        end = time.time()
        print("TRAINING TOOK ", (end - start))

    end = time.time()
    time_nce = end - start

    return f_model, f_noise, time_nce, time_modular


if __name__ == '__main__':
    for dataset, fold in product(datasets, range(1, n_folds + 1)):
        print('-' * 30)
        print('dataset: {} (fold {})'.format(dataset, fold))
        for dim in dim_range:
            np.random.seed(20150820)

            print("--> dims = %d" % dim)

            sets_train_f = os.path.join(DATA_PATH, '{}_train_fold_{}.csv'.format(dataset, fold))
            result_submod_f = os.path.join(RESULT_PATH, '{}_submod_d_{}_fold_{}.pkl'.format(dataset, dim, fold))
            result_mod_f = os.path.join(RESULT_PATH, '{}_mod_fold_{}.pkl'.format(dataset, fold))

            data = load_amazon_data(sets_train_f)

            random.shuffle(data)

            n_items = 10

            print('# of data items (train): ', len(data))

            f_model, f_noise, time_logsubmod_, time_modular_ = train_model(data, n_items=n_items)
            results_model = {'model': f_model, 'time': time_logsubmod_}
            results_modular = {'model': f_noise, 'time': time_modular_}

            # dump
            pickle.dump(results_model, open(result_submod_f, 'wb'))
            pickle.dump(results_modular, open(result_mod_f, 'wb'))


