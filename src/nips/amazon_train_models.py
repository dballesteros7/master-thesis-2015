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
F_NOISE = 20  # Number of noise samples = F_NOISE * (number of training samples)
N_CPUS = 4  # How many processors to parallelize the sampling on.

n_folds = 10
dim_range = [2]

datasets = ['safety']

def sample_once(f_noise):
    print("N_SAMPLES = ", N_SAMPLES)
    return f_noise.sample(N_SAMPLES // N_CPUS)


def train_model(data, n_items=None):
    global N_SAMPLES

    if n_items is None:
        n_items = max(max(x) for x in data) + 1

    print('total items:', n_items)
    print('    in test:', (max(max(x) for x in data_test) + 1))

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
    for dataset, fold in product(datasets, range(n_folds)):
        print('-' * 30)
        print('dataset: %s (fold %d)' % (dataset, fold + 1))
        for dim in dim_range:
            np.random.seed(20150820)

            print("--> dims = %d" % dim)

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

            n_items = max([max(max(x) for x in data), max(max(x) for x in data_test)]) + 1

            # print('names', len(names))
            print('# of data items (train): ', len(data))
            print('# of data items (test) : ', len(data_test))

            f_model, f_noise, time_logsubmod_, time_modular_ = train_model(data, n_items=n_items)
            results_model = {'model': f_model, 'time': time_logsubmod_}
            results_modular = {'model': f_noise, 'time': time_modular_}

            # dump
            pickle.dump(results_model, open(result_submod_f, 'wb'))
            pickle.dump(results_modular, open(result_mod_f, 'wb'))

            # TODO Add switch for computing the LL exactly/approximately

            # Compute LLs
            #print("TRAIN: Estimated nLL: ", -f_
            #model._estimate_LL(data)

            test = 0
            #test = -f_model._estimate_LL(data_test)
            #print("TEST:  Estimated nLL: ", test)

            # test = -f_model._estimate_LL_exact(data_test)
            # print("TEST:      Exact nLL: ", test)
            # import sys
            # sys.exit(1)

            #plt.matshow(f_model.W)
            #plt.show(block=False)
            #IPython.embed()
            #import sys
            #sys.exit(1)
            #IPython.embed()

