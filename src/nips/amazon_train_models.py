import os
import random
import numpy as np
from itertools import product
import time
from multiprocessing import Pool

import constants
from models.fast_train_features import TrainerFeatures
from models.modular import ModularWithFeatures
from nips.amazon_utils import load_amazon_data
from nips.fast_train import Trainer
from nips.ml_novel_nonexp_nce import DiversityFun, ModularFun

try:
    import cPickle as pickle
except ImportError:
    import pickle

DATA_PATH = constants.DATA_PATH
RESULT_PATH = os.path.join(DATA_PATH, 'models')
N_SAMPLES = 5
F_NOISE = 20  # Number of noise samples = F_NOISE * (number of training samples)
N_CPUS = 4  # How many processors to parallelize the sampling on.

n_folds = 10
dim_range = [2]

datasets = ['path_set']


def sample_once(f_noise):
    print("N_SAMPLES = ", N_SAMPLES)
    return f_noise.sample(N_SAMPLES // N_CPUS)


def train_model(data, n_items=None):
    global N_SAMPLES

    if n_items is None:
        n_items = max(max(x) for x in data) + 1

    print('total items:', n_items)
    # print('    in test:', (max(max(x) for x in data_test) + 1))

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
    print('<utilities>')
    print(s)
    print('</utilities>')

    # features = []
    # for idx in range(n_items):
    #     features.append([])
    #     for idx_2 in range(n_items):
    #         if idx == idx_2:
    #             features[idx].append(1)
    #         else:
    #             features[idx].append(0)
    #
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

    f_model = DiversityFun(list(range(n_items)), dim)
    f_model.utilities = np.copy(f_noise.s)

    start = time.time()
    # trainer = TrainerFeatures(data, data_noise, f_noise.feature_weights,
    #                           features, n_items, l_dims=dim,
    #                           m_features=n_items)
    # trainer.train(10, 0.01, 0.1)
    # f_model.W = trainer.b_weights
    # f_model.utilities = trainer.a_weights
    trainer = Trainer(data, data_noise, f_noise.s, n_items=n_items, dim=dim)
    trainer.train(10, 0.01, 0.1, plot=False)
    f_model.W = trainer.weights
    f_model.utilities = trainer.unaries
    f_model.n_logz = trainer.n_logz
    print(f_model.W)
    print(f_model.utilities)
    print(f_model.n_logz)
    end = time.time()
    print("TRAINING TOOK ", (end - start))

    end = time.time()
    time_nce = end - start

    return f_model, f_noise, time_nce, time_modular


if __name__ == '__main__':
    for dataset, fold in product(datasets, range(1, 2)):
        print('-' * 30)
        print('dataset: {} (fold {})'.format(dataset, fold))
        for dim in dim_range:
            np.random.seed(20150820)

            print("--> dims = %d" % dim)

            sets_train_f = os.path.join(DATA_PATH,
                                        '{}_train_fold_{}.csv'.format(dataset,
                                                                      fold))
            result_submod_f = os.path.join(RESULT_PATH,
                                           '{}_submod_d_{}_fold_{}.pkl'.format(
                                               dataset, dim, fold))
            result_mod_f = os.path.join(RESULT_PATH,
                                        '{}_mod_fold_{}.pkl'.format(dataset,
                                                                    fold))

            data = load_amazon_data(sets_train_f)

            random.shuffle(data)

            n_items = 10

            print('# of data items (train): ', len(data))

            f_model, f_noise, time_logsubmod_, time_modular_ = train_model(data,
                                                                           n_items=n_items)
            results_model = {'model': f_model, 'time': time_logsubmod_}
            results_modular = {'model': f_noise, 'time': time_modular_}

            # dump
            pickle.dump(results_model, open(result_submod_f, 'wb'))
            pickle.dump(results_modular, open(result_mod_f, 'wb'))
