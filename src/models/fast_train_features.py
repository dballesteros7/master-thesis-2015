import os
from itertools import chain

import time
import numpy as np
from cffi import FFI

# The datatype that we use for computation. We always convert the given data
# to a double array to make sure we have enough bits for precise computation.
_double = np.dtype('d')


_ffi = FFI()
_ffi.cdef(r"""
    void train(const long *data, size_t data_size, long n_steps,
               double eta_0, double power, int start_step,
               const double *features,
               double *b_weights, double *a_weights, double *n_logz,
               size_t n_items, size_t l_dim, size_t m_feat);
""")
_lib = _ffi.verify('#include "train_features.h"',
                   sources=[os.path.join(
                       os.path.dirname(__file__), 'train_features.cpp')],
                   include_dirs=[os.path.dirname(__file__)],
                   extra_compile_args=['-O3', '-DNDEBUG', '-std=c++11'])


class TrainerFeatures:
    def __init__(self, model_data, noise_data, a_noise,
                 features, n_items, l_dims, m_features):
        data = []
        rolled_features = []
        for i, subset in enumerate(chain(model_data, noise_data)):
            subset = list(subset)
            assert min(subset) >= 0
            assert len(subset) > 0
            label = 1 if i < len(model_data) else 0
            if data:
                data.append(-1)
            data.append(label)
            data.extend(subset)
        for row in features:
            for value in row:
                rolled_features.append(value)

        self.data = _ffi.new('long []', data)
        self.features = _ffi.new('double []', rolled_features)
        self.data_size = len(data)
        self.orig_data = model_data
        self.orig_nois = noise_data
        self.orig_features = features

        self.n_items = n_items
        self.l_dims = l_dims
        self.m_features = m_features
        self.b_weights = 1e-3 * np.asarray(
            np.random.rand(*(self.m_features, self.l_dims)),
            dtype=np.float64)
        self.a_weights = np.array(a_noise, dtype=np.float64)
        self.unaries = np.dot(np.array(self.orig_features), self.a_weights)
        self.iteration = 0
        self.n_logz = np.array([-np.sum(np.log(1 + np.exp(self.unaries)))],
                               dtype=np.float64)

    def train(self, n_steps, eta_0, power):
        step = self.data_size
        n_steps *= 1

        for i in range(0, step * n_steps, step):
            print(100. * i / (step * n_steps), '%')
            time_s = time.time()
            print('iter start')
            _lib.train(
                self.data, self.data_size, step,
                eta_0, power, i,
                self.features,
                _ffi.cast("double *", self.b_weights.ctypes.data),
                _ffi.cast("double *", self.a_weights.ctypes.data),
                _ffi.cast("double *", self.n_logz.ctypes.data),
                self.n_items, self.l_dims, self.m_features)
            print('iter done in ', time.time() - time_s, 'seconds')
