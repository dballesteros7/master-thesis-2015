from __future__ import division, print_function

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
           const double *unaries_noise,
           double *weights, double *unaries, double *n_logz,
           size_t n, size_t m);

int* sample(const double *probabilities, int n, int n_samples, int *out_size);
""")
_lib = _ffi.verify('#include "train.h"',
                   sources=["train.cpp"],
                   include_dirs=['.'],
                   extra_compile_args=['-O3', '-DNDEBUG', '-std=c++11'])


def fast_sample(probabilities, n_samples):
    out_size = _ffi.new("int *")
    time_s = time.time()
    result = _lib.sample(
        _ffi.cast("const double *", probabilities.ctypes.data),
        np.size(probabilities),
        n_samples,
        out_size)
    samples = []
    current = []
    time_s = time.time()
    for i in range(out_size[0]):
        x = result[i]
        if x == -1:
            samples.append(current)
            current = []
        else:
            current.append(x)
    return samples


class Trainer:
    def __init__(self, model_data, noise_data, unaries_noise,
                 unaries=None, n_items=None, dim=5):
        data = []
        for i, subset in enumerate(chain(model_data, noise_data)):
            subset = list(subset)
            assert min(subset) >= 0
            assert len(subset) > 0
            label = 1 if i < len(model_data) else 0
            if data:
                data.append(-1)
            data.append(label)
            data.extend(subset)

        self.data = _ffi.new("long []", data)
        self.data_size = len(data)
        self.orig_data = model_data
        self.orig_nois = noise_data

        assert n_items is not None
        self.n = n_items
        self.dim = dim
        self.weights = 1e-3 * np.asarray(np.random.rand(*(self.n, dim)), dtype=np.float64)
        self.unaries_noise = np.array(unaries_noise, dtype=np.float64)
        self.unaries = np.copy(self.unaries_noise)
        self.iteration = 0
        self.n_logz = np.array([-np.sum(np.log(1 + np.exp(self.unaries)))], dtype=np.float64)

    def train(self, n_steps, eta_0, power, plot=False):
        step = self.data_size
        from matplotlib import pyplot as plt
        n_steps *= 1

        values = []
        values.append(self.objective())
        for i in range(0, step * n_steps, step):
            print(100. * i / (step * n_steps), '%')
            time_s = time.time()
            print('iter start')
            assert self.weights.shape == (self.n, self.dim)
            assert np.size(self.unaries) == self.n
            _lib.train(
                self.data, self.data_size, step,
                eta_0, power, i,
                _ffi.cast("const double *", self.unaries_noise.ctypes.data),
                _ffi.cast("double *", self.weights.ctypes.data),
                _ffi.cast("double *", self.unaries.ctypes.data),
                _ffi.cast("double *", self.n_logz.ctypes.data),
                self.n, self.dim)
            print('iter done in ', time.time() - time_s, 'seconds')
            if plot:
                values.append(self.objective())
        if plot:
            plt.plot(values, 'bo--')
            plt.show()

    def objective(self):
        from ml_novel_nonexp_nce import ModularFun, DiversityFun, NCE
        f_noise = ModularFun(range(self.n), self.unaries_noise)
        f_model = DiversityFun(range(self.n), self.dim)
        f_model.n_logz = self.n_logz
        f_model.W = self.weights
        f_model.utilities = self.unaries
        return NCE(f_model, f_noise)._objective(self.orig_data, self.orig_nois)
