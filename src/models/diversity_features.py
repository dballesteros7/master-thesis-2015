import time
from itertools import chain

from sklearn.utils import shuffle
from scipy import special

import numpy as np

import constants
from models.features import BasicFeatures, Features, IdentityFeatures, \
    BasicFeaturesExtended, BasicFeaturesNoNormalized, GaussianFeatures
from models.modular import ModularWithFeatures
from utils import file

W_eps = 1e-3


class DiversityFeatures:
    def __init__(self, n_items: int, features: np.ndarray, l_dims: int):
        assert features.shape[0] == n_items

        # Problem parameters
        self.n_items = n_items
        self.features = features
        self.l_dims = l_dims
        self.m_feats = features.shape[1]

        # Model parameters
        self.a_weights = np.zeros(self.m_feats)
        self.b_weights = np.zeros((self.m_feats, self.l_dims))
        self.n_logz = 0.

        # Composite model parameters
        self.weights = np.dot(self.features, self.b_weights)
        self.utilities = np.dot(self.features, self.a_weights)

        # Gradient memory allocation
        self._gradient_a = np.zeros_like(self.a_weights)
        self._gradient_b = np.zeros_like(self.b_weights.transpose())

        # Statistics
        self.stats = {
            'gradient_a_time': [0.0, 0],
            'gradient_b_time': [0.0, 0],
            'params_time': [0.0, 0],
            'evaluation': [0.0, 0],
            'suggest': [0.0, 0]
        }

    def update_composite_parameters(self):
        start = time.time()
        np.dot(self.features, self.b_weights, out=self.weights)
        np.dot(self.features, self.a_weights, out=self.utilities)
        elapsed = time.time() - start
        self.stats['params_time'][0] += elapsed
        self.stats['params_time'][1] += 1

    def __call__(self, s: np.ndarray) -> np.ndarray:
        start = time.time()
        weights_view = self.weights[s, :]
        logprob = self.n_logz + \
               np.sum(self.utilities[s]) + \
               np.sum((np.max(weights_view, axis=0) -
                       np.sum(weights_view, axis=0)))
        elapsed = time.time() - start
        self.stats['evaluation'][0] += elapsed
        self.stats['evaluation'][1] += 1
        return logprob

    def gradient_a(self, s: np.ndarray) -> np.ndarray:
        start = time.time()
        np.sum(self.features[s, :], axis=0, out=self._gradient_a)
        elapsed = time.time() - start
        self.stats['gradient_a_time'][0] += elapsed
        self.stats['gradient_a_time'][1] += 1

    def gradient_b(self, s: np.ndarray) -> np.ndarray:
        start = time.time()
        weights_view = self.weights[s, :]
        max_indexes = s[np.argmax(weights_view, axis=0)]
        np.subtract(self.features[max_indexes, :], self._gradient_a,
                    out=self._gradient_b)
        elapsed = time.time() - start
        self.stats['gradient_b_time'][0] += elapsed
        self.stats['gradient_b_time'][1] += 1

    def propose_set_item(self, to_complete: np.ndarray,
                         use_new: bool = False) -> np.ndarray:
        start = time.time()
        if use_new:
            gains = np.zeros(self.n_items)

            gains += self.utilities
            gains -= np.sum(self.weights, axis=1)

            current_max_per_d = np.max(self.weights[to_complete, :], axis=0)
            max_per_dimension = np.repeat(
                    current_max_per_d.reshape((1, self.l_dims, 1)),
                    self.n_items, axis=0)
            gains += np.sum(
                np.max(
                    np.concatenate(
                        (self.weights.reshape((self.n_items, self.l_dims, 1)),
                         max_per_dimension),
                        axis=2),
                    axis=2),
                axis=1)

            gains[to_complete] = -np.inf
            result = np.argsort(gains)[to_complete.shape[0]:][::-1]
        else:
            current_value = np.exp(self(to_complete))
            other_items = set(range(self.n_items)) - set(to_complete)
            gains = np.zeros(self.n_items)
            gains[to_complete] = -np.inf
            for item in other_items:
                other_value = np.exp(self(np.append(to_complete, item)))
                gains[item] = other_value - current_value
            result = np.argsort(gains)[to_complete.shape[0]:][::-1]
        elapsed = time.time() - start
        self.stats['suggest'][0] += elapsed
        self.stats['suggest'][1] += 1
        return result

    def load_from_file(self, input_path: str):
        with open(input_path) as input_file:
            lines = list(input_file)

            n_logz = float(lines[0].strip())
            a_weights = [float(x) for x in lines[1].strip().split(',')]
            b_weights = []
            for line in lines[2:]:
                row = []
                for item in line.strip().split(','):
                    row.append(float(item))
                b_weights.append(row)

            a_weights = np.array(a_weights)
            b_weights = np.array(b_weights)

            self.a_weights = a_weights
            self.b_weights = b_weights
            self.n_logz = n_logz
        self.update_composite_parameters()


class NCETrainer:
    def __init__(self, model: DiversityFeatures, noise: ModularWithFeatures):
        self.model = model
        self.noise = noise
        self.stats = {
            'gradient_update': [0.0, 0]
        }

    def train(self, data_samples: np.ndarray, noise_factor: int,
              n_iter: int, eta_0: float, iter_power: float):
        # Sample from the modular model.
        self.noise.train(data_samples)
        n_data = len(data_samples)
        noise_samples = self.noise.sample(noise_factor * n_data)

        # Copy data
        labels = np.hstack((
            np.ones_like(data_samples), np.zeros_like(noise_samples)))
        all_data = np.hstack((data_samples, noise_samples))

        # Initialize parameters
        self.model.a_weights = np.copy(self.noise.feature_weights)
        self.model.b_weights = W_eps * np.random.random_sample(
            (self.model.m_feats, self.model.l_dims))
        self.model.update_composite_parameters()
        self.model.n_logz = -np.sum(np.log1p(np.exp(self.model.utilities)))

        log_nu = np.log(noise_samples.shape[0] / data_samples.shape[0])
        abs_iter = 0

        a_gradient = np.zeros_like(self.model.a_weights)
        b_gradient = np.zeros_like(self.model.b_weights)
        for iter in range(n_iter):
            print('{} out of {}.'.format(iter + 1, n_iter))
            iter_data, iter_labels = shuffle(all_data, labels)
            for sample, label in zip(iter_data, iter_labels):
                abs_iter += 1
                self.model.update_composite_parameters()

                p_model = self.model(sample)
                p_noise = self.noise(sample)
                learning_rate = eta_0 / np.power(abs_iter, iter_power)
                factor = learning_rate * (
                    label - special.expit(p_model - p_noise - log_nu))

                self.model.gradient_a(sample)
                self.model.gradient_b(sample)
                start = time.time()
                np.multiply(factor, self.model._gradient_a,
                            out=a_gradient)
                np.multiply(factor, self.model._gradient_b.transpose(),
                            out=b_gradient)
                self.model.a_weights += a_gradient
                self.model.b_weights += b_gradient
                self.model.n_logz += factor
                neg_indexes = self.model.b_weights < 0
                self.model.b_weights[neg_indexes] =\
                    1e-3 * np.random.random_sample(np.sum(neg_indexes))
                elapsed = time.time() - start
                self.stats['gradient_update'][0] += elapsed
                self.stats['gradient_update'][1] += 1

        self.model.update_composite_parameters()
