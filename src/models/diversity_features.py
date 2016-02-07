import time
from itertools import chain

from sklearn.utils import shuffle
from scipy import special

import numpy as np

import constants
from models.features import BasicFeatures, Features, IdentityFeatures, \
    BasicFeaturesExtended, BasicFeaturesNoNormalized
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


def store_to_file(n_items: int, features: np.ndarray,
                  data_samples: np.ndarray, noise_factor: int,
                  output_file_path: str, output_noise_path: str):
    noise = ModularWithFeatures(n_items, features)
    noise.train(data_samples)
    with open(output_file_path, 'w') as output_file:
        n_data = data_samples.shape[0]
        noise_samples = noise.sample(noise_factor * n_data)
        n_data += noise_samples.shape[0]
        total = 0
        for sample in chain(data_samples, noise_samples):
            total += len(sample) + 1
        output_file.write('{},{}\n'.format(total, n_data))
        for sample in data_samples:
            output_file.write('1,')
            output_file.write(','.join([str(x) for x in sample]))
            output_file.write('\n')
        for sample in noise_samples:
            output_file.write('0,')
            output_file.write(','.join([str(x) for x in sample]))
            output_file.write('\n')
    with open(output_noise_path, 'w') as output_file:
        output_file.write(','.join(
            [str(x) for x in noise.feature_weights]))
        output_file.write('\n')


def process_data_and_store(dataset_name: str, features: Features):
    print('Storing noise and data for C++ processing.')
    for fold in range(1, constants.N_FOLDS + 1):
        print('Fold {}'.format(fold))
        loaded_data = file.load_set_data(
            constants.TRAIN_DATA_PATH_TPL.format(
                fold=fold, dataset=dataset_name))
        store_to_file(constants.N_ITEMS, features.as_array(),
                      loaded_data, noise_factor=constants.NCE_NOISE_FACTOR,
                      output_file_path=constants.NCE_DATA_PATH_TPL.format(
                              dataset=dataset_name, index=features.index,
                              fold=fold),
                      output_noise_path=constants.NCE_NOISE_PATH_TPL.format(
                              dataset=dataset_name, index=features.index,
                              fold=fold))


def load_and_evaluate(dataset_name: str, n_items: int, features: Features):
    full_stats = []
    for fold in range(1, constants.N_FOLDS + 1):
        for dim in range(1, 11):
            model = DiversityFeatures(n_items, features.as_array(), dim)
            model.load_from_file(constants.NCE_OUT_PATH_TPL.format(
                dataset=dataset_name, fold=fold, dim=dim,
                index=features.index))
            loaded_test_data = file.load_set_data(
                constants.PARTIAL_DATA_PATH_TPL.format(
                    fold=fold, dataset=constants.DATASET_NAME))
            target_path = constants.RANKING_MODEL_PATH_TPL.format(
                dataset=constants.DATASET_NAME, fold=fold,
                model='submod_f_{}_d_{}'.format(features.index, dim))
            with open(target_path, 'w') as output_file:
                for subset in loaded_test_data:
                    result = model.propose_set_item(subset, use_new=True)
                    output_file.write(','.join(str(item) for item in result))
                    output_file.write('\n')
            stats = model.stats['suggest']
            full_stats.append(stats[0] / stats[1])
    print('The evaluation speed is {}\u03bcs per sample.'.format(
                    np.mean(np.array(full_stats)) * 1e6))


def main():
    features = BasicFeaturesExtended(constants.DATASET_NAME,
                                     n_items=constants.N_ITEMS,
                                     m_features=4)
    features.load_from_file()
    process_data_and_store(constants.DATASET_NAME, features)
    features.store_for_training()

if __name__ == '__main__':
    main()
