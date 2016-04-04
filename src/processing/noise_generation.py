from itertools import chain

import numpy as np

import constants
from models.features import GaussianFeatures, Features, IdentityFeatures, \
    BasicFeaturesNoNormalized, GaussianExtended
from models.modular import ModularWithFeatures
from utils import file


def store_to_file(n_items: int, features: np.ndarray,
                  data_samples: np.ndarray, noise_factor: int,
                  output_file_path: str, output_noise_path: str):
    noise = ModularWithFeatures(n_items, features)
    noise.train(data_samples)
    with open(output_file_path, 'w') as output_file:
        n_data = data_samples.shape[0]
        noise_samples = noise.sample(noise_factor * n_data, use_real_probs=True)
        n_data += noise_samples.shape[0]
        total = 0
        for sample in chain(data_samples, noise_samples):
            total += len(sample) + 1
        output_file.write('{},{}\n'.format(total, n_data))
        for sample in data_samples:
            if len(sample):
                output_file.write('1,')
                output_file.write(','.join([str(x) for x in sample]))
            else:
                output_file.write('1')
            output_file.write('\n')
        for sample in noise_samples:
            if len(sample):
                output_file.write('0,')
                output_file.write(','.join([str(x) for x in sample]))
            else:
                output_file.write('0')
            output_file.write('\n')
    with open(output_noise_path, 'w') as output_file:
        output_file.write(','.join(
            [str(x) for x in noise.feature_weights]))
        output_file.write('\n')
        output_file.write(','.join(
            [str(x) for x in noise.exact_utilities]))
        output_file.write('\n')


def process_data_and_store(dataset_name: str, features: Features, n_items: int):
    print('Storing noise and data for C++ processing.')
    for noise_factor in range(20, 22, 2):
        for fold in range(1, constants.N_FOLDS + 1):
            print('Fold {}'.format(fold))
            loaded_data = file.load_set_data(
                constants.TRAIN_DATA_PATH_TPL.format(
                    fold=fold, dataset=dataset_name))
            store_to_file(n_items, features.as_array(),
                          loaded_data, noise_factor=noise_factor,
                          output_file_path=constants.NCE_DATA_PATH_TPL.format(
                              dataset=dataset_name, index=features.index,
                              fold=fold, noise_factor=noise_factor),
                          output_noise_path=constants.NCE_NOISE_PATH_TPL.format(
                              dataset=dataset_name, index=features.index,
                              fold=fold))


def main():
    n_items = 6
    dataset_name = constants.DATASET_NAME_TPL.format('synthetic_4')
    features = BasicFeaturesNoNormalized(dataset_name, n_items=n_items,
                                         m_features=3)
    features.load_from_file()
    process_data_and_store(dataset_name, features, n_items)
    features.store_for_training()


if __name__ == '__main__':
    main()
