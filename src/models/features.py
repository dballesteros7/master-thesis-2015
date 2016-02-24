import numpy as np

import constants


class Features:
    index = -1

    def __init__(self, dataset_name: str, n_items: int, m_features: int):
        self.keys = {}
        self.n_items = n_items
        self.m_features = m_features
        self.dataset_name = dataset_name
        self.features = np.empty((n_items, m_features))

    def as_array(self) -> np.ndarray:
        raise NotImplementedError

    def load_from_file(self):
        path = constants.ITEMS_FEATURE_PATH_TPL.format(
            dataset=self.dataset_name, i=self.index)
        with open(path, 'r') as input_file:
            first_line = input_file.readline()
            tokens = first_line.strip().split(',')
            self.keys = dict((token, index)
                             for index, token in enumerate(tokens))
            for item_index, line in enumerate(input_file):
                tokens = line.strip().split(',')
                for feature_index, token in enumerate(tokens):
                    self.features[item_index, feature_index] = float(token)

    def store_for_training(self):
        path = constants.NCE_FEATURES_PATH_TPL.format(
            dataset=self.dataset_name, index=self.index)
        with open(path, 'w') as output_file:
            output_file.write('{},{}\n'.format(self.n_items, self.m_features))
            for item_features in self.as_array():
                output_file.write(','.join(
                        [str(item) for item in item_features]))
                output_file.write('\n')


class IdentityFeatures(Features):
    index = 0

    def load_from_file(self):
        assert self.n_items == self.m_features
        self.features = np.identity(self.n_items)

    def as_array(self):
        return np.copy(self.features)


class BasicFeatures(Features):
    index = 1

    def as_array(self):
        return (self.features - np.min(self.features, axis=0)) /\
               (np.max(self.features, axis=0) - np.min(self.features, axis=0))


class BasicFeaturesNoNormalized(Features):
    index = 1

    def load_from_file(self):
        path = constants.ITEMS_FEATURE_PATH_TPL.format(
            dataset=self.dataset_name, i=self.index)
        with open(path, 'r') as input_file:
            for item_index, line in enumerate(input_file):
                tokens = line.strip().split(',')
                for feature_index, token in enumerate(tokens):
                    self.features[item_index, feature_index] = float(token)

    def as_array(self):
        return np.copy(self.features)


class BasicFeaturesExtended(Features):
    index = 2

    def load_from_file(self):
        super(BasicFeaturesExtended, self).load_from_file()
        self.features = np.hstack((self.features, np.identity(self.n_items)))
        self.m_features += self.n_items

    def as_array(self):
        # Scale latitude, longitude
        normalized_features = (
            (self.features[:, :4] - np.min(self.features[:, :4], axis=0)) /
            (np.max(self.features[:, :4], axis=0) -
             np.min(self.features[:, :4], axis=0)))

        return np.hstack((normalized_features, np.identity(self.n_items)))
