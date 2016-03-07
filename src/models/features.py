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


class GaussianFeatures(Features):
    def __init__(self, dataset_name: str, n_items: int, m_features: int, sigma: float):
        super(GaussianFeatures, self).__init__(dataset_name, n_items, m_features)
        self.sigma = sigma
        self.index = 'gauss_{}_k_{}'.format(sigma, m_features)

    def load_from_file(self):
        path = constants.ITEMS_DATA_PATH_TPL.format(
            dataset=self.dataset_name)
        with open(path, 'r') as input_file:
            first_line = input_file.readline()
            tokens = first_line.strip().split(',')
            self.keys = dict((token, index)
                             for index, token in enumerate(tokens))
            locations = []
            for item_index, line in enumerate(input_file):
                tokens = line.strip().split(',')
                latitude = float(tokens[self.keys['latitude']])
                longitude = float(tokens[self.keys['longitude']])
                locations.append((latitude, longitude))

            locations = np.array(locations)
            locations = (locations - np.min(locations, axis=0)) /\
                (np.max(locations, axis=0) - np.min(locations, axis=0))

            for i in range(self.n_items):
                for j in range(self.m_features):
                    if i == j:
                        self.features[i, j] = 1
                    x_diff = np.power(locations[j][0] - locations[i][0], 2)
                    y_diff = np.power(locations[j][1] - locations[i][1], 2)
                    self.features[i, j] = np.exp(
                        -(x_diff + y_diff) / (2 * np.power(self.sigma, 2)))

    def as_array(self):
        return np.copy(self.features)

if __name__ == '__main__':
    f = GaussianFeatures('path_set_10', 10, 5, 0.3)
    f.load_from_file()
    print(f.features)
