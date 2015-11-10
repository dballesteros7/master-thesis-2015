import logging

import numpy as np
from sklearn.cluster import MeanShift

from storage.data_inserter import DataInserter
from storage.data_loader import DataLoader

BANDWITH_100M = 0.0009009009  # Approximate bandwidth


def cluster_photos(entries):
    np_locations = np.array([(float(entry['latitude']), float(entry['longitude'])) for entry in entries])
    clusterer = MeanShift(bandwidth=BANDWITH_100M, bin_seeding=True, min_bin_freq=10)
    logging.info('Clustering started.')
    clusterer.fit(np_locations)
    logging.info('Clustering finished.')
    return (clusterer.cluster_centers_, clusterer.labels_, {
        'bandwidth': BANDWITH_100M,
        'min_bin_freq': 10
    })

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(funcName)s:%(module)s:%(message)s')
    loader = DataLoader()
    entries = loader.load_entries('london')
    result = cluster_photos(entries)
    entries.rewind()
    inserter = DataInserter()
    inserter.insert_clusters('london', entries, *result)