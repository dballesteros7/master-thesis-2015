import logging

import numpy as np
from sklearn.cluster import MeanShift

BANDWITDHS = {
    '100m': 1/1110,
    '200m': 1/555,
    '500m': 1/222,
    '1km': 1/111
}


def cluster_photos(entries, bandwidth='100m'):
    bandwidth = BANDWITDHS[bandwidth]
    np_locations = np.array([(float(entry['latitude']), float(entry['longitude'])) for entry in entries])
    clusterer = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1,
                          cluster_all=False)
    logging.info('Clustering started.')
    clusterer.fit(np_locations)
    logging.info('Clustering finished.')
    return clusterer.cluster_centers_, clusterer.labels_
