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

if __name__ == '__main__':
    from processing.path_discovery import PathFinder
    finder = PathFinder()
    all_photos = finder.photo_storage.get_photos_for_city(city_name='zurich')
    centers_1, labels_1 = cluster_photos(all_photos, bandwidth='100m')
    centers_2, labels_2 = cluster_photos(all_photos, bandwidth='200m')
    centers_3, labels_3 = cluster_photos(all_photos, bandwidth='500m')
    centers_4, labels_4 = cluster_photos(all_photos, bandwidth='1km')
    print(len(centers_1))
    print(len(centers_2))
    print(len(centers_3))
    print(len(centers_4))
