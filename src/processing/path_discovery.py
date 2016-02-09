import datetime
import logging
import random
from collections import defaultdict
from collections import OrderedDict

import numpy as np
from sklearn.cross_validation import KFold

import constants
from clustering.meanshift import cluster_photos
from storage.cluster_storage import ClusterStorage
from storage.path_storage import PathStorage
from storage.photo_storage import PhotoStorage
from utils.google_api import GoogleApi


class PathFinder:
    def __init__(self):
        self.cluster_storage = ClusterStorage()
        self.photo_storage = PhotoStorage()
        self.path_storage = PathStorage()

    def find_random_path(self, city_name, bandwidth, n_paths=10):
        cluster_result = self.cluster_storage.get_cluster(city_name=city_name, bandwidth=bandwidth)
        logging.info('Loaded cluster.')
        photos_by_cluster = {}
        for index, cluster in enumerate(cluster_result['clusters']):
            for photo_id in cluster['photos']:
                photos_by_cluster[photo_id] = index
        logging.info('Indexed photos in cluster.')
        users = self.photo_storage.get_top_users(limit=100, city_name=city_name)
        logging.info('Retrieved {} users.'.format(len(users)))
        logging.debug(users)
        selected_users = random.sample(users, k=n_paths)
        paths = []
        logging.info('Sampled users.')
        for user in selected_users:
            photos = self.photo_storage.get_photos_for_user(user=user['_id'])
            grouped_photos = sort_and_group_by_day(photos)
            day = random.choice(list(grouped_photos.keys()))
            selected_photos = grouped_photos[day]
            path = []
            logging.info('Using {} photos for {}.'.format(len(selected_photos), day))
            for photo_id in selected_photos:
                cluster_index = photos_by_cluster[photo_id]
                cluster = cluster_result['clusters'][cluster_index]
                if len(path) == 0 or path[-1][0] != cluster_index:
                    path.append((cluster_index, cluster['latitude'], cluster['longitude']))
            paths.append(path)
        return paths

    def find_and_store_all_paths(self, city_name, bandwidth='100m',
                                 min_unique_users=1, min_cluster_photos=1):
        all_photos = self.photo_storage.get_photos_for_city(
            city_name=city_name)
        logging.info('Loading cursor for photo collection.')
        all_paths = defaultdict(OrderedDict)
        logging.info('Starting iteration over photos.')
        counter = 0
        included = 0
        discarded = 0
        cluster_centers, labels = cluster_photos(all_photos, bandwidth)

        photo_cluster_mapping = {}
        cluster_counts = defaultdict(int)
        for photo, label in zip(all_photos, labels):
            photo_cluster_mapping[photo['id']] = label
            if label < 0:
                continue
            cluster_counts[label] += 1

        n_top = 50

        top_clusters = sorted(
            cluster_counts.items(), key=lambda x: x[1], reverse=True)[:n_top]
        top_clusters = set(x[0] for x in top_clusters)

        for photo in all_photos:
            cluster_label = photo_cluster_mapping[photo['id']]
            if cluster_label in top_clusters:
                all_paths[(parse_datetaken(photo), photo['owner'])][
                    cluster_label] = True
                included += 1
            else:
                discarded += 1
            counter += 1
            if counter % 5000 == 0:
                logging.info('Processed {} photos so far, {} included and {} discarded'.format(
                    counter, included, discarded))
        logging.info('Done processing {} photos.'.format(counter))
        paths = []
        for date, owner in all_paths:
            paths.append(list(all_paths[(date, owner)].keys()))

        all_clusters = {}
        next_cluster_index = 0
        path_sets = []
        for path in paths:
            path_set = []
            for cluster_id in path:
                if cluster_id not in all_clusters:
                    all_clusters[cluster_id] = next_cluster_index
                    next_cluster_index += 1
                path_set.append(str(all_clusters[cluster_id]))
            path_sets.append(path_set)

        with open(constants.ITEMS_DATA_PATH_TPL.format(
                dataset=constants.DATASET_NAME_TPL.format(n_top)),
                'w') as items_file:
            sorted_clusters = sorted(all_clusters.items(), key=lambda x: x[1])
            for cluster_id, _ in sorted_clusters:
                cluster_info = cluster_centers[cluster_id]
                top_places = GoogleApi.get_places(
                        cluster_info[0], cluster_info[1])
                values = [str(cluster_info[0]),
                          str(cluster_info[1]),
                          str(cluster_counts[cluster_id])]
                items_file.write(','.join(values))
                items_file.write(',')
                items_file.write(';'.join([result['name']
                                           for result in top_places[:5]]))
                items_file.write('\n')

        data = np.array(path_sets)
        kf = KFold(len(path_sets), n_folds=10, shuffle=True)
        for idx, (train_index, test_index) in enumerate(kf):
            with open(constants.DATA_PATH_TPL.format(
                    dataset=constants.DATASET_NAME_TPL.format(n_top),
                    type='train', fold=idx + 1), 'w') as output_train, \
                    open(constants.DATA_PATH_TPL.format(
                        dataset=constants.DATASET_NAME_TPL.format(n_top),
                        type='test', fold=idx + 1), 'w') as output_test:
                for path_set in data[train_index]:
                    output_train.write(','.join(path_set) + '\n')
                for path_set in data[test_index]:
                    output_test.write(','.join(path_set) + '\n')


def parse_datetaken(photo):
    return datetime.datetime.strptime(
        photo['datetaken'], '%Y-%m-%d %H:%M:%S').date()


def sort_and_group_by_day(photos):
    photos.sort(key=parse_datetaken)
    grouped_by_day = defaultdict(list)
    for photo in photos:
        date_taken = parse_datetaken(photo)
        grouped_by_day[date_taken].append(photo['_id'])
    logging.debug(grouped_by_day)
    return grouped_by_day


if __name__ == '__main__':
    np.random.seed(constants.SEED)
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s:%(asctime)s:%(funcName)s:%(module)s:%(message)s')
    finder = PathFinder()
    finder.find_and_store_all_paths('zurich')
    #finder.write_path_csv('zurich', '100m', -1, -1)
