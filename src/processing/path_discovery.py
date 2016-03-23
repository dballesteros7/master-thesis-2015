import datetime
import logging
import random
from collections import defaultdict
from collections import OrderedDict

import numpy as np
import sys
from scipy import spatial
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

    def unclustered_paths(self, all_photos, dataset_name: str):
        all_paths = defaultdict(list)
        seen_locations = set()
        unique_photos = []
        for photo in all_photos:
            location = (photo['latitude'], photo['longitude'])
            if location not in seen_locations:
                seen_locations.add(location)
                unique_photos.append(photo)

        for photo in unique_photos:
            photo_date = parse_datetaken(photo)
            photo_owner = photo['owner']

            key = (photo_date, photo_owner)
            all_paths[key].append(photo)
        filtered_paths = defaultdict(list)
        filtered_photos = []
        for key in all_paths:
            if len(all_paths[key]) > 3:
                filtered_paths[key] = all_paths[key]
                filtered_photos.extend(all_paths[key])
        sample = np.random.choice(filtered_photos, 10000, replace=False)
        sampled_paths = defaultdict(list)
        sorted_sample = sorted(
            sample, key=lambda x: datetime.datetime.strptime(
                x['datetaken'], '%Y-%m-%d %H:%M:%S'))
        for photo in sorted_sample:
            photo_date = parse_datetaken(photo)
            photo_owner = photo['owner']
            key = (photo_date, photo_owner)
            sampled_paths[key].append(photo)

        final_paths = []
        for key in sampled_paths:
            if len(sampled_paths[key]) > 1:
                final_paths.append(sampled_paths[key])

        items = []
        indexed_paths = []
        index = 0
        for path in final_paths:
            indexed_path = []
            for photo in path:
                items.append(photo)
                indexed_path.append(str(index))
                index += 1
            indexed_paths.append(indexed_path)

        with open(constants.ITEMS_DATA_PATH_TPL.format(
                dataset=constants.DATASET_NAME_TPL.format(dataset_name)),
                'w') as items_file:
            first_photo = items[0]
            sorted_keys = sorted(first_photo.keys())
            items_file.write('{}\n'.format(','.join(sorted_keys)))
            for photo in items:
                values = [photo[key] for key in sorted_keys]
                items_file.write('{}\n'.format(','.join(values)))

        with open(constants.ALL_DATA_PATH_TPL.format(
                dataset=constants.DATASET_NAME_TPL.format(dataset_name)),
                'w') as paths_file:
            for path in indexed_paths:
                paths_file.write('{}\n'.format(','.join(path)))

        return items, indexed_paths


def produce_top_clusters(all_photos, n_items, bandwidth='100m'):
    cluster_centers, labels = cluster_photos(all_photos, bandwidth)
    cluster_counts = defaultdict(int)
    photo_count = 0
    for photo, label in zip(all_photos, labels):
        if label < 0:
            continue
        cluster_counts[label] += 1
        photo_count += 1

    top_clusters = sorted(
        cluster_counts.items(), key=lambda x: x[1], reverse=True)
    if n_items:
        top_clusters = top_clusters[:n_items]
    top_clusters = set(x[0] for x in top_clusters)

    cluster_label_to_idx = {}
    for idx, cluster_label in enumerate(top_clusters):
        cluster_label_to_idx[cluster_label] = idx

    ordered_cluster_counts = []
    for cluster_label in top_clusters:
        ordered_cluster_counts.append(cluster_counts[cluster_label])

    with open(constants.CLUSTER_FILE.format(id='k_{}'.format(n_items)), 'w') as cluster_file:
        for cluster_label in top_clusters:
            cluster = cluster_centers[cluster_label]
            count = cluster_counts[cluster_label]
            cluster_file.write('{},{},{}\n'.format(cluster[0], cluster[1], count))

    with open(constants.CLUSTER_ASSIGNATION_FILE.format(id='k_{}'.format(n_items)), 'w') as cluster_assign_file:
        for photo, label in zip(all_photos, labels):
            if label in cluster_label_to_idx:
                cluster_assign_file.write('{},{}\n'.format(photo['id'], cluster_label_to_idx[label]))

    clusters = []
    for cluster_label in top_clusters:
        clusters.append(cluster_centers[cluster_label])

    cluster_assignment = {}
    for photo, label in zip(all_photos, labels):
        if label in cluster_label_to_idx:
            cluster_assignment[photo['id']] = cluster_label_to_idx[label]

    return clusters, cluster_assignment, ordered_cluster_counts


def find_and_store_all_paths(dataset_name, all_photos, cluster_assignment):
    all_paths = defaultdict(OrderedDict)

    for photo in all_photos:
        if photo['id'] in cluster_assignment:
            all_paths[(parse_datetaken(photo), photo['owner'])][
                cluster_assignment[photo['id']]] = True

    path_sets = list([str(item) for item in path]
                     for path in all_paths.values())

    with open(constants.ALL_DATA_PATH_TPL.format(
            dataset=constants.DATASET_NAME_TPL.format(dataset_name)),
            'w') as paths_file:
        for path in path_sets:
            paths_file.write('{}\n'.format(','.join(path)))

    no_singleton_paths = [path for path in path_sets if len(path) > 1]
    no_singleton_dataset_name = '{}_no_singles'.format(dataset_name)
    with open(constants.ALL_DATA_PATH_TPL.format(
            dataset=constants.DATASET_NAME_TPL.format(
                no_singleton_dataset_name)), 'w') as paths_file:
        for path in no_singleton_paths:
            paths_file.write('{}\n'.format(','.join(path)))

    just_pairs_path = [path for path in path_sets if len(path) == 2]
    just_pairs_dataset_name = '{}_pairs'.format(dataset_name)
    with open(constants.ALL_DATA_PATH_TPL.format(
            dataset=constants.DATASET_NAME_TPL.format(
                just_pairs_dataset_name)), 'w') as paths_file:
        for path in just_pairs_path:
            paths_file.write('{}\n'.format(','.join(path)))

    return path_sets, no_singleton_paths, just_pairs_path


def shuffle_train_and_test(dataset_name, total_set):
    data = np.array(total_set)
    kf = KFold(len(total_set), n_folds=10, shuffle=True)
    for idx, (train_index, test_index) in enumerate(kf):
        with open(constants.DATA_PATH_TPL.format(
                dataset=constants.DATASET_NAME_TPL.format(dataset_name),
                type='train', fold=idx + 1), 'w') as output_train, open(
                constants.DATA_PATH_TPL.format(
                        dataset=constants.DATASET_NAME_TPL.format(dataset_name),
                        type='test', fold=idx + 1), 'w') as output_test:
            for path_set in data[train_index]:
                output_train.write(','.join(path_set) + '\n')
            for path_set in data[test_index]:
                output_test.write(','.join(path_set) + '\n')


def calculate_features_for_unclustered_items(dataset_name, items,
                                             clusters):
    item_features = []
    for item in items:
        item_pos = np.array(
            [float(item['latitude']), float(item['longitude'])])
        features = []
        for cluster in clusters:
            np_cluster = np.array(cluster)
            features.append(
                spatial.distance.euclidean(item_pos, np_cluster))
        item_features.append(features)

    with open(constants.ITEMS_FEATURE_PATH_TPL.format(
            dataset=constants.DATASET_NAME_TPL.format(dataset_name),
            i=1), 'w') as feature_file:
        for features in item_features:
            feature_file.write(
                '{}\n'.format(','.join([str(x) for x in features])))


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


def main():
    np.random.seed(constants.SEED)
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s:%(asctime)s:%(funcName)s:%(module)s:%(message)s')
    finder = PathFinder()
    all_photos = finder.photo_storage.get_photos_for_city(city_name='zurich')
    clusters, cluster_assignment, ordered_counts = produce_top_clusters(
        all_photos, 100)
    #items, indexed_paths = finder.unclustered_paths(all_photos, dataset_name)
    #shuffle_train_and_test(dataset_name, indexed_paths)
    #calculate_features_for_unclustered_items(dataset_name, items, clusters)
    dataset_name = '100'
    paths, no_singleton_paths, just_pairs_path = find_and_store_all_paths(
        dataset_name, all_photos, cluster_assignment)
    shuffle_train_and_test(dataset_name, paths)
    shuffle_train_and_test('{}_no_singles'.format(dataset_name),
                           no_singleton_paths)
    shuffle_train_and_test('{}_pairs'.format(dataset_name),
                           just_pairs_path)


if __name__ == '__main__':
    main()
