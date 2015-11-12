import datetime
import logging
import random
from collections import defaultdict

from storage.cluster_storage import ClusterStorage
from storage.photo_storage import PhotoStorage


class PathFinder:
    def __init__(self):
        self.cluster_storage = ClusterStorage()
        self.photo_storage = PhotoStorage()

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
            grouped_photos = self.sort_and_group_by_day(photos)
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

    @staticmethod
    def sort_and_group_by_day(photos):
        def parse_datetaken(photo):
            return datetime.datetime.strptime(photo['datetaken'], '%Y-%m-%d %H:%M:%S')
        photos.sort(key=parse_datetaken)
        grouped_by_day = defaultdict(list)
        for photo in photos:
            date_taken = parse_datetaken(photo).date()
            grouped_by_day[date_taken].append(photo['_id'])
        logging.debug(grouped_by_day)
        return grouped_by_day

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(asctime)s:%(funcName)s:%(module)s:%(message)s')
    finder = PathFinder()
    paths = finder.find_random_path('zurich', '100m')
    logging.info(paths)
