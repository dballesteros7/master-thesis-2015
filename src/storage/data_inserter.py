import json
import logging
import os

from pymongo import MongoClient


class DataInserter:
    def __init__(self):
        client = MongoClient()
        self.database = client.flickrdata

    def read_jsons(self, path='/local/workspace/master-thesis-2015/data/raw'):
        all_files = os.listdir(path)
        logging.info('Processing {} files.'.format(len(all_files)))
        for file_name in all_files:
            city_name = file_name.split('-')[0]
            with open(os.path.join(path, file_name), 'r') as input:
                entries = json.load(input)
                for entry in entries:
                    entry['city_name'] = city_name
                logging.info('Inserting {} entries for {}.'.format(len(entries), city_name))
                self.database.photos.insert_many(entries)
                logging.info('Finished inserting entries.')

    def insert_clusters(self, city_name, entries, clusters_centers, cluster_labels, settings):
        logging.info('Collecting cluster data.')
        clusters = []
        for cluster_center in clusters_centers:
            clusters.append({
                'latitude': cluster_center[0],
                'longitude': cluster_center[1],
                'photos': []
            })
        for cluster_label, entry in zip(cluster_labels, entries):
            if cluster_label == -1:
                continue
            clusters[cluster_label]['photos'].append(entry)
        good_clusters = []
        for cluster in clusters:
            unique_users = set()
            for index, entry in enumerate(cluster['photos']):
                unique_users.add(entry['owner'])
                cluster['photos'][index] = entry['_id']
            if len(unique_users) >= 3:
                good_clusters.append(cluster)

        cluster_result = {
            'city_name': city_name,
            'bandwidth': settings['bandwidth'],
            'min_bin_freq': settings['min_bin_freq'],
            'clusters': good_clusters
        }
        self.database.cluster_results.insert_one(cluster_result)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(funcName)s:%(module)s:%(message)s')
    inserter = DataInserter()
    inserter.read_jsons()
