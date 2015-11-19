import logging

from pymongo import MongoClient


class ClusterStorage:
    def __init__(self):
        client = MongoClient()
        self.collection = client.flickrdata.clusters

    def get_clusters(self, city_name, bandwidth):
        result = self.collection.find({
            'city_name': city_name,
            'bandwidth': bandwidth
        })
        return list(result)

    def insert_clusters(self, city_name, bandwidth, entries, cluster_centers, cluster_labels):
        logging.info('Collecting cluster data.')
        clusters = []
        unique_users_per_cluster = []
        for cluster_center in cluster_centers:
            clusters.append({
                'latitude': cluster_center[0],
                'longitude': cluster_center[1],
                'photos': [],
                'unique_users': 0,
                'number_of_photos': 0,
                'city_name': city_name,
                'bandwidth': bandwidth
            })
            unique_users_per_cluster.append(set())

        for cluster_label, entry in zip(cluster_labels, entries):
            if cluster_label == -1:
                continue
            clusters[cluster_label]['photos'].append(entry['_id'])
            clusters[cluster_label]['number_of_photos'] += 1
            unique_users_per_cluster[cluster_label].add(entry['owner'])
        for cluster, unique_users in zip(clusters, unique_users_per_cluster):
            cluster['unique_users'] = len(unique_users)

        self.collection.insert_many(clusters, ordered=False)
        return clusters

    def get_cluster_for_photo(self, photo_id, city_name, bandwidth):
        return self.collection.find_one({
            'photos': photo_id,
            'city_name': city_name,
            'bandwidth': bandwidth
        })

    def get_top_ten_clusters(self, city_name, bandwidth):
        return self.collection.find({
            'city_name': city_name,
            'bandwidth': bandwidth
        }, sort=[('number_of_photos', -1), ('unique_users', -1)], limit=10)
