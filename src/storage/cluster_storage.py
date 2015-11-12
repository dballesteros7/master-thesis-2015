import logging

from pymongo import MongoClient


class ClusterStorage:
    def __init__(self):
        client = MongoClient(port=4321)
        self.collection = client.flickrdata.cluster_results

    def get_cluster(self, city_name, bandwidth):
        result = self.collection.find_one({
            'city_name': city_name,
            'bandwidth': bandwidth
        })
        return result

    def insert_cluster(self, city_name, bandwidth, entries, cluster_centers, cluster_labels):
        logging.info('Collecting cluster data.')
        clusters = []
        unique_users_per_cluster = []
        for cluster_center in cluster_centers:
            clusters.append({
                'latitude': cluster_center[0],
                'longitude': cluster_center[1],
                'photos': [],
                'unique_users': 0,
                'number_of_photos': 0
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

        cluster_result = {
            'city_name': city_name,
            'bandwidth': bandwidth,
            'clusters': clusters
        }
        inserted_id = self.collection.insert_one(cluster_result).inserted_id
        cluster_result['_id'] = inserted_id
        return cluster_result
