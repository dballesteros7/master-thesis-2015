from pymongo import MongoClient


class DataLoader:
    def __init__(self):
        client = MongoClient()
        self.database = client.flickrdata

    def load_entries(self, city_name):
        return self.database.photos.find({'city_name': city_name}, projection=['latitude', 'longitude', 'owner', '_id'])

    def load_cluster(self, city_name):
        return self.database.cluster_results.find_one({'city_name': city_name,
                                                       'settings.bandwidth': 5/1110,
                                                       'settings.min_bin_freq': 50})