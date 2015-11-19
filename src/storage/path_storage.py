from pymongo import MongoClient


class PathStorage:
    def __init__(self):
        client = MongoClient()
        self.collection = client.flickrdata.paths

    def store_paths(self, paths):
        self.collection.insert_many(paths)

    def get_paths(self, city_name, bandwidth, min_unique_users, min_cluster_photos):
        return self.collection.find({
            'city_name': city_name,
            'bandwidth': bandwidth,
            'min_unique_users': min_unique_users,
            'min_cluster_photos': min_cluster_photos
        })
