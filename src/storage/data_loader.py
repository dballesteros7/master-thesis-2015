from pymongo import MongoClient


class DataLoader:
    def __init__(self):
        client = MongoClient()
        self.database = client.flickrdata

    def load_entries(self, city_name):
        return self.database.photos.find({'city_name': city_name})
