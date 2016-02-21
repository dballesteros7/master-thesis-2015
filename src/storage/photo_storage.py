import os

import datetime
from pymongo import MongoClient

import constants


class PhotoStorage:
    def __init__(self):
        client = MongoClient()
        self.collection = client.flickrdata.photos

    def get_top_users(self, city_name, limit):
        pipeline = [
            {
                '$match': {
                    'city_name': city_name
                }
            }, {
                '$group': {
                    '_id': '$owner',
                    'total': {
                        '$sum': 1
                    }
                }
            }, {
                '$sort': {
                    'total': -1
                }
            }, {
                '$limit': limit
            }
        ]
        return list(self.collection.aggregate(pipeline))

    def get_photos_for_user(self, user):
        return list(self.collection.find({'owner': user}))

    def get_photos_for_city(self, city_name):
        cache_path = constants.LOCAL_PHOTO_CACHE.format(city=city_name)
        if os.path.isfile(cache_path):
            with open(cache_path, 'r') as cache_file:
                keys = cache_file.readline().strip().split(',')
                result = []
                for line in cache_file:
                    tokens = line.strip().split(',')
                    item = {}
                    for key, token in zip(keys, tokens):
                        item[key] = token
                    result.append(item)
                sorted_result = sorted(
                    result, key=lambda x: datetime.datetime.strptime(
                        x['datetaken'], '%Y-%m-%d %H:%M:%S'))
                return sorted_result
        else:
            result = self.collection.find(
                    {'city_name': city_name}, sort=[('datetaken', 1)])
            ordered_keys = ['latitude', 'longitude', 'owner', 'id', 'datetaken', '_id']
            with open(cache_path, 'w') as cache_file:
                cache_file.write(','.join(ordered_keys))
                cache_file.write('\n')
                for entry in result:
                    if entry['accuracy'] == 0:
                        continue # Ignore bad geo-located entries.
                    items = []
                    for key in ordered_keys:
                        items.append(str(entry[key]))
                    cache_file.write(','.join(items))
                    cache_file.write('\n')
                return self.get_photos_for_city(city_name)


def main():
    storage = PhotoStorage()
    storage.get_photos_for_city('zurich')

if __name__ == '__main__':
    main()
