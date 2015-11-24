from pymongo import MongoClient


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
        return self.collection.find({'city_name': city_name}, sort=[('datetaken', 1)])

