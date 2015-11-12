import json
import logging
import os

from pymongo import MongoClient


class DataInserter:
    def __init__(self):
        client = MongoClient()
        self.database = client.flickrdata

    def read_jsons(self, path='/local/workspace/master-thesis-2015/data/nodups'):
        all_files = os.listdir(path)
        logging.info('Processing {} files.'.format(len(all_files)))
        for file_name in all_files:
            city_name = file_name.split('-')[0]
            with open(os.path.join(path, file_name), 'r') as input:
                logging.info('Processing {}.'.format(file_name))
                entries = json.load(input)
                for entry in entries:
                    entry['city_name'] = city_name
                logging.info('Inserting {} entries for {}.'.format(len(entries), city_name))
                if len(entries) > 0:
                    self.database.photos.insert_many(entries)
                logging.info('Finished inserting entries.')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(funcName)s:%(module)s:%(message)s')
    inserter = DataInserter()
    inserter.read_jsons()
