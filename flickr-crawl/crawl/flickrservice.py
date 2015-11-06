import calendar
import datetime
import json
import logging
import os
import time

import flickrapi

from crawl import secrets
from crawl.boundingbox import BoundingBox
from crawl.city import City

_REQUESTS_PER_HOUR_LIMIT = 3000


class RateLimiter:
    def __init__(self):
        self.requests_in_last_hour = 0
        self.current_hour = datetime.datetime.now().hour

    def check_and_sleep(self):
        current_hour = datetime.datetime.now().hour
        if current_hour != self.current_hour:
            self.current_hour = current_hour
            self.requests_in_last_hour = 0
            logging.info('It is a new hour, resetting rate limiter\'s counter.')
        elif self.requests_in_last_hour > _REQUESTS_PER_HOUR_LIMIT:
            nap_time = time.sleep((60 - datetime.datetime.now().minute) * 60)
            logging.info('Sleeping for {} seconds before continuing, limit has been hit.'.format(nap_time))
            time.sleep(nap_time)
        self.requests_in_last_hour += 1


class FlickrService:
    _CITY_STREET_ACCURACY = 16
    _PHOTOS_CONTENT_TYPE = 1

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FlickrService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        try:
            self._initialized
        except AttributeError:
            self._initialized = True
            self._rate_limiter = RateLimiter()
            self._flickr = flickrapi.FlickrAPI(secrets.FLICKR_API_KEY, secrets.FLICKR_API_SECRET, format='parsed-json')

    def get_geotagged_photos(self, bbox: BoundingBox, start_date: datetime.date, end_date: datetime.date,
                             page: int) -> dict:
        self._rate_limiter.check_and_sleep()
        result = self._flickr.photos.search(
            min_taken_date=calendar.timegm(start_date.timetuple()),
            max_taken_date=calendar.timegm(end_date.timetuple()),
            bbox=bbox.to_flickr_bounding_box(),
            accuracy=self._CITY_STREET_ACCURACY, content_type=self._PHOTOS_CONTENT_TYPE, has_geo="1",
            page=page, extras='description,license,date_upload,date_taken,geo,tags,machine_tags,views')
        logging.info('Got a new result for page {} out of {} with {} photos.'.format(
            page, result['photos']['pages'], result['photos']['perpage']))
        return result


def get_photos_for_city(city: City):
    service = FlickrService()
    years = range(2015, 2016)
    output_file_template = '/local/workspace/master-thesis-2015/data/raw/{city_name}-{year}-{page}.json'
    logging.info('Processing city {}.'.format(city.city_name))
    for year in years:
        logging.info('Getting photos for year {}.'.format(year))
        page = 1
        result = service.get_geotagged_photos(city.bounding_box, datetime.date(year, 1, 1), datetime.date(year, 12, 31),
                                              page)
        total_pages = result['photos']['pages']
        for page in range(1, total_pages + 1):
            result = service.get_geotagged_photos(city.bounding_box, datetime.date(year, 1, 1),
                                                  datetime.date(year, 12, 31),
                                                  page)
            output_filename = output_file_template.format(city_name=city.city_name, year=year, page=page)
            with open(output_filename, 'w') as output_file:
                json.dump(result['photos']['photo'], output_file)
                logging.info('Stored file {}.'.format(output_filename))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(funcName)s:%(module)s:%(message)s')
    get_photos_for_city(City.ZURICH)
