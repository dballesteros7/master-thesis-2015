import calendar
import json
import datetime
import logging
import time

import flickrapi

from crawl import secrets
from crawl.boundingbox import BoundingBox
from crawl.city import City


class FlickrService:
    _CITY_STREET_ACCURACY = 16
    _PHOTOS_CONTENT_TYPE = 1
    _REQUESTS_PER_HOUR_LIMIT = 3600
    _HOUR_IN_SECONDS = 3900

    def __init__(self):
        self.requests_in_last_hour = None
        self.last_hour_start = None
        self.flickr = flickrapi.FlickrAPI(secrets.FLICKR_API_KEY, secrets.FLICKR_API_SECRET, format='parsed-json')

    def get_geotagged_photos(self, bbox: BoundingBox, start_date: datetime.date, end_date: datetime.date, page: int):
        if self.last_hour_start is None:
            self.last_hour_start = time.time()
            self.requests_in_last_hour = 0
            logging.info('First API call.')
        elif time.time() - self.last_hour_start > self._HOUR_IN_SECONDS:
            self.last_hour_start = time.time()
            self.requests_in_last_hour = 0
            logging.info('It has been more than an hour since the first API call in this cycle.')
        elif self.requests_in_last_hour >= self._REQUESTS_PER_HOUR_LIMIT:
            sleep_time = self._HOUR_IN_SECONDS - (time.time() - self.last_hour_start)
            logging.info('Too many calls in an hour, sleeping for {} seconds.'.format(sleep_time))
            time.sleep(sleep_time)
            logging.info('Sleeping finished, now continuing.')
            self.last_hour_start = time.time()
            self.requests_in_last_hour = 0

        result = self.flickr.photos.search(
            min_taken_date=calendar.timegm(start_date.timetuple()),
            max_taken_date=calendar.timegm(end_date.timetuple()),
            bbox=bbox.to_flickr_bounding_box(),
            accuracy=self._CITY_STREET_ACCURACY, content_type=self._PHOTOS_CONTENT_TYPE, has_geo="1",
            page=page)
        self.requests_in_last_hour += 1
        logging.info('Made another successful API call.')
        return result


def main(city: City):
    service = FlickrService()
    years = range(2011, 2012)
    output_file_template = '/local/workspace/master-thesis-2015/data/{city_name}-{year}-{page}.json'
    logging.info('Processing city {}.'.format(city.city_name))
    for year in years:
        logging.info('Getting photos for year {}.'.format(year))
        page = 1
        result = service.get_geotagged_photos(city.bounding_box, datetime.date(year, 1, 1), datetime.date(year, 12, 31),
                                              page)
        total_pages = result['photos']['pages']
        for page in range(1, total_pages + 1):
            logging.info('Getting photos for page {} in year {}.'.format(page, year))
            result = service.get_geotagged_photos(city.bounding_box, datetime.date(year, 1, 1),
                                                  datetime.date(year, 12, 31),
                                                  page)
            with open(output_file_template.format(city_name=city.city_name, year=year, page=page), 'w') as output_file:
                json.dump(result['photos']['photo'], output_file)
                logging.info('Stored file for page {} in year {}.'.format(page, year))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(City.LONDON)
