from enum import Enum

from crawl.boundingbox import BoundingBox


class City(Enum):
    LONDON = ('london', BoundingBox(51.672343, 0.148271, 51.384940, -0.351468))
    ZURICH = ('zurich', BoundingBox(47.434680, 8.625370, 47.320230, 8.448060))
    NEW_YORK = ('new-york', BoundingBox(40.915256, -73.700272, 40.491370, -74.259090))
    SHANGHAI = ('shanghai', BoundingBox(31.868217, 122.247066, 30.680270, 120.858217))

    def __init__(self, city_name: str, bounding_box: BoundingBox):
        self.city_name = city_name
        self.bounding_box = bounding_box
