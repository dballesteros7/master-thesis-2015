from enum import Enum

from crawl.boundingbox import BoundingBox


class City(Enum):
    LONDON = ('london', BoundingBox(51.672343, 0.148271, 51.384940, -0.351468))
    ZURICH = ('zurich', BoundingBox(47.434680, 8.625370, 47.320230, 8.448060))

    def __init__(self, city_name: str, bounding_box: BoundingBox):
        self.city_name = city_name
        self.bounding_box = bounding_box
