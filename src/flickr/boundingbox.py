class BoundingBox:
    def __init__(self, top: float, right: float, bottom: float, left: float):
        self.top = top
        self.right = right
        self.bottom = bottom
        self.left = left

    def to_flickr_bounding_box(self):
        return '{self.left}, {self.bottom}, {self.right}, {self.top}'.format(self=self)
