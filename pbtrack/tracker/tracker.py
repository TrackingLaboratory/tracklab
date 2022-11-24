from pbtrack.tracker.categories import Categories
from pbtrack.tracker.detections import Detections
from pbtrack.tracker.images import Images


class Tracker:
    # TODO add functions train/val/test or split(train/val/test)
    def __init__(self, images: Images, categories: Categories, detections: Detections):
        self.images = images
        self.categories = categories
        self.dets_gt = detections
        self.dets_pd = Detections()

    def add_predictions(self, dets_pd):
        self.dets_pd = Detections(dets_pd)
