# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.
    keypoints: array_like Nx3
        Keypoint format `(x, y, conf)`

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, id, bbox_ltwh, confidence, feature, keypoints):
        self.ltwh = bbox_ltwh
        self.confidence = float(confidence)
        self.feature = feature
        self.id = id
        self.keypoints = keypoints
        self.matched_with = None
        self.costs = {}

    def to_ltwh(self):
        return self.ltwh.copy()

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.to_ltwh()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.to_ltwh()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
