import pandas as pd
from pbtrack.utils.coordinates import *


@pd.api.extensions.register_dataframe_accessor("bbox")
class BBoxDataFrameAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if "bbox_ltwh" not in obj.columns:
            raise AttributeError("Must have 'bbox_ltwh'.")
        if "bbox_conf" not in obj.columns:
            raise AttributeError("Must have 'bbox_conf'.")

    def ltwh(self, image_shape=None, rounded=False):
        return self._obj.bbox_ltwh.map(
            lambda x: sanitize_bbox_ltwh(x, image_shape, rounded)
        )

    def ltrb(self, image_shape=None, rounded=False):
        return self._obj.bbox_ltwh.map(lambda x: ltwh_to_ltrb(x, image_shape, rounded))

    def xywh(self, image_shape=None, rounded=False):
        return self._obj.bbox_ltwh.map(lambda x: ltwh_to_xywh(x, image_shape, rounded))

    def conf(self):
        return self._obj.bbox_conf


@pd.api.extensions.register_series_accessor("bbox")
class BBoxSeriesAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if "bbox_ltwh" not in obj.index:
            raise AttributeError("Must have 'bbox_ltwh'.")
        if "bbox_conf" not in obj.index:
            raise AttributeError("Must have 'bbox_conf'.")

    def ltwh(self, image_shape=None, rounded=False):
        return sanitize_bbox_ltwh(self._obj.bbox_ltwh, image_shape, rounded)

    def ltrb(self, image_shape=None, rounded=False):
        return ltwh_to_ltrb(self._obj.bbox_ltwh, image_shape, rounded)

    def xywh(self, image_shape=None, rounded=False):
        return ltwh_to_xywh(self._obj.bbox_ltwh, image_shape, rounded)

    def conf(self):
        return self._obj.bbox_conf


@pd.api.extensions.register_dataframe_accessor("keypoints")
class KeypointsDataFrameAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if "keypoints_xyc" not in obj.columns:
            raise AttributeError("Must have 'keypoints_xyc'.")
        if "keypoints_conf" not in obj.columns:
            raise AttributeError("Must have 'keypoints_conf'.")

    def xyc(self, image_shape=None, rounded=False):
        return self._obj.keypoints_xyc.map(
            lambda x: sanitize_keypoints(x, image_shape, rounded)
        )

    def xy(self, image_shape=None, rounded=False):
        return self._obj.keypoints_xyc.map(
            lambda x: sanitize_keypoints(x, image_shape, rounded)[:, :2]
        )

    def c(self):
        return self._obj.keypoints_xyc.map(lambda x: x[:, 2])

    def conf(self):
        return self._obj.keypoints_conf

    def in_bbox_coord(self, bbox_ltwh):
        return self._obj.keypoints_xyc.map(
            lambda x: keypoints_in_bbox_coord(x, bbox_ltwh)
        )


@pd.api.extensions.register_series_accessor("keypoints")
class KeypointsSeriesAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if "keypoints_xyc" not in obj.index:
            raise AttributeError("Must have 'keypoints_xyc'.")
        if "keypoints_conf" not in obj.index:
            raise AttributeError("Must have 'keypoints_conf'.")

    def xyc(self, image_shape=None, rounded=False):
        return sanitize_keypoints(self._obj.keypoints_xyc[:, :2], image_shape, rounded)

    def xy(self, image_shape=None, rounded=False):
        return sanitize_keypoints(self._obj.keypoints_xyc[:, :2], image_shape, rounded)[
            :, :2
        ]

    def c(self):
        return self._obj.keypoints_xyc[:, 2]

    def conf(self):
        return self._obj.keypoints_conf

    def in_bbox_coord(self, bbox_ltwh):
        return keypoints_in_bbox_coord(self._obj.keypoints_xyc, bbox_ltwh)
