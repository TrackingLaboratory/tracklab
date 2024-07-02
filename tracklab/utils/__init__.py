import pandas as pd
from tracklab.utils.coordinates import *


@pd.api.extensions.register_dataframe_accessor("bbox")
class BBoxDataFrameAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if "bbox_ltwh" not in obj.columns:
            raise AttributeError("Must have 'bbox_ltwh'.")

    def ltwh(self, image_shape=None, rounded=False):
        return self._obj.bbox_ltwh.map(
            lambda x: sanitize_bbox_ltwh(x, image_shape, rounded)
        )

    def ltrb(self, image_shape=None, rounded=False):
        return self._obj.bbox_ltwh.map(lambda x: ltwh_to_ltrb(x, image_shape, rounded))

    def xywh(self, image_shape=None, rounded=False):
        return self._obj.bbox_ltwh.map(lambda x: ltwh_to_xywh(x, image_shape, rounded))

    def conf(self):
        if "bbox_conf" in self._obj:
            return self._obj.bbox_conf
        else:
            return self._obj.bbox_ltwh.map(lambda x: 1.)

@pd.api.extensions.register_series_accessor("bbox")
class BBoxSeriesAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if "bbox_ltwh" not in obj.index:
            raise AttributeError("Must have 'bbox_ltwh'.")

    def ltwh(self, image_shape=None, rounded=False):
        return sanitize_bbox_ltwh(self._obj.bbox_ltwh, image_shape, rounded)

    def ltrb(self, image_shape=None, rounded=False):
        return ltwh_to_ltrb(self._obj.bbox_ltwh, image_shape, rounded)

    def xywh(self, image_shape=None, rounded=False):
        return ltwh_to_xywh(self._obj.bbox_ltwh, image_shape, rounded)

    def conf(self):
        if "bbox_conf" in self._obj:
            return self._obj.bbox_conf
        else:
            return 1.

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

    def keypoints_bbox_xyc(self):
        """Converts from keypoints in image coordinates to keypoints in bbox coordinates"""
        return self._obj.apply(
            lambda r: keypoints_in_bbox_coord(r.keypoints_xyc, r.bbox_ltwh), axis=1)

    def add_negative_samples(self):
        self._obj["id"] = self._obj.index
        self._obj.reset_index(drop=True, inplace=True)
        all_kps_in_img = np.array(list(self._obj.keypoints_xyc))
        id_to_index = {k: v for v, k in enumerate(list(self._obj.id))}
        self._obj["negative_kps"] = self._obj \
            .apply(
            lambda bb: keypoints_in_bbox_coord(
                np.delete(all_kps_in_img, id_to_index[bb.id], axis=0), bb.bbox_ltwh),
            axis=1) \
            .apply(
            lambda kp_xyc_bbox: kp_xyc_bbox[
                kp_xyc_bbox[:, :, 2].sum(axis=1) > 0])  # remove non visibile skeletons
        self._obj.set_index("id", inplace=True)
        return self._obj

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

    def keypoints_bbox_xyc(self):
        """Converts from keypoints in image coordinates to keypoints in bbox coordinates"""
        return keypoints_in_bbox_coord(self._obj.keypoints_xyc, self._obj.bbox_ltwh)