import numpy as np
import pandas as pd

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

    def ltwh(self, image_shape=None):
        if image_shape:
            width, height = image_shape[0] - 1, image_shape[1] - 1
            return self._obj.bbox_ltwh.map(
                lambda x: np.round(
                    np.array(
                        [
                            np.clip(x[0], 0, width),
                            np.clip(x[1], 0, height),
                            np.clip(x[2], 0, np.maximum(0, width - x[0])),
                            np.clip(x[3], 0, np.maximum(0, height - x[1])),
                        ]
                    ),
                ).astype(int)
            )
        else:
            return self._obj.bbox_ltwh.map(lambda x: x.astype(float))

    def xywh(self, image_shape=None):
        if image_shape:
            return self.ltwh(image_shape).map(
                lambda x: np.round(
                    np.array([x[0] + x[2] / 2, x[1] + x[3] / 2, x[2], x[3]])
                ).astype(int)
            )
        else:
            return self._obj.bbox_ltwh.map(
                lambda x: np.array([x[0] + x[2] / 2, x[1] + x[3] / 2, x[2], x[3]], dtype=float))

    def ltrb(self, image_shape=None):
        if image_shape:
            return self.ltwh(image_shape).map(
                lambda x: np.array([x[0], x[1], x[0] + x[2], x[1] + x[3]]).astype(int)
            )
        else:
            return self._obj.bbox_ltwh.map(
                lambda x: np.array([x[0], x[1], x[0] + x[2], x[1] + x[3]])
            )

    def conf(self):
        return self._obj.bbox_conf.map(lambda x: float(x))

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

    def ltwh(self, image_shape=None):
        if image_shape:
            width, height = image_shape[0] - 1, image_shape[1] - 1
            return self._obj.bbox_ltwh.map(
                lambda x: np.round(
                    np.array(
                        [
                            np.clip(x[0], 0, width),
                            np.clip(x[1], 0, height),
                            np.clip(x[2], 0, np.maximum(0, width - x[0])),
                            np.clip(x[3], 0, np.maximum(0, height - x[1])),
                        ]
                    ),
                ).astype(int)
            )
        else:
            return self._obj.bbox_ltwh.map(lambda x: x.astype(float))

    def xywh(self, image_shape=None):
        if image_shape:
            return self.ltwh(image_shape).map(
                lambda x: np.round(
                    np.array([x[0] + x[2] / 2, x[1] + x[3] / 2, x[2], x[3]])
                ).astype(int)
            )
        else:
            return self._obj.bbox_ltwh.map(
                lambda x: np.array([x[0] + x[2] / 2, x[1] + x[3] / 2, x[2], x[3]], dtype=float))

    def ltrb(self, image_shape=None):
        if image_shape:
            return self.ltwh(image_shape).map(
                lambda x: np.array([x[0], x[1], x[0] + x[2], x[1] + x[3]]).astype(int)
            )
        else:
            return self._obj.bbox_ltwh.map(
                lambda x: np.array([x[0], x[1], x[0] + x[2], x[1] + x[3]])
            )

    def conf(self):
        return self._obj.bbox_conf.map(lambda x: float(x))


@pd.api.extensions.register_dataframe_accessor("keypoints")
@pd.api.extensions.register_series_accessor("keypoints")
class KeypointsAccessor:
    def __init__(self, pandas_obj):
        if type(pandas_obj) is pd.Series:
            pandas_obj = pandas_obj.to_frame().T
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if "keypoints_xyc" not in obj.columns:
            raise AttributeError("Must have 'keypoints_xyc'.")
        if "keypoints_conf" not in obj.columns:
            raise AttributeError("Must have 'keypoints_conf'.")

    def xyc(self, image_shape=None):
        if image_shape:
            width, height = image_shape[0] - 1, image_shape[1] - 1
            return self._obj.keypoints_xyc.map(
                lambda x: np.concatenate(
                    (
                        np.round(
                            np.array(np.clip(x[:, :2], 0, [width, height]))
                        ).astype(int),
                        np.array(x[:, 2]).reshape(-1, 1),
                    ),
                    axis=1,
                )
            )
        else:
            return self._obj.keypoints_xyc.map(lambda x: np.array(x).astype(float))

    def xy(self, image_shape=None):
        if image_shape:
            keypoints_xyc = self.xyc(image_shape)
            return keypoints_xyc.map(lambda x: np.array(x[:, :2]).astype(int))
        else:
            return self._obj.keypoints_xyc.map(
                lambda x: np.array(x[:, :2]).astype(float)
            )

    def c(self):
        return self._obj.keypoints_xyc.map(lambda x: np.array(x[:, 2]).astype(float))

    def conf(self):
        return self._obj.keypoints_conf.map(lambda x: float(x))
