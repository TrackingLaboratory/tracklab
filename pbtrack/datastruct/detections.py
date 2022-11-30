import numpy as np
import pandas as pd
from pbtrack.utils.coordinates import kp_img_to_kp_bbox


class Detections(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super(Detections, self).__init__(*args, **kwargs)

    # Required for DataFrame subclassing
    @property
    def _constructor(self):
        return Detections

    # not needed - can be suppressed
    @property
    def _constructor_sliced(self):
        return pd.Series # we lose the link with Detection here

    @property
    def aaa_base_class_view(self):
        # use this to view the base class, needed for debugging in some IDEs.
        return pd.DataFrame(self)

    # Utils for converting between formats
    @property
    def bbox_ltrb(self):
        """Converts from (left, top, width, heights) to (left, top, right, bottom)"""
        return self.bbox.apply(
            lambda ltwh: np.concatenate((ltwh[:2], ltwh[:2] + ltwh[2:]))  # type: ignore
        )

    @property
    def bbox_cmwh(self):
        """Converts from (left, top, width, heights) to (horizontal center, vertical middle, width, height)"""
        return self.bbox.apply(
            lambda ltwh: np.concatenate((ltwh[:2] + ltwh[2:] / 2, ltwh[2:]))  # type: ignore
        )

    @property
    def keypoints_bbox_xyc(self):
        """Converts from keypoints in image coordinates to keypoints in bbox coordinates"""
        return self.bbox.apply(
            lambda r: kp_img_to_kp_bbox(r.keypoints_xyc, r.bbox_ltwh), axis=1
        )


class Detection(pd.Series):
    def __init__(
            self,
            image_id,
            video_id,
            bbox = None, # COCO bbox format [top_left_x, top_left_y, width, height]
            keypoints_xyc = None,
            keypoints_bbox_xyc = None,
            visibility = None, # visibility score for each keypoint ?
            id = None, # can't it be the index in the dataframe ?
            person_id = None,
            category_id = None,
        ):
        super(Detection, self).__init__(
            dict(
                image_id=image_id,
                video_id=video_id,
                bbox=bbox,
                keypoints_xyc=keypoints_xyc,
                keypoints_bbox_xyc=keypoints_bbox_xyc,
                visibility=visibility,
                id=id,
                person_id=person_id,
                category_id=category_id
            )  # type: ignore
        )
    
    @property
    def _constructor_expanddim(self):
        return Detections
    
    # not needed - can be suppressed
    @property
    def _constructor(self):
        return pd.Series # we lose the link with Detection here
    
    # Allows to convert automatically from Detection to Detections
    # and use their @property methods
    def __getattr__(self, attr):
        if hasattr(Detections, attr):
            return getattr(self.to_frame().T, attr)
        else:
            return super().__getattr__(attr)
        """ other version in case of bug with the implemented one
        try:
            return pd.Series.__getattr__(self, attr)
        except AttributeError as e:
            if hasattr(Images, attr):
                return getattr(self.to_frame().T, attr)
            else:
                raise e
        """
