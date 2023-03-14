import pandas as pd

from pbtrack.utils.coordinates import kp_img_to_kp_bbox, bbox_ltwh2ltrb, bbox_ltwh2cmwh


class Detections(pd.DataFrame):
    # Required for DataFrame subclassing
    @property
    def _constructor(self):
        return Detections

    # Required for DataFrame subclassing
    @property
    def _constructor_sliced(self):
        return Detection

    # use this to view the base class, needed for debugging in some IDEs.
    @property
    def aaa_base_class_view(self):
        return pd.DataFrame(self)

    # Utils for converting between formats
    @property
    def bbox_ltrb(self):
        """Converts from (left, top, width, heights) to (left, top, right, bottom)"""
        return self.bbox_ltwh.apply(lambda ltwh: bbox_ltwh2ltrb(ltwh))  # type: ignore

    @property
    def bbox_cmwh(self):
        """Converts from (left, top, width, heights) to (horizontal center, vertical middle, width, height)"""
        return self.bbox_ltwh.apply(lambda ltwh: bbox_ltwh2cmwh(ltwh))  # type: ignore

    @property
    def keypoints_bbox_xyc(self):
        """Converts from keypoints in image coordinates to keypoints in bbox coordinates"""
        return self.apply(lambda r: kp_img_to_kp_bbox(r.keypoints_xyc, r.bbox_ltwh))


class Detection(pd.Series):
    @classmethod
    def create(
        cls,
        image_id,
        id,
        bbox_ltwh=pd.NA,
        keypoints_xyc=pd.NA,
        person_id=pd.NA,
        category_id=pd.NA,
        **kwargs
    ):
        return cls(
            dict(
                image_id=image_id,
                id=id,
                bbox_ltwh=bbox_ltwh,
                keypoints_xyc=keypoints_xyc,
                person_id=person_id,
                category_id=category_id,
                **kwargs
            ),
            name=id,
        )

    @property
    def _constructor_expanddim(self):
        return Detections

    # Required for DataFrame subclassing
    @property
    def _constructor(self):
        return Detection

    # Allows to convert automatically from Detection to Detections
    # and use their @property methods
    def __getattr__(self, attr):
        if hasattr(Detections, attr):
            return getattr(self.to_frame().T, attr).item()
        else:
            return super().__getattr__(attr)
        """ other version in case of bug with the implemented one
        try:
            return pd.Series.__getattr__(self, attr)
        except AttributeError as e:
            if hasattr(Detections, attr):
                return getattr(self.to_frame().T, attr)
            else:
                raise e
        """
