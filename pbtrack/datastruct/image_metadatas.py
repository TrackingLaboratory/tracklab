import pandas as pd

from pbtrack.utils.images import cv2_load_image


class ImageMetadatas(pd.DataFrame):
    def __init__(self, data, *args, **kwargs) -> None:
        """FIXME ?
        if isinstance(data, list):
            if isinstance(data[0], ImageMetadata):
                indices = [x.id for x in data]
            elif isinstance(data[0], dict):
                indices = [x["id"] for x in data]
        else:
            indices = None
        kwargs = {**kwargs, "index": indices}
        """
        super().__init__(data, *args, **kwargs)

    # Required for DataFrame subclassing
    @property
    def _constructor(self):
        return ImageMetadatas

    # Required for DataFrame subclassing
    @property
    def _constructor_sliced(self):
        return ImageMetadata

    # use this to view the base class, needed for debugging in some IDEs.
    @property
    def aaa_base_class_view(self):
        # use this to view the base class, needed for debugging in some IDEs.
        return pd.DataFrame(self)

    def load_image(self):
        return self.file_path.apply(cv2_load_image)

    # add the properties here


class ImageMetadata(pd.Series):
    @classmethod
    def create(cls, id, video_id, frame, nframe, file_path, **kwargs):
        return cls(
            dict(
                id=id,
                video_id=video_id,
                frame=frame,
                nframe=nframe,
                file_path=file_path,
                **kwargs
            ),
        )

    # Required for DataFrame subclassing
    @property
    def _constructor_expanddim(self):
        return ImageMetadatas

    # Required for DataFrame subclassing
    @property
    def _constructor(self):
        return ImageMetadata

    # Allows to convert automatically from ImageMetadata to ImageMetadatas
    # and use their @property methods
    def __getattr__(
        self, attr
    ):  # FIXME both doesn't work with helper functions (properties are ok)
        if hasattr(ImageMetadatas, attr):
            return getattr(self.to_frame().T, attr).item()
        else:
            return super().__getattr__(attr)
        """ other version in case of bug with the implemented one
        try:
            return pd.Series.__getattr__(self, attr)
        except AttributeError as e:
            if hasattr(ImageMetadatas, attr):
                return getattr(self.to_frame().T, attr)
            else:
                raise e
        """
