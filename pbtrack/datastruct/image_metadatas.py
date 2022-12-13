import pandas as pd
import cv2


class ImageMetadatas(pd.DataFrame):
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

    @property
    def image(self):
        def open_image(file_path):
            image = cv2.imread(str(file_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image

        return self.file_path.apply(open_image)

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
            name=id,
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
    def __getattr__(self, attr):
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
