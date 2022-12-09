import pandas as pd
import cv2

class Metadatas(pd.DataFrame):

    def __init__(self, data, *args, **kwargs) -> None:
        if isinstance(data, list):
            indices = [x.id for x in data]
        else:
            indices = None
        kwargs = {**kwargs, "index": indices}
        super().__init__(data, *args, **kwargs)

    @property
    def _constructor(self):
        return Metadatas

    @property
    def image(self):
        def open_image(file_path):
            image = cv2.imread(str(file_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image

        return self.file_path.apply(open_image)

    @property # not needed
    def _constructor_sliced(self):
        return Metadata # we lose the link with Metadata here
    
    @property
    def aaa_base_class_view(self):
        # use this to view the base class, needed for debugging in some IDEs.
        return pd.DataFrame(self)
    
    # add the properties here
    
class Metadata(pd.Series):

    @classmethod
    def create(
            cls,
            id,
            video_id,
            frame,
            nframe,
            file_path,
            is_labeled=None,
            ignore_regions_x=None,
            ignore_regions_y=None
        ):
        return cls(
            dict(
                id = id,
                video_id = video_id,
                frame = frame,
                nframe = nframe,
                file_path = file_path,
                is_labeled = is_labeled,
                ignore_regions_x = ignore_regions_x,
                ignore_regions_y = ignore_regions_y
            ),  # type: ignore
        )
    
    @property
    def _constructor_expanddim(self):
        return Metadatas
    
    # not needed - can be suppressed
    @property
    def _constructor(self):
        return pd.Series # we lose the link with Metadata here
    
    # Allows to convert automatically from Metadata to Metadatas
    # and use their @property methods
    def __getattr__(self, attr):
        if hasattr(Metadatas, attr):
            return getattr(self.to_frame().T, attr).item()
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