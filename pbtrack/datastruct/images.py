import pandas as pd

class Images(pd.DataFrame):
    @property
    def _constructor(self):
        return Images

    @property # not needed
    def _constructor_sliced(self):
        return pd.Series # we lose the link with Image here
    
    @property
    def aaa_base_class_view(self):
        # use this to view the base class, needed for debugging in some IDEs.
        return pd.DataFrame(self)
    
    # add the properties here
    
class Image(pd.Series):
    def __init__(
            self,
            id,
            video_id,
            nframe,
            file_path,
            is_labeled=None,
            ignore_regions_x=None,
            ignore_regions_y=None
        ):
        super(Image, self).__init__(
            dict(
                id=id,
                video_id=video_id,
                nframe=nframe,
                file_path=file_path,
                is_labeled=is_labeled,
                ignore_regions_x=ignore_regions_x,
                ignore_regions_y=ignore_regions_y
            )  # type: ignore
        )
    
    @property
    def _constructor_expanddim(self):
        return Images
    
    # not needed - can be suppressed
    @property
    def _constructor(self):
        return pd.Series # we lose the link with Detection here
    
    # Allows to convert automatically from Detection to Detections
    # and use their @property methods
    def __getattr__(self, attr):
        if hasattr(Images, attr):
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