import pandas as pd

class Metadatas(pd.DataFrame):

    def __init__(self, data) -> None:
        if isinstance(data, list):
            indices = [x.id for x in data]
        else:
            indices = None
        super().__init__(data, index=indices)

    @property
    def _constructor(self):
        return Metadatas

    @property # not needed
    def _constructor_sliced(self):
        return pd.Series # we lose the link with Metadata here
    
    @property
    def aaa_base_class_view(self):
        # use this to view the base class, needed for debugging in some IDEs.
        return pd.DataFrame(self)
    
    # add the properties here
    
class Metadata(pd.Series):
    def __init__(
            self,
            id,
            video_id,
            frame,
            nframe,
            file_path,
            is_labeled=None,
            ignore_regions_x=None,
            ignore_regions_y=None
        ):
        super(Metadata, self).__init__(
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