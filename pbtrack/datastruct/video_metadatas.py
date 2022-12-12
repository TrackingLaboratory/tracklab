import pandas as pd


class VideoMetadatas(pd.DataFrame):
    
    def __init__(self, data, *args, **kwargs) -> None:
        if isinstance(data, list):
            indices = [x.id for x in data]
        else:
            indices = None
        kwargs = {**kwargs, "index": indices}
        super().__init__(data, *args, **kwargs)
    
    # Required for DataFrame subclassing
    @property
    def _constructor(self):
        return VideoMetadatas

    # not needed - can be suppressed
    @property
    def _constructor_sliced(self):
        return VideoMetadata

    @property
    def aaa_base_class_view(self):
        # use this to view the base class, needed for debugging in some IDEs.
        return pd.DataFrame(self)

    # add the properties here


class VideoMetadata(pd.Series):
    @classmethod
    def create(
            cls,
            id,
            name,
            supercategory = None,
            keypoints = None,
            skeleton = None,
            **kwargs
        ):
        return cls(
            dict(
                id = id,
                name = name,
                supercategory = supercategory,
                keypoints = keypoints,
                skeleton = skeleton,
                **kwargs
            )
        )
    
    @property
    def _constructor_expanddim(self):
        return VideoMetadatas
    
    # not needed - can be suppressed
    @property
    def _constructor(self):
        return pd.Series # we lose the link with Categorie here
    
    # Allows to convert automatically from Categorie to VideoMetadatas
    # and use their @property methods
    def __getattr__(self, attr):
        if hasattr(VideoMetadatas, attr):
            return getattr(self.to_frame().T, attr)
        else:
            return super().__getattr__(attr)
        """ other version in case of bug with the implemented one
        try:
            return pd.Series.__getattr__(self, attr)
        except AttributeError as e:
            if hasattr(VideoMetadatas, attr):
                return getattr(self.to_frame().T, attr)
            else:
                raise e
        """
