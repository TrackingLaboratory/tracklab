import pandas as pd


class Categories(pd.DataFrame):
    # Required for DataFrame subclassing
    @property
    def _constructor(self):
        return Categories

    # not needed - can be suppressed
    @property
    def _constructor_sliced(self):
        return pd.Series # we lose the link with Categorie here

    @property
    def aaa_base_class_view(self):
        # use this to view the base class, needed for debugging in some IDEs.
        return pd.DataFrame(self)

    # add the properties here


class Categorie(pd.Series):
    def __init__(
            self,
            id,
            name,
            supercategory = None,
            keypoints = None,
            skeleton = None,
        ):
        super(Categorie, self).__init__(
            dict(
                id = id,
                name = name,
                supercategory = supercategory,
                keypoints = keypoints,
                skeleton = skeleton,
            )  # type: ignore
        )
    
    @property
    def _constructor_expanddim(self):
        return Categories
    
    # not needed - can be suppressed
    @property
    def _constructor(self):
        return pd.Series # we lose the link with Categorie here
    
    # Allows to convert automatically from Categorie to Categories
    # and use their @property methods
    def __getattr__(self, attr):
        if hasattr(Categories, attr):
            return getattr(self.to_frame().T, attr)
        else:
            return super().__getattr__(attr)
        """ other version in case of bug with the implemented one
        try:
            return pd.Series.__getattr__(self, attr)
        except AttributeError as e:
            if hasattr(Categories, attr):
                return getattr(self.to_frame().T, attr)
            else:
                raise e
        """
