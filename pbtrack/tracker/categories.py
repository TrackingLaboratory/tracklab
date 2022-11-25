import pandas as pd


class Categories(pd.DataFrame):
    rcols = {
        'id': True,
        'name': True,
        'supercategory': False,
        'keypoints': False,
        'skeleton': False,
    }

    def __init__(self, *args, **kwargs):
        super(Categories, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return Categories

    @property
    def _constructor_sliced(self):
        return CategoriesSeries

    @property
    def aaa_base_class_view(self):
        # use this to view the base class, needed for debugging in some IDEs.
        return pd.DataFrame(self)


class CategoriesSeries(pd.Series):
    @property
    def _constructor(self):
        return CategoriesSeries

    @property
    def _constructor_expanddim(self):
        return Categories
