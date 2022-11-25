import pandas as pd


class Images(pd.DataFrame):
    rcols = {
        'id': True,
        'video_id': True,
        'nframes': True,
        'is_labeled': True,
        'file_name': True,
        'ignore_regions_y': True,
        'ignore_regions_x': True,
    }

    def __init__(self, *args, **kwargs):
        super(Images, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return Images

    @property
    def _constructor_sliced(self):
        return ImagesSeries

    @property
    def aaa_base_class_view(self):
        # use this to view the base class, needed for debugging in some IDEs.
        return pd.DataFrame(self)


class ImagesSeries(pd.Series):
    @property
    def _constructor(self):
        return ImagesSeries

    @property
    def _constructor_expanddim(self):
        return Images
