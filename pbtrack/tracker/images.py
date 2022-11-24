import pandas as pd


class Images(pd.DataFrame):
    required_columns = {'is_labeled', 'nframes', 'image_id', 'id', 'video_id', 'file_name',
       'has_labeled_person', 'ignore_regions_y', 'ignore_regions_x', 'split'}

    def __init__(self, *args, **kwargs):
        super(Images, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return Images

    @property
    def _constructor_sliced(self):
        return ImagesSeries

    @property
    def base_class_view(self):
        # use this to view the base class, needed for debugging in some IDEs.
        return pd.DataFrame(self)


class ImagesSeries(pd.Series):
    @property
    def _constructor(self):
        return ImagesSeries

    @property
    def _constructor_expanddim(self):
        return Images
