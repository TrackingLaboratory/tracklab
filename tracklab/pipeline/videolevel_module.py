from abc import abstractmethod
from typing import Any

import pandas as pd
from torch.utils.data.dataloader import default_collate, DataLoader
from tracklab.datastruct import EngineDatapipe
from tracklab.pipeline import Module
from tracklab.utils.cv2 import cv2_load_image


class VideoLevelModule(Module):
    """Abstract class to implement a module that operates on whole videos, image per image.

    This can for example be an offline tracker, or a video visualizer, by implementing
    process_video directly, or an online tracker, by implementing preprocess and process

    The functions to implement are
     - __init__, which can take any configuration needed
     - process

     You should also provide the following class properties :
      - input_columns : what info you need for the detections
      - output_columns : what info you will provide when called

     A description of the expected behavior is provided below.
    """

    input_columns = None
    output_columns = None

    @abstractmethod
    def __init__(self):
        """Init function

        The arguments to this function are completely free
        and will be provided by a configuration file.

        You should call the __init__ function from the super() class.
        """
        pass

    @abstractmethod
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        """The main processing function. Runs on GPU.

        Args:
            detections: The previous detections.
            metadatas: The previous image metadatas

        Returns:
            output : Either a DataFrame containing the new/updated detections
                    or a tuple containing detections and metadatas (in that order)
                    The DataFrames can be either a list of Series, a list of DataFrames
                    or a single DataFrame. The returned objects will be aggregated
                    automatically according to the `name` of the Series/`index` of
                    the DataFrame. **It is thus mandatory here to name correctly
                    your series or index your dataframes.**
                    The output will override the previous detections
                    with the same name/index.
        """
        pass
