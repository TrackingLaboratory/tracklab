from abc import abstractmethod
from typing import Any

import pandas as pd

from pbtrack.datastruct import EngineDatapipe
from pbtrack.pipeline import Module

from torch.utils.data.dataloader import default_collate, DataLoader

from pbtrack.utils.cv2 import cv2_load_image


class VideoLevelModule(Module):
    """Abstract class to implement a module that operates on whole videos, image per image.

    This can for example be an offline tracker, or a video visualizer, by implementing
    process_video directly, or an online tracker, by implementing preprocess and process

    The functions to implement are
     - __init__, which can take any configuration needed
     - process_video
     - preprocess
     - process
     - datapipe (optional) : returns an object which will be used to create the pipeline.
                            (Only modify this if you know what you're doing)
     - dataloader (optional) : returns a dataloader for the datapipe

     You should also provide the following class properties :
      - input_columns : what info you need for the detections
      - output_columns : what info you will provide when called
      - collate_fn (optional) : the function that will be used for collating the inputs
                                in a batch. (Default : pytorch collate function)

     A description of the expected behavior is provided below.
    """

    collate_fn = default_collate
    input_columns = None
    output_columns = None
    level = "image"

    @abstractmethod
    def __init__(self, batch_size: int):
        """Init function

        The arguments to this function are completely free
        and will be provided by a configuration file.

        You should call the __init__ function from the super() class.
        """
        self.batch_size = batch_size
        self._datapipe = None

    def process_video(self, detections: pd.DataFrame, metadatas: pd.DataFrame, engine) -> pd.DataFrame:
        for image_id in metadatas.index:
            image = cv2_load_image(metadatas.loc[image_id].file_path)
            image_detections = detections[detections.image_id == image_id]

            if len(image_detections) != 0:
                self.datapipe.update(image, metadatas, image_detections)
                for batch in self.dataloader(engine):
                    detections = engine.default_step(batch, self.name, detections, image=image)
        return detections

    @abstractmethod
    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        """Adapts the default input to your specific case.

        Args:
            image: a numpy array of the current image
            detections: a DataFrame containing all the detections pertaining to a single
                        image
            metadata: additional information about the image

        Returns:
            preprocessed_sample: input for the process function
        """
        pass

    @abstractmethod
    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        """The main processing function. Runs on GPU.

        Args:
            batch: The batched outputs of `preprocess`
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

    @property
    def datapipe(self):
        if self._datapipe is None:
            self._datapipe = EngineDatapipe(self)
        return self._datapipe

    def dataloader(self, engine: "TrackingEngine"):
        datapipe = self.datapipe
        return DataLoader(
            dataset=datapipe,
            batch_size=2**16,
            num_workers=engine.num_workers,
            persistent_workers=False,
        )
