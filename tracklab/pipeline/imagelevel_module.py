from abc import abstractmethod
from typing import Any

import pandas as pd
from torch.utils.data.dataloader import default_collate, DataLoader
from tracklab.datastruct import EngineDatapipe
from tracklab.pipeline import Module


class ImageLevelModule(Module):
    """Abstract class to implement a module that operates directly on images.

    This can for example be a bounding box detector, or a bottom-up
    pose estimator (which outputs keypoints directly).

    The functions to implement are
     - __init__, which can take any configuration needed
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

    @abstractmethod
    def __init__(self, batch_size: int):
        """Init function

        The arguments to this function are completely free
        and will be provided by a configuration file.

        You should call the __init__ function from the super() class.
        """
        self.batch_size = batch_size
        self._datapipe = None

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
            batch_size=self.batch_size,
            collate_fn=type(self).collate_fn,
            num_workers=engine.num_workers,
            persistent_workers=False,
        )
