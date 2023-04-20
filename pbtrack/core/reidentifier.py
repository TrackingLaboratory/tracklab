from typing import List, Any, Union
from abc import abstractmethod, ABC

import pandas as pd
from torch.utils.data import DataLoader

import pbtrack
from ..utils.collate import default_collate


class ReIdentifier(ABC):
    """Abstract class to implement for the integration of a new re-identifier in wrappers/reid.
    The functions to implement are __init__, preprocess and process.
    A description of the expected behavior is provided below.
    """

    @abstractmethod
    def __init__(self, cfg, device, batch_size):
        """Init function
        Args:
            cfg (NameSpace): configuration file from Hydra for the detector
            device (int): device to use for the detector
            batch_size (int): batch size for the detector
        """
        self.cfg = cfg
        self.device = device
        self.batch_size = batch_size
        self._datapipe = None

    @abstractmethod
    def preprocess(self, detection: pd.Series, metadata: pd.Series) -> Any:
        """Your preprocessing function to adapt the input to your detector.
        The output is being batched by the collate_fn() and fed to process().
        Args:
            detection (pd.Series): the detection to process
            metadata (pd.Series): the image metadata to process
        Returns:
            preprocessed (Any): preprocessed input for the process step
        """
        pass

    @abstractmethod
    def process(
            self, batch: Any, detections: pd.DataFrame
    ) -> Union[pd.Series, List[pd.Series], pd.DataFrame, List[pd.DataFrame]]:
        """Your processing function to run the re-identifier
        Args:
            batch (Any): the batched outputs from preprocess() by collate_fn()
            detections (pd.DataFrame): the images metadata associated to the batch
        Returns:
            detections (Union[pd.Series, List[pd.Series], pd.DataFrame, List[pd.DataFrame]]): the new detections
            from the batch. The framework will aggregate automatically all the results according to the `name` of the
            Series/`index` of the DataFrame. It is thus mandatory here to name correctly your series or index your
            dataframes. The output will override the previous detections with the same name/index.
        """
        pass

    @property
    def datapipe(self):
        if self._datapipe is None:
            self._datapipe = pbtrack.EngineDatapipe(self)
        return self._datapipe

    def dataloader(self, engine: "pbtrack.TrackingEngine"):
        datapipe = self.datapipe
        return DataLoader(
            dataset=datapipe,
            batch_size=self.batch_size,
            collate_fn=default_collate,
            num_workers=engine.num_workers,
            persistent_workers=False,
        )
