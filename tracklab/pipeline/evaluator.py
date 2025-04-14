from abc import abstractmethod, ABC

from tracklab.datastruct.tracker_state import TrackerState


# FIXME is this usefull ?
class Evaluator(ABC):
    """Abstract class to implement for the integration of a new dataset evaluator
    in wrappers/eval. The functions to implement are __init__ and run. A description
    of the expected behavior is provided below."""

    @abstractmethod
    def __init__(self, cfg):
        """Init function
        Args:
            cfg (NameSpace): configuration file from Hydra for the evaluator
        """
        self.cfg = cfg

    @abstractmethod
    def run(self, tracker_state: TrackerState):
        """Run the evaluation
        Args:
            tracker_state (TrackerState): the tracker state for the evaluation
        """
        pass
