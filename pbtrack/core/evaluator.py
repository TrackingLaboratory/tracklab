from abc import abstractmethod, ABC
from pbtrack.datastruct.tracker_state import TrackerState


class Evaluator(ABC):
    @abstractmethod
    def run(self, tracker_state: TrackerState):
        pass
