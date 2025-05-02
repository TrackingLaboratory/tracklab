import pycocotools

from tracklab.pipeline import Evaluator


class SoccerAccuracy(Evaluator):
    def __init__(self, eval_set, *args, **kwargs):
        self.eval_set = eval_set

    def run(self, tracker_state):
        pycocotools