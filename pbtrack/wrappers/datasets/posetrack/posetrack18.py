from .posetrack21 import PoseTrack21


class PoseTrack18(PoseTrack21):
    def __init__(self, dataset_path: str, annotation_path: str, *args, **kwargs):
        super().__init__(
            dataset_path, annotation_path, posetrack_version=18, *args, **kwargs
        )
