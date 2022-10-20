from .posetrackreid_api import PoseTrackReIDEvaluator 
from .posetrack_api import PoseTrackEvaluator 
from .posetrack_mot_api import PoseTrackMOTEvaluator 
from .posetrack_pose_estim_api import PoseTrackPoseEvaluator

__all__ = ['get_api',  'PoseTrackMOTEvaluator',  'PoseTrackEvaluator',  'PoseTrackReIDEvaluator', 'PoseTrackPoseEvaluator']


def get_api(trackers_folder, gt_folder, use_parallel, num_parallel_cores, eval_type, **kwargs):
    """
    eval_type: ['pose_estim', 'pose_tracking', 'reid_tracking', 'posetrack_mot']
    """

    if eval_type == 'tracking':
        return PoseTrackEvaluator(trackers_folder, gt_folder, use_parallel, num_parallel_cores, **kwargs)
    elif eval_type == 'reid_tracking':
        return PoseTrackReIDEvaluator(trackers_folder, gt_folder, **kwargs)
    elif eval_type == 'posetrack_mot':
        return PoseTrackMOTEvaluator(trackers_folder, gt_folder, use_parallel, num_parallel_cores, **kwargs)
    elif eval_type == 'pose_estim':
        return PoseTrackPoseEvaluator(trackers_folder, gt_folder, use_parallel, num_parallel_cores, **kwargs)
    else:
        raise NotImplementedError(f"No api for eval_type: {eval_type}")
