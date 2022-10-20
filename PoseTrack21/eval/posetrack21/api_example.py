import posetrack21.api as api 

"""
obtain the evaluator class. Possible eval_types: ['pose_tracking', 'reid_tracking', 'posetrack_mot']
"""
evaluator = api.get_api(trackers_folder='/home/group-cvg/doering/2022/PoseTrackReIDEvaluationData/dummy_pr/', 
                        gt_folder='/home/group-cvg/doering/2022/PoseTrackReIDEvaluationData/dummy_gt/', 
                        eval_type='tracking', 
                        num_parallel_cores=8,
                        use_parallel=True)

# obtain results for each evaluation threshold and for each joint class respectively, i.e. 19x16.
# The last element, i.e. results['HOTA'][:, -1] is the total score over all keypoints.
results = evaluator.eval()

# get average results over evaluation thresholds
avg_results = evaluator.get_avg_results(results)
