import posetrack21.trackeval as trackeval
from posetrack21.api.base_api import BaseEvaluator 


class PoseTrackReIDEvaluator(BaseEvaluator):
    def __init__(self, trackers_folder, gt_folder, **kwargs):
        self.metric = 'HOTAReidKeypoints'
        self.dataset = 'PoseTrackReID'
        
        default_eval_config = trackeval.EvaluatorReid.get_default_eval_config()
        default_dataset_config = trackeval.datasets.PoseTrackReID.get_default_dataset_config()
        default_metrics_config = {'METRICS': [f'{self.metric}']}

        config = {**default_eval_config, **default_dataset_config, **default_metrics_config}
        super().__init__(config, trackers_folder, gt_folder, use_parallel=False, num_parallel_cores=0, **kwargs)

        # split config dict 
        eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
        dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()} 
        metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()} 
        self.evaluator = trackeval.EvaluatorReid(eval_config)
        self.dataset_list = [trackeval.datasets.PoseTrackReID(dataset_config)]
        self.metrics_list = list() 

        for metric in [trackeval.metrics.HOTAReidKeypoints]:
            if metric.get_name() in metrics_config['METRICS']:
                self.metrics_list.append(metric()) 
        if len(self.metrics_list) == 0:
            raise Exception("No metrics selected for evaluation")


