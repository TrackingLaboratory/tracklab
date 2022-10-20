
class BaseEvaluator:

    def __init__(self, config, trackers_folder, gt_folder, use_parallel, num_parallel_cores, **kwargs):
        
        self.setup_config(config, trackers_folder, gt_folder, use_parallel, num_parallel_cores)

        for k, v in kwargs.items():
            if k in config:
                config[k] = v

    def setup_config(self, config, trackers_folder, gt_folder, use_parallel, num_parallel_cores):

        config['TRACKERS_FOLDER'] = trackers_folder 
        config['GT_FOLDER'] = gt_folder
        config['USE_PARALLEL'] = use_parallel 
        config['NUM_PARALLEL_CORES'] = num_parallel_cores 
        config['PRINT_RESULTS'] = False 
        config['PRINT_ONLY_COMBINED'] = False 
        config['OUTPUT_DETAILED'] = False 
        config['PRINT_PAPER_SUMMARY'] = False
        config['ASSURE_SINGLE_TRACKER'] = True 

    def eval(self, mean_only=True):
        output_res, output_msg = self.evaluator.evaluate(self.dataset_list, self.metrics_list)

        pt_results = output_res[f'{self.dataset}']
        tracker_name = list(pt_results.keys())[0]
        results = pt_results[tracker_name]['COMBINED_SEQ']['person'][f'{self.metric}']

        return results 

    @staticmethod
    def get_avg_results(results):

        avg_results = dict() 
        for k,  v in results.items():
            avg_results[k] = v.mean(axis=0)

        return avg_results


