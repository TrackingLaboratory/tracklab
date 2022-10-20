import time
import traceback
import numpy as np 
from multiprocessing.pool import Pool
from functools import partial
import os
from . import utils
from .utils import TrackEvalException
from . import _timing
from .metrics import Count


class EvaluatorReid:
    """Evaluator class for evaluating different metrics for different datasets"""

    @staticmethod
    def get_default_eval_config():
        """Returns the default config values for evaluation"""
        code_path = utils.get_code_path()
        default_config = {
            'BREAK_ON_ERROR': True,  # Raises exception and exits with error
            'RETURN_ON_ERROR': False,  # if not BREAK_ON_ERROR, then returns from function on error
            'LOG_ON_ERROR': os.path.join(code_path, 'error_log.txt'),  # if not None, save any errors into a log file.

            'PRINT_RESULTS': False,
            'PRINT_COMBINED_SUMMARY_ONLY': False,
            'PRINT_PAPER_SUMMARY': False,
            'TIME_PROGRESS': True,
            'DISPLAY_LESS_PROGRESS': True,

            'OUTPUT_PAPER_SUMMARY': True,
            'OUTPUT_DETAILED': False,
        }
        return default_config

    def __init__(self, config=None):
        """Initialise the evaluator with a config file"""
        self.config = utils.init_config(config, self.get_default_eval_config(), 'Eval')
        # Only run timing analysis if not run in parallel.
        if self.config['TIME_PROGRESS'] :
            _timing.DO_TIMING = True
            if self.config['DISPLAY_LESS_PROGRESS']:
                _timing.DISPLAY_LESS_PROGRESS = True

    @_timing.time
    def evaluate(self, dataset_list, metrics_list):
        """Evaluate a set of metrics on a set of datasets"""
        config = self.config
        metrics_list = metrics_list + [Count()]  # Count metrics are always run
        metric_names = utils.validate_metrics_list(metrics_list)
        dataset_names = [dataset.get_name() for dataset in dataset_list]
        output_res = {}
        output_msg = {}

        for dataset, dataset_name in zip(dataset_list, dataset_names):
            # Get dataset info about what to evaluate
            output_res[dataset_name] = {}
            output_msg[dataset_name] = {}
            tracker_list, seq_list, class_list = dataset.get_eval_info()
            print('\nEvaluating %i tracker(s) on %i sequence(s) for %i class(es) on %s dataset using the following '
                  'metrics: %s\n' % (len(tracker_list), len(seq_list), len(class_list), dataset_name,
                                     ', '.join(metric_names)))

            # Evaluate each tracker
            for tracker in tracker_list:
                try:
                    print('\nEvaluating %s\n' % tracker)
                    time_start = time.time()
                    assert "HOTAReidKeypoints" in metric_names 

                    processed_seqs, global_gt_ids, global_pr_ids, total_frames = pre_process_sequences(seq_list, dataset, tracker, class_list)
                    reid_res = dict()
                    res = dict()
                    for cls in class_list:
                        cls_sequences = processed_seqs[cls]
                        for metric, met_name in zip(metrics_list, metric_names):
                            if met_name == 'HOTAReidKeypoints':
                                reid_res[cls] = metric.eval_sequences(cls_sequences, global_gt_ids, global_pr_ids, total_frames)
                            else:
                                # ToDO: run other metrics 
                                for seq_name, seq_data in cls_sequences.items():
                                    res[seq_name] = dict() 
                                    #print(f"[{cls}][{met_name}][{seq_name}]")
                                    res[seq_name][cls] = dict()
                                    res[seq_name][cls][met_name] = metric.eval_sequence(seq_data)


                    # collecting combined cls keys (cls averaged, det averaged, super classes)
                    combined_cls_keys = []
                    res['COMBINED_SEQ'] = {}
                    # combine sequences for each class
                    for c_cls in class_list:
                        res['COMBINED_SEQ'][c_cls] = {}
                        for metric, metric_name in zip(metrics_list, metric_names):
                            if metric_name == 'HOTAReidKeypoints':
                                continue
                            curr_res = {seq_key: seq_value[c_cls][metric_name] for seq_key, seq_value in res.items() if
                                        seq_key != 'COMBINED_SEQ'}
                            res['COMBINED_SEQ'][c_cls][metric_name] = metric.combine_sequences(curr_res)
                        res['COMBINED_SEQ'][c_cls]['HOTAReidKeypoints'] = reid_res[c_cls]

                    # Print and output results in various formats
                    if config['TIME_PROGRESS']:
                        print('\nAll sequences for %s finished in %.2f seconds' % (tracker, time.time() - time_start))
                    output_fol = dataset.get_output_fol(tracker)
                    tracker_display_name = dataset.get_display_name(tracker)
                    for c_cls in res['COMBINED_SEQ'].keys():  # class_list + combined classes if calculated
                        summaries = []
                        paper_summaries = []
                        details = []
                        num_dets = res['COMBINED_SEQ'][c_cls]['Count']['Dets']

                        for metric, metric_name in zip(metrics_list, metric_names):
                            # for combined classes there is no per sequence evaluation
                            table_res = {'COMBINED_SEQ': res['COMBINED_SEQ'][c_cls][metric_name]}
                            if config['PRINT_RESULTS'] and config['PRINT_COMBINED_SUMMARY_ONLY']:
                                metric.print_table({'COMBINED_SEQ': table_res['COMBINED_SEQ']},
                                                   tracker_display_name, c_cls)

                            elif config['PRINT_RESULTS'] and not config['OUTPUT_DETAILED']:
                                metric.print_table(table_res, tracker_display_name, c_cls)
                            elif config['PRINT_RESULTS'] and config['OUTPUT_DETAILED']:
                                metric.print_table_detailed(table_res, tracker_display_name, c_cls)

                            if config['PRINT_PAPER_SUMMARY']:
                                metric.print_paper_summary(table_res, tracker_display_name, c_cls)

                            if config['OUTPUT_PAPER_SUMMARY']:                              
                                paper_summaries.append(metric.paper_summary_results(table_res))

                        utils.write_paper_summary_results(paper_summaries, c_cls, output_fol, is_reid=True, _file_name='pose_reid_hota_results.txt')
                    # Output for returning from function
                    output_res[dataset_name][tracker] = res
                    output_msg[dataset_name][tracker] = 'Success'

                except Exception as err:
                    output_res[dataset_name][tracker] = None
                    if type(err) == TrackEvalException:
                        output_msg[dataset_name][tracker] = str(err)
                    else:
                        output_msg[dataset_name][tracker] = 'Unknown error occurred.'
                    print('Tracker %s was unable to be evaluated.' % tracker)
                    print(err)
                    traceback.print_exc()
                    if config['LOG_ON_ERROR'] is not None:
                        with open(config['LOG_ON_ERROR'], 'a') as f:
                            print(dataset_name, file=f)
                            print(tracker, file=f)
                            print(traceback.format_exc(), file=f)
                            print('\n\n\n', file=f)
                    if config['BREAK_ON_ERROR']:
                        raise err
                    elif config['RETURN_ON_ERROR']:
                        return output_res, output_msg

        return output_res, output_msg

@_timing.time 
def pre_process_sequences(seq_list, dataset, tracker, class_list):
    total_frames = 0 
    raw_seqs = dict() 
    for curr_seq in sorted(seq_list):
        raw_seqs[curr_seq] = dataset.get_raw_seq_data(tracker, curr_seq)
    
    pre_seqs = dict() 
    for cls in class_list: 
        pre_seqs[cls] = dict() 
        # First obtain all gt person ids and pred person ids
        all_gt_ids = list()
        all_pr_ids = list() 
        for curr_seq, raw_data in raw_seqs.items():    
            pre_seqs[cls][curr_seq] = dataset.get_preprocessed_seq_data(raw_data, cls)
            original_gt_ids = list()
            original_tracker_ids = list()
            total_frames += pre_seqs[cls][curr_seq]['num_timesteps']
            for t in range(pre_seqs[cls][curr_seq]['num_timesteps']):
                gt_ids_t = pre_seqs[cls][curr_seq]['original_gt_ids'][t]
                tracker_ids_t = pre_seqs[cls][curr_seq]['original_tracker_ids'][t]
                
                if gt_ids_t is not None:
                    original_gt_ids += gt_ids_t.tolist()

                if tracker_ids_t is not None:
                    original_tracker_ids += tracker_ids_t.tolist()

            curr_gt_ids = np.unique(original_gt_ids)
            curr_pr_ids = np.unique(original_tracker_ids)

            if len(curr_gt_ids) > 0:
                all_gt_ids += curr_gt_ids.tolist()
            if len(curr_pr_ids) > 0:
                all_pr_ids += curr_pr_ids.tolist()
            
        if len(all_gt_ids) > 0:
            unique_gt_ids = np.unique(all_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))

        if len(all_pr_ids) > 0:
            unique_pr_ids = np.unique(all_pr_ids)
            pr_id_map = np.nan * np.ones((np.max(all_pr_ids) + 1))
            pr_id_map[unique_pr_ids] = np.arange(len(unique_pr_ids))

        # re-label all ids! 
        for curr_seq, pre_seq in pre_seqs[cls].items():
            for t in range(pre_seq['num_timesteps']):
                gt_ids_t = pre_seq['original_gt_ids'][t]
                tracker_ids_t = pre_seq['original_tracker_ids'][t]
                
                if gt_ids_t is not None and len(gt_ids_t) > 0:
                    pre_seq['gt_ids'][t] = gt_id_map[gt_ids_t].astype(int)

                if tracker_ids_t is not None and len(tracker_ids_t) > 0:
                    pre_seq['tracker_ids'][t] = pr_id_map[tracker_ids_t].astype(int) 

    return pre_seqs, unique_gt_ids, unique_pr_ids, total_frames

@_timing.time
def eval_sequence(seq, dataset, tracker, class_list, metrics_list, metric_names):
    """Function for evaluating a single sequence"""
    raw_data = dataset.get_raw_seq_data(tracker, seq)
    seq_res = {}
    for cls in class_list:
        seq_res[cls] = {}
        data = dataset.get_preprocessed_seq_data(raw_data, cls)
        for metric, met_name in zip(metrics_list, metric_names):
            seq_res[cls][met_name] = metric.eval_sequence(data)
    return seq_res
