import time
import traceback
from multiprocessing.pool import Pool
from functools import partial
import os
from . import utils
from .utils import TrackEvalException
from . import _timing
from .metrics import Count


class Evaluator:
    """Evaluator class for evaluating different metrics for different datasets"""

    @staticmethod
    def get_default_eval_config():
        """Returns the default config values for evaluation"""
        code_path = utils.get_code_path()
        default_config = {
            'USE_PARALLEL': False,
            'NUM_PARALLEL_CORES': 8,
            'BREAK_ON_ERROR': True,  # Raises exception and exits with error
            'RETURN_ON_ERROR': False,  # if not BREAK_ON_ERROR, then returns from function on error
            'LOG_ON_ERROR': os.path.join(code_path, 'error_log.txt'),  # if not None, save any errors into a log file.

            'PRINT_RESULTS': False,
            'PRINT_COMBINED_SUMMARY_ONLY': False,
            'PRINT_PAPER_SUMMARY': False,
            'TIME_PROGRESS': True,
            'DISPLAY_LESS_PROGRESS': True,

            'OUTPUT_PAPER_SUMMARY': True,
            'OUTPUT_EMPTY_CLASSES': True,  # If False, summary files are not output for classes with no detections
            'OUTPUT_DETAILED': False,
        }
        return default_config

    def __init__(self, config=None):
        """Initialise the evaluator with a config file"""
        self.config = utils.init_config(config, self.get_default_eval_config(), 'Eval')
        # Only run timing analysis if not run in parallel.
        if self.config['TIME_PROGRESS'] and not self.config['USE_PARALLEL']:
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
                    if config['USE_PARALLEL']:
                        with Pool(config['NUM_PARALLEL_CORES']) as pool:
                            _eval_sequence = partial(eval_sequence, dataset=dataset, tracker=tracker,
                                                     class_list=class_list, metrics_list=metrics_list,
                                                     metric_names=metric_names)
                            results = pool.map(_eval_sequence, seq_list)
                            res = dict(zip(seq_list, results))
                    else:
                        res = {}
                        for curr_seq in sorted(seq_list):
                            res[curr_seq] = eval_sequence(curr_seq, dataset, tracker, class_list, metrics_list,
                                                          metric_names)

                    combined_cls_keys = []
                    res['COMBINED_SEQ'] = {}
                    # combine sequences for each class
                    for c_cls in class_list:
                        res['COMBINED_SEQ'][c_cls] = {}
                        for metric, metric_name in zip(metrics_list, metric_names):
                            curr_res = {seq_key: seq_value[c_cls][metric_name] for seq_key, seq_value in res.items() if
                                        seq_key is not 'COMBINED_SEQ'}
                            res['COMBINED_SEQ'][c_cls][metric_name] = metric.combine_sequences(curr_res)

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
                            if c_cls in combined_cls_keys:
                                table_res = {'COMBINED_SEQ': res['COMBINED_SEQ'][c_cls][metric_name]}
                            else:
                                table_res = {seq_key: seq_value[c_cls][metric_name] for seq_key, seq_value
                                             in res.items()}
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

                        utils.write_paper_summary_results(paper_summaries, c_cls, output_fol, _file_name='mot_hota_results.txt')
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
def eval_sequence(seq, dataset, tracker, class_list, metrics_list, metric_names):
    # import pdb; pdb.set_trace()
    """Function for evaluating a single sequence"""
    raw_data = dataset.get_raw_seq_data(tracker, seq)
    seq_res = {}
    for cls in class_list:
        seq_res[cls] = {}
        data = dataset.get_preprocessed_seq_data(raw_data, cls)
        for metric, met_name in zip(metrics_list, metric_names):
            seq_res[cls][met_name] = metric.eval_sequence(data)
    return seq_res
