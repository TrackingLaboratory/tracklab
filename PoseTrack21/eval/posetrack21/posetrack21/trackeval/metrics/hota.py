
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from .. import _timing


class HOTA(_BaseMetric):
    """Class which implements the HOTA metrics.
    See: https://link.springer.com/article/10.1007/s11263-020-01375-2
    """
    def __init__(self, config=None):
        super().__init__()
        self.plottable = True
        self.array_labels = np.arange(0.05, 0.99, 0.05)
        self.integer_fields = ['HOTA_TP(0)', 'HOTA_FN(0)', 'HOTA_FP(0)']
        self.integer_array_fields = ['HOTA_TP', 'HOTA_FN', 'HOTA_FP']
        self.float_array_fields = ['HOTA', 'DetA', 'AssA', 'FragA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA', 'RHOTA', 'FA-HOTA', 'FA-RHOTA']
        self.float_fields = ['HOTA(0)', 'LocA(0)', 'HOTALocA(0)']
        self.fields = self.float_array_fields + self.integer_array_fields + self.float_fields + self.integer_fields
        self.summary_fields = self.float_array_fields + self.float_fields
    
    def print_table_detailed(self, table_res, tracker, cls):
        pass

    @_timing.time
    def eval_sequence(self, data):
        """Calculates the HOTA metrics for one sequence"""

        # Initialise results
        res = {}
        for field in self.float_array_fields + self.integer_array_fields:
            res[field] = np.zeros((len(self.array_labels)), dtype=np.float)
        for field in self.float_fields:
            res[field] = 0

        # Return result quickly if tracker or gt sequence is empty
        if data['num_tracker_dets'] == 0:
            res['HOTA_FN'] = data['num_gt_dets'] * np.ones((len(self.array_labels)), dtype=np.float)
            res['LocA'] = np.ones((len(self.array_labels)), dtype=np.float)
            res['LocA(0)'] = 1.0
            res = self._compute_final_fields(res)
            return res
        if data['num_gt_dets'] == 0:
            res['HOTA_FP'] = data['num_tracker_dets'] * np.ones((len(self.array_labels)), dtype=np.float)
            res['LocA'] = np.ones((len(self.array_labels)), dtype=np.float)
            res['LocA(0)'] = 1.0
            res = self._compute_final_fields(res)
            return res

        # Variables counting global association
        potential_matches_count = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
        gt_id_count = np.zeros((data['num_gt_ids'], 1))
        tracker_id_count = np.zeros((1, data['num_tracker_ids']))

        last_matched_id = np.ones((len(self.array_labels), data['num_gt_ids']), dtype=int) * -1 
        num_gt_fragmentations = np.zeros((len(self.array_labels), data['num_gt_ids']), dtype=int)
        tp_fragmentation_count = np.zeros((len(self.array_labels), data['num_gt_ids'], data['num_tracker_ids']), dtype=int)
        fragments = np.zeros((len(self.array_labels), data['num_gt_ids'], data['num_tracker_ids'], data['num_timesteps']), dtype=int)

        # First loop through each timestep and accumulate global track information.
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            # Count the potential matches between ids in each timestep
            # These are normalised, weighted by the match similarity.
            similarity = data['similarity_scores'][t]
            sim_iou_denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
            sim_iou = np.zeros_like(similarity)
            sim_iou_mask = sim_iou_denom > 0 + np.finfo('float').eps
            sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]
            potential_matches_count[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] += sim_iou

            # Calculate the total number of dets for each gt_id and tracker_id.
            gt_id_count[gt_ids_t] += 1
            tracker_id_count[0, tracker_ids_t] += 1

        # Calculate overall jaccard alignment score (before unique matching) between IDs
        global_alignment_score = potential_matches_count / (gt_id_count + tracker_id_count - potential_matches_count)
        matches_counts = [np.zeros_like(potential_matches_count) for _ in self.array_labels]

        # Calculate scores for each timestep
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            # Deal with the case that there are no gt_det/tracker_det in a timestep.
            if len(gt_ids_t) == 0:
                for a, alpha in enumerate(self.array_labels):
                    res['HOTA_FP'][a] += len(tracker_ids_t)
                continue
            if len(tracker_ids_t) == 0:
                for a, alpha in enumerate(self.array_labels):
                    res['HOTA_FN'][a] += len(gt_ids_t)
                continue

            # Get matching scores between pairs of dets for optimizing HOTA
            similarity = data['similarity_scores'][t]
            score_mat = global_alignment_score[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] * similarity

            # Hungarian algorithm to find best matches
            match_rows, match_cols = linear_sum_assignment(-score_mat)

            # Calculate and accumulate basic statistics
            for a, alpha in enumerate(self.array_labels):
                actually_matched_mask = similarity[match_rows, match_cols] >= alpha - np.finfo('float').eps
                alpha_match_rows = match_rows[actually_matched_mask]
                alpha_match_cols = match_cols[actually_matched_mask]
                num_matches = len(alpha_match_rows)
                res['HOTA_TP'][a] += num_matches
                res['HOTA_FN'][a] += len(gt_ids_t) - num_matches
                res['HOTA_FP'][a] += len(tracker_ids_t) - num_matches
                if num_matches > 0:
                    res['LocA'][a] += sum(similarity[alpha_match_rows, alpha_match_cols])
                    matches_counts[a][gt_ids_t[alpha_match_rows], tracker_ids_t[alpha_match_cols]] += 1
                
                # =============================== Count Fragmentations ==============================
                matched_gt_ids = gt_ids_t[alpha_match_rows]
                matched_det_ids = tracker_ids_t[alpha_match_cols]

                # get gt id and pr id with a new fragmentation
                last_matched_id_a = last_matched_id[a, matched_gt_ids]
                fragmentation_idxs = last_matched_id_a != matched_det_ids
                gt_id_w_fragmentation = matched_gt_ids[fragmentation_idxs] 
                pr_id_w_fragmentation = matched_det_ids[fragmentation_idxs]

                # update fragmentation counts 
                last_matched_id[a, gt_id_w_fragmentation] = matched_det_ids[fragmentation_idxs]
                num_gt_fragmentations[a, gt_id_w_fragmentation] += 1
                tp_fragmentation_count[a, gt_id_w_fragmentation, pr_id_w_fragmentation] += 1

                # count tp inside current fragment! 
                fragment_indices = np.maximum(0, tp_fragmentation_count[a, matched_gt_ids, matched_det_ids] - 1)
                fragments[a, matched_gt_ids, matched_det_ids, fragment_indices] += 1

        # Calculate association scores (AssA, AssRe, AssPr) for the alpha value.
        # First calculate scores per gt_id/tracker_id combo and then average over the number of detections.
        for a, alpha in enumerate(self.array_labels):
            matches_count = matches_counts[a]

            tpa_fna_fpa = np.maximum(1, gt_id_count + tracker_id_count - matches_count)

            ass_a = matches_count / tpa_fna_fpa 
            res['AssA'][a] = np.sum(matches_count * ass_a) / np.maximum(1, res['HOTA_TP'][a])

            ass_re = matches_count / np.maximum(1, gt_id_count)
            res['AssRe'][a] = np.sum(matches_count * ass_re) / np.maximum(1, res['HOTA_TP'][a])
            ass_pr = matches_count / np.maximum(1, tracker_id_count)
            res['AssPr'][a] = np.sum(matches_count * ass_pr) / np.maximum(1, res['HOTA_TP'][a])
            
            curr_fragments = fragments[a]
            frag = curr_fragments / tpa_fna_fpa[:, :, np.newaxis]
            frag = np.sum(curr_fragments * frag, -1)
            res['FragA'][a] = np.sum(np.sum(frag, 0), 0) / np.maximum(1, res['HOTA_TP'][a])

        # Calculate final scores
        res['LocA'] = np.maximum(1e-10, res['LocA']) / np.maximum(1e-10, res['HOTA_TP'])
        res = self._compute_final_fields(res)
        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.integer_fields:
            res[field] = self._combine_sum(all_res, field)
        for field in self.integer_array_fields:
            res[field] = self._combine_sum(all_res, field)
        for field in ['AssRe', 'AssPr', 'AssA', 'FragA']:
            res[field] = self._combine_weighted_av(all_res, field, res, weight_field='HOTA_TP')
        loca_weighted_sum = sum([all_res[k]['LocA'] * all_res[k]['HOTA_TP'] for k in all_res.keys()])
        res['LocA'] = np.maximum(1e-10, loca_weighted_sum) / np.maximum(1e-10, res['HOTA_TP'])
        res = self._compute_final_fields(res)
        return res

    def combine_classes_class_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the class values"""
        res = {}
        for field in self.integer_array_fields:
            res[field] = self._combine_sum(
                {k: v for k, v in all_res.items()
                 if (v['HOTA_TP'] + v['HOTA_FN'] + v['HOTA_FP'] > 0 + np.finfo('float').eps).any()}, field)
        for field in self.float_fields:
            res[field] = np.mean([v[field] for v in all_res.values()
                                  if (v['HOTA_TP'] + v['HOTA_FN'] + v['HOTA_FP'] > 0 + np.finfo('float').eps).any()],
                                 axis=0)
        for field in self.float_array_fields:
            res[field] = np.mean([v[field] for v in all_res.values()
                                  if (v['HOTA_TP'] + v['HOTA_FN'] + v['HOTA_FP'] > 0 + np.finfo('float').eps).any()],
                                 axis=0)
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        res = {}
        for field in self.integer_array_fields:
            res[field] = self._combine_sum(all_res, field)
        for field in ['AssRe', 'AssPr', 'AssA']:
            res[field] = self._combine_weighted_av(all_res, field, res, weight_field='HOTA_TP')
        loca_weighted_sum = sum([all_res[k]['LocA'] * all_res[k]['HOTA_TP'] for k in all_res.keys()])
        res['LocA'] = np.maximum(1e-10, loca_weighted_sum) / np.maximum(1e-10, res['HOTA_TP'])
        res = self._compute_final_fields(res)
        return res

    @staticmethod
    def _compute_final_fields(res):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        res['DetRe'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'])
        res['DetPr'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FP'])
        res['DetA'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'] + res['HOTA_FP'])
        res['HOTA'] = np.sqrt(res['DetA'] * res['AssA'])
        res['RHOTA'] = np.sqrt(res['DetRe'] * res['AssA'])

        res['FA-HOTA'] = np.sqrt(res['DetA'] * np.sqrt(res['AssA'] * res['FragA']))
        res['FA-RHOTA'] = np.sqrt(res['DetRe'] * np.sqrt(res['AssA'] * res['FragA']))

        res['HOTA(0)'] = res['HOTA'][0]
        res['LocA(0)'] = res['LocA'][0]
        res['HOTALocA(0)'] = res['HOTA(0)']*res['LocA(0)']
        res['HOTA_TP(0)'] = res['HOTA_TP'][0]
        res['HOTA_FN(0)'] = res['HOTA_FN'][0]
        res['HOTA_FP(0)'] = res['HOTA_FP'][0]

        return res

    def print_paper_summary(self, table_res, tracker, cls):
        print('')
        print("Latex Paper Summary")
        order = ['DetPr', 'DetRe', 'DetA', 'AssPr', 'AssRe', 'AssA', 'LocA', 'FragA', 'HOTA', 'RHOTA', 'FA-HOTA', 'FA-RHOTA']
        order = order + ['LocA(0)', 'HOTALocA(0)', 'HOTA(0)', 'HOTA_TP', 'HOTA_FP', 'HOTA_FN']
        order = order + ['HOTA_TP(0)', 'HOTA_FP(0)', 'HOTA_FN(0)']

        order_print = [metric.replace('_', '\\_') for metric in order]

        self._row_print_latex(['Summary'] + order_print)
        output = ['']
        for metric in order:
            metric_results = table_res['COMBINED_SEQ'][metric]
            summary_res = self._summary_result(metric, metric_results)
            output.append(summary_res[-1])

        self._row_print_latex(output)

    def print_table(self, table_res, tracker, cls):
        """Prints table of results for all sequences"""
        print('')
        metric_name = self.get_name()
        self._row_print([metric_name + '->evaluating: ' + tracker + ':'])
        seq_names = list(table_res.keys())
        metric_names = list(table_res[seq_names[0]].keys())
        self._row_print(['Sequence'] + metric_names, space=15)
        for seq, results in sorted(table_res.items()):
            if seq == 'COMBINED_SEQ':
                continue
            seq_results = []
            for metric, metric_results in results.items():
                summary_res = self._summary_result(metric, metric_results)
                seq_results.append(summary_res[-1])
            self._row_print([seq] + seq_results, space=15)

        self._row_print([metric_name + '->Summary: ' + tracker + ':'])
        self._row_print(['\t'] + self.joint_names + ['Total'])
        self._row_print(['\tMetric'])
        for metric, metric_results in table_res['COMBINED_SEQ'].items():
            summary_res = self._summary_result(metric, metric_results)
            self._row_print([f"\t{metric}"] + summary_res)

    def _summary_row(self, results_):
        vals = []
        for h in self.summary_fields:
            if h in self.float_array_fields:
                vals.append("{0:1.5g}".format(100 * np.mean(results_[h])))
            elif h in self.float_fields:
                vals.append("{0:1.5g}".format(100 * float(results_[h])))
            elif h in self.integer_fields:
                vals.append("{0:d}".format(int(results_[h])))
            else:
                raise NotImplementedError("Summary function not implemented for this field type.")
        return vals

    @staticmethod
    def _row_print(*argv, **kwargs):
        space = kwargs['space'] if 'space' in kwargs else 10

        """Prints results in an evenly spaced rows, with more space in first row"""
        if len(argv) == 1:
            argv = argv[0]
        to_print = '%-25s' % argv[0]
        for v in argv[1:]:
            to_print += f'%-{space}s' % str(v)
        print(to_print)

    @staticmethod
    def _row_print_latex(*argv):
        to_print = HOTAeypoints._row_to_latex(*argv)
        print(to_print)

    @staticmethod
    def _row_to_latex(*argv):
        """Prints results in an evenly spaced rows, with more space in first row"""
        if len(argv) == 1:
            argv = argv[0]
        to_print = f'{argv[0]} &'
        for v in argv[1:-1]:
            to_print += f' {v} &'
        to_print += f' {argv[-1]}'

        return to_print

    def summary_results(self, table_res):
        """Returns a simple summary of final results for a tracker"""
        ret = dict()
        for metric, result in table_res['COMBINED_SEQ'].items():
            ret[metric] = self._summary_result(metric, result)

        return ret

    def paper_summary_results(self, table_res):
        order = ['DetPr', 'DetRe', 'DetA', 'AssPr', 'AssRe', 'AssA', 'LocA', 'FragA', 'HOTA', 'RHOTA', 'FA-HOTA', 'FA-RHOTA']
        order = order + ['LocA(0)', 'HOTALocA(0)', 'HOTA(0)', 'HOTA_TP', 'HOTA_FP', 'HOTA_FN']
        order = order + ['HOTA_TP(0)', 'HOTA_FP(0)', 'HOTA_FN(0)']

        order_print = [metric.replace('_', '\\_') for metric in order]

        rows = [] 
        header_row = self._row_to_latex(['Summary'] + order_print)
        rows.append(header_row)

        output = ['']
        for metric in order:
            metric_results = table_res['COMBINED_SEQ'][metric]
            summary_res = self._summary_result(metric, metric_results)
            output.append(summary_res[-1])

        rows.append(self._row_to_latex(output))

        return rows

    def _summary_result(self, metric, result):
        vals = []
        # we have a float value per joint
        if metric in self.float_fields:
            vals.append("{0:1.5g}".format(100 * result))
        # we have an array for each joint
        elif metric in self.float_array_fields:
            vals.append("{0:1.5g}".format(100 * np.mean(result)))
        # we have an array for each joint
        elif metric in self.integer_array_fields:
                vals.append("{0:d}".format(np.mean(result).astype(np.int)))
        elif metric in self.integer_fields:
            vals.append("{0:d}".format(result.astype(np.int)))
        else:
            raise TrackEvalException(f"Unknown metric {metric}")

        return vals

    def detailed_results(self, table_res):
        """Returns detailed final results for a tracker"""
        # Get detailed field information
        detailed_fields = self.float_fields + self.integer_fields
        for h in self.float_array_fields + self.integer_array_fields:
            for alpha in [int(100*x) for x in self.array_labels]:
                detailed_fields.append(h + '___' + str(alpha))
            detailed_fields.append(h + '___AUC')

        # Get detailed results
        detailed_results = {}
        for seq, res in table_res.items():
            detailed_row = self._detailed_row(res)
            if len(detailed_row) != len(detailed_fields):
                raise TrackEvalException(
                    'Field names and data have different sizes (%i and %i)' % (len(detailed_row), len(detailed_fields)))
            detailed_results[seq] = dict(zip(detailed_fields, detailed_row))
        return detailed_results

    def _detailed_row(self, res):
        detailed_row = []
        for h in self.float_fields + self.integer_fields:
            detailed_row.append(res[h])
        for h in self.float_array_fields + self.integer_array_fields:
            for i, alpha in enumerate([int(100 * x) for x in self.array_labels]):
                detailed_row.append(res[h][i])
            detailed_row.append(np.mean(res[h], axis=0))
        return detailed_row
