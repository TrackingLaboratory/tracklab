
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from .. import _timing

class Joint:
    nose = 0 
    head_bottom = 1
    head_top = 2
    left_shoulder = 3
    right_shoulder = 4 
    left_elbow = 5 
    right_elbow = 6 
    left_wrist = 7 
    right_wrist = 8
    left_hip = 9
    right_hip = 10 
    left_knee = 11 
    right_knee = 12 
    left_ankle = 13
    right_ankle = 14 

class PosemAP(_BaseMetric):
    """Class which implements the HOTA metrics.
    See: https://link.springer.com/article/10.1007/s11263-020-01375-2
    """
    def __init__(self, config=None):
        super().__init__()
        self.plottable = False
        
        self.integer_fields = []
        self.integer_array_fields = ['TP', 'FP', 'FN']
        self.float_array_fields = ['mAP']
        self.float_fields = []

        self.fields = self.float_array_fields + self.integer_array_fields + self.float_fields + self.integer_fields
        self.summary_fields = self.float_array_fields + self.float_fields

    @_timing.time
    def eval_sequence(self, data):
        """Calculates the HOTA metrics for one sequence"""

        res = dict() 

        scores_all = dict()
        labels_all = dict() 
        gt_kpts_ctr = np.zeros((15, len(data['gt_ids'])))

        for j in range(15):
            scores_all[j] = dict()
            labels_all[j] = dict()
            for t in range(len(data['gt_ids'])):
                scores_all[j][t] = np.zeros([0, 0],  dtype=np.float32)
                labels_all[j][t] = np.zeros([0, 0],  dtype=np.int8)

        # First loop through each timestep and accumulate global track information.
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):

            pr_kpts = data['tracker_dets'][t]
            has_pr = (pr_kpts[:, :, 0] > 0) & (pr_kpts[:, :, 1] > 0) 
            pred_scores = pr_kpts[:,  :,  2]

            matches = data['keypoint_matches'][t]
            pck = data['similarity_scores'][t]
            gt_kpts = data['gt_dets'][t]
            has_gt = (gt_kpts[:, :, 0] > 0) & (gt_kpts[:,  :,  1] > 0)

            # update gt count 
            for n in range(has_gt.shape[0]):
                for j_idx in range(has_gt.shape[1]):
                    gt_kpts_ctr[j_idx, t] += int(has_gt[n, j_idx])

            if len(gt_ids_t) > 0 and len(tracker_ids_t) > 0:
                # transpose pck to have predictions on first axis and tracks on second 
                pck = pck.T
                # preserve best GT match only 
                idx = np.argmax(pck,  axis=1)
                val = np.max(pck,  axis=1)
                for det_idx in range(pck.shape[0]):
                    for gt_idx in range(pck.shape[1]):
                        if gt_idx != idx[det_idx]:
                            pck[det_idx,  gt_idx] = 0

                pr_to_gt = np.argmax(pck,  axis=0)
                val = np.max(pck,  axis=0)
                pr_to_gt[val == 0] = - 1

                # assign predicted poses to GT poses 
                for pr_idx in range(has_pr.shape[0]):
                    if (pr_idx in pr_to_gt):
                        # GT pose that matches predicted pose 
                        gt_idx = np.argwhere(pr_to_gt == pr_idx) 
                        assert gt_idx.size == 1
                        gt_idx = gt_idx[0, 0]
                        kpt_scores = pred_scores[pr_idx, :]
                        has_kpt_pred = has_pr[pr_idx]
                        kpt_matches = matches[gt_idx, pr_idx]

                        for j in range(len(has_kpt_pred)):
                            if has_kpt_pred[j]:
                                scores_all[j][t] = np.append(scores_all[j][t],  kpt_scores[j])
                                labels_all[j][t] = np.append(labels_all[j][t],  kpt_matches[j])
                    else:
                        # no matching gt
                        kpt_scores = pred_scores[pr_idx, :]
                        kpt_matches = np.zeros([matches.shape[2],  1],  dtype=bool)
                        has_kpt_pred = has_pr[pr_idx]
                        for j in range(len(has_kpt_pred)):
                            if has_kpt_pred[j]:
                                scores_all[j][t] = np.append(scores_all[j][t],  kpt_scores[j])
                                labels_all[j][t] = np.append(labels_all[j][t],  kpt_matches[j])
            else:
                # no gt, all predictions are false positives!
                for pr_idx in range(has_pr.shape[0]):
                    kpt_scores = pred_scores[pre_idx, :]
                    kpt_matches = np.zeros([matches.shape[2], 1],  dtype==bool)
                    has_kpt_pred = has_pr[pr_idx]

                    for j in range(len(has_kpt_pred)):
                        if has_kpt_pred[j]:
                            scores_all[j][t] = np.append(scores_all[j][t],  kpt_scores[j])
                            labels_all[j][t] = np.append(labels_all[j][t],  kpt_matches[j])

        res['scores_all'] = scores_all 
        res['labels_all'] = labels_all
        res['gt_kpts_ctr'] = gt_kpts_ctr

        ap_all, prec_all, rec_all = PosemAP.compute_metrics(scores_all,  labels_all,  gt_kpts_ctr)

        res['map'] = dict() 
        res['precision'] = dict()
        res['recall'] = dict()

        self.add_metric_results(res['map'], self.get_pose_parts(), self.get_cum_vals(ap_all))
        self.add_metric_results(res['precision'], self.get_pose_parts(), self.get_cum_vals(prec_all))
        self.add_metric_results(res['recall'], self.get_pose_parts(), self.get_cum_vals(rec_all))

        return res

    @staticmethod 
    def add_metric_results(res, names, values):
        for k, v in zip(names, values):
            res[k] = v

    @staticmethod 
    def get_pose_parts():
        return ['Head', 'Shou', 'Elb', 'Wri', 'Hip', 'Knee', 'Ankl', 'Total']

    @staticmethod
    def get_cum_vals(vals):
        # vals being either ap,  prec, recall 
        cum = [] 
        cum += [(vals[[Joint.head_top, Joint.head_bottom, Joint.nose], 0].mean())]
        cum += [(vals[[Joint.left_shoulder, Joint.right_shoulder], 0].mean())]
        cum += [(vals[[Joint.left_elbow, Joint.right_elbow], 0].mean())]
        cum += [(vals[[Joint.left_wrist, Joint.right_wrist], 0].mean())]
        cum += [(vals[[Joint.left_hip, Joint.right_hip], 0].mean())]
        cum += [(vals[[Joint.left_knee, Joint.right_knee], 0].mean())]
        cum += [(vals[[Joint.right_ankle, Joint.left_ankle], 0].mean())]
        cum += [vals[-1, 0]]

        return cum

    @staticmethod
    def compute_metrics(all_scores,  all_labels,  num_gt_kpts):
        ap_all = np.zeros((num_gt_kpts.shape[0] + 1, 1))
        rec_all = np.zeros((num_gt_kpts.shape[0] + 1, 1))
        prec_all = np.zeros((num_gt_kpts.shape[0] + 1, 1))

        for j in range(num_gt_kpts.shape[0]):
            scores = np.zeros([0, 0],  dtype=np.float32)
            labels = np.zeros([0, 0],  dtype=np.int8)

            for fr_idx in range(num_gt_kpts.shape[1]):
                scores = np.append(scores,  all_scores[j][fr_idx])
                labels = np.append(labels,  all_labels[j][fr_idx])

            # compute recall / precision 
            num_gt = sum(num_gt_kpts[j, :])
            precision, recall, scores_sorted_idxs = PosemAP.compute_rpc(scores,  labels,  num_gt)
            if len(precision) > 0:
                ap_all[j] = PosemAP.VOCap(recall, precision) * 100 
                prec_all[j] = precision[-1] * 100 
                rec_all[j] = recall[-1] * 100

        idxs = np.argwhere(~np.isnan(ap_all[:num_gt_kpts.shape[0], 0]))
        ap_all[num_gt_kpts.shape[0]] = ap_all[idxs, 0].mean()
        idxs = np.argwhere(~np.isnan(rec_all[:num_gt_kpts.shape[0], 0]))
        rec_all[num_gt_kpts.shape[0]] = rec_all[idxs, 0].mean() 
        idxs = np.argwhere(~np.isnan(prec_all[:num_gt_kpts.shape[0], 0]))
        prec_all[num_gt_kpts.shape[0]] = prec_all[idxs, 0].mean()

        return ap_all,  prec_all,  rec_all

    # compute Average Precision using recall/precision values
    @staticmethod
    def VOCap(rec,prec):

        mpre = np.zeros([1,2+len(prec)])
        mpre[0,1:len(prec)+1] = prec
        mrec = np.zeros([1,2+len(rec)])
        mrec[0,1:len(rec)+1] = rec
        mrec[0,len(rec)+1] = 1.0

        for i in range(mpre.size-2,-1,-1):
            mpre[0,i] = max(mpre[0,i],mpre[0,i+1])

        i = np.argwhere( ~np.equal( mrec[0,1:], mrec[0,:mrec.shape[1]-1]) )+1
        i = i.flatten()

        # compute area under the curve
        ap = np.sum( np.multiply( np.subtract( mrec[0,i], mrec[0,i-1]), mpre[0,i] ) )

        return ap

    @staticmethod
    def compute_rpc(scores, labels, total_pos):
        precision = np.zeros(len(scores))
        recall = np.zeros(len(scores)) 
        npos = 0 

        idxs_sort = np.array(scores).argsort()[::-1]
        labels_sort = labels[idxs_sort] 

        for s_idx in range(len(idxs_sort)):
            if labels_sort[s_idx] == 1:
                npos += 1 

            # recall: how many true positives were found out of the total number of positives
            recall[s_idx] = npos / max(total_pos, 1)
            # precision: how many true positives were found out of total num of samples
            precision[s_idx] = npos / (s_idx + 1)

        return precision,  recall,  idxs_sort

    def combine_sequences(self, all_res):
        res = self._compute_final_fields(all_res)
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
        res = self._compute_final_fields(res)
        return res

    @staticmethod
    def _compute_final_fields(res):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        combined_scores = dict() 
        combined_labels = dict()
        combined_gt_kpts = list()

        for j in range(15):
            combined_scores[j] = dict() 
            combined_labels[j] = dict()

        total_frames = 0

        for seq_name, seq_res in res.items():
                scores = seq_res['scores_all']
                labels = seq_res['labels_all']
                num_gt = seq_res['gt_kpts_ctr']

                num_kpts = len(scores)
                for j in range(num_kpts):
                    for fr_idx in scores[j].keys():
                        total_frame_idx = fr_idx + total_frames
                        combined_scores[j][total_frame_idx] = scores[j][fr_idx]
                        combined_labels[j][total_frame_idx] = labels[j][fr_idx]
                
                total_frames += num_gt.shape[1]
                # add gt count
                combined_gt_kpts.append(num_gt)

        combined_gt_kpts = np.concatenate(combined_gt_kpts, axis=1)
        total_ap, precision, recall = PosemAP.compute_metrics(combined_scores,  combined_labels, combined_gt_kpts) 
        
        out = dict()
        out['map'] = dict() 
        out['precision'] = dict()
        out['recall'] = dict()

        PosemAP.add_metric_results(out['map'], PosemAP.get_pose_parts(), PosemAP.get_cum_vals(total_ap))
        PosemAP.add_metric_results(out['precision'], PosemAP.get_pose_parts(), PosemAP.get_cum_vals(precision))
        PosemAP.add_metric_results(out['recall'], PosemAP.get_pose_parts(), PosemAP.get_cum_vals(recall))

        return out

    def print_paper_summary(self, table_res, tracker, cls):
        print("Latex Paper Summary")
        metric_names = ['map', 'precision', 'recall']
        for metric in metric_names:
            header = list() 
            header = PosemAP.get_pose_parts()
            self._row_print_latex([f'{metric}'] + header)

            output = ['']
            metric_results = table_res['COMBINED_SEQ'][metric]
            for name, val in metric_results.items():
                output.append("%.1f" % val)

            self._row_print_latex(output)

    def print_table(self, table_res, tracker, cls):
        """Prints table of results for all sequences"""
        print('')
        metric_name = self.get_name()
        self._row_print([metric_name + '->evaluating: ' + tracker + ':'])
        seq_names = list(table_res.keys())
        metric_names = ['map', 'precision', 'recall']
        for metric_name in metric_names:
            header_names = list(table_res[seq_names[0]][metric_name].keys())
            self._row_print([f'[{metric_name}]'] + header_names, space=15)
            for seq, results in sorted(table_res.items()):
                if seq == 'COMBINED_SEQ':
                    continue
                seq_results = []
                for metric, metric_results in results[metric_name].items():
                    seq_results.append('%.1f' % metric_results)
                self._row_print([seq] + seq_results, space=15)

            self._row_print(['Summary ' + tracker + ':'])
            seq_results = []
            for metric, metric_results in table_res['COMBINED_SEQ'][metric_name].items():
                seq_results.append('%.1f' % metric_results)
            self._row_print([''] + seq_results, space=15)
            print("")

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
        to_print = PosemAP._row_to_latex(*argv)
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
        metric_names = ['map', 'precision', 'recall']
        rows = list()
        for metric in metric_names:
            header = list() 
            header = PosemAP.get_pose_parts()
            header_row = self._row_to_latex([f'{metric}'] + header)
            rows.append(header_row)

            output = ['']
            metric_results = table_res['COMBINED_SEQ'][metric]
            for name, val in metric_results.items():
                output.append("%.1f" % val)

            rows.append(self._row_to_latex(output))

        return rows

    def detailed_results(self, table_res):
        import pdb; pdb.set_trace()
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
