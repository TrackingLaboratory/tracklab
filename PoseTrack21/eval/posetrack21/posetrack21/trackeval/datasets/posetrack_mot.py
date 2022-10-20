import os
import csv
import configparser
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_dataset import _BaseDataset
from .. import utils
from .. import _timing
from ..utils import TrackEvalException
from shapely.geometry import box, Polygon, MultiPolygon
import json 

class PoseTrackMOT(_BaseDataset):
    """Dataset class for MOT Challenge 2D bounding box tracking"""

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            'GT_FOLDER': os.path.join(code_path, 'data/gt/PoseTrackReID/posetrack_data/mot/val/'),  # Location of GT data
            'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/mot/'),  # Trackers location
            'PRINT_CONFIG': True,
            'ASSURE_SINGLE_TRACKER': False
        }
        return default_config

    def get_output_fol(self, tracker):
        return os.path.join(self.output_fol, self.tracker_to_disp[tracker], self.output_sub_fol)

    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()

        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())

        self.gt_fol = self.config['GT_FOLDER']
        self.tracker_fol = self.config['TRACKERS_FOLDER']
        self.output_sub_fol = ''
        self.should_classes_combine = False
        self.use_super_categories = False

        self.output_fol = self.tracker_fol

        # Get classes to eval
        self.valid_classes = ['pedestrian']
        self.class_list = [cls for cls in self.valid_classes]
        self.class_name_to_class_id = {'pedestrian': -1}
        self.valid_class_numbers = list(self.class_name_to_class_id.values())

        # Get sequences to eval and check gt files exist
        self.seq_list, self.seq_lengths, self.seq_ignore_regions = self._get_seq_info()
        if len(self.seq_list) < 1:
            raise TrackEvalException('No sequences are selected to be evaluated.')

        # Check gt files exist
        for seq in self.seq_list:
            curr_file = os.path.join(self.gt_fol, seq, 'gt/gt.txt')
            if not os.path.isfile(curr_file):
                print('GT file not found ' + curr_file)
                raise TrackEvalException('GT file not found for sequence: ' + seq)

        # Get trackers to eval
        self.tracker_list = os.listdir(self.tracker_fol)

        # check,  if tracker fol contains results 
        contains_all_files = True
        for track_file in self.tracker_list:
            if track_file not in self.seq_list:
                tmp_file = track_file[:-4] 
                if tmp_file not in self.seq_list:
                    contains_all_files = False

        if contains_all_files: 
            self.tracker_list = []
            self.tracker_to_disp = dict()

            tracker_name = os.path.basename(self.tracker_fol)
            self.tracker_list.append('')
            self.tracker_to_disp[''] = tracker_name

        else:
            # tracker folder does not contain result files -> check folders
            self.tracker_list = [] 
            self.tracker_to_disp = dict() 
            for folder in os.listdir(self.tracker_fol):
                folder_path = os.path.join(self.tracker_fol, folder)
                if not os.path.isdir(folder_path):
                    continue

                self.tracker_list.append(folder) 
                self.tracker_to_disp[folder] = folder
            
        keep_tracker_list = list()
        for tracker in self.tracker_list:
            ignore_file = os.path.join(self.tracker_fol, tracker, '.mot_ignore')
            if os.path.isfile(ignore_file):
                continue
            for seq in self.seq_list:
                #curr_file = os.path.join(self.tracker_fol, tracker, seq + '.txt')
                curr_file = os.path.join(self.tracker_fol, tracker, seq)
                if not os.path.isfile(curr_file):
                    curr_file = f'{curr_file}.txt'
                if not os.path.isfile(curr_file):
                    import pdb; pdb.set_trace()
                    print('Tracker file not found: ' + curr_file)
                    raise TrackEvalException(
                        'Tracker file not found: ' + tracker + '/' + os.path.basename(
                            curr_file))
            keep_tracker_list.append(tracker)
        self.tracker_list = keep_tracker_list

    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def _get_seq_info(self):
        seq_list = os.listdir(self.gt_fol)
        seq_ignore_regions = dict()
        seq_lengths = dict()

        for seq in seq_list:
            config_file = os.path.join(self.gt_fol, seq, 'image_info.json')
            if not os.path.isfile(config_file):
                raise TrackEvalException('image info file does not exist: ' + seq + '/image_info.json')
            with open(config_file, 'r') as file:
                image_info = json.load(file)

            ignore_regions = {}
            for image_data in image_info:
                frame_index = str(image_data['frame_index'])

                image_ignore_regions = [] 
                if len(image_data['ignore_regions_x']) > 0 and len(image_data['ignore_regions_y']):
                    ignore_x = image_data['ignore_regions_x']
                    ignore_y = image_data['ignore_regions_y']

                    if not isinstance(ignore_x[0], list):
                        ignore_x = [ignore_x]
                        ignore_y = [ignore_y]

                    for r_idx in range(len(ignore_x)):
                        region = []

                        for x, y in zip(ignore_x[r_idx], ignore_y[r_idx]):
                            region.append([x, y])
                        ignore_region = Polygon(region)
                        image_ignore_regions.append(ignore_region)

                ignore_regions[frame_index] = image_ignore_regions

            seq_ignore_regions[seq] = ignore_regions

            seq_lengths[seq] =  len(image_info)

        return seq_list, seq_lengths, seq_ignore_regions

    def _load_raw_file(self, tracker, seq, is_gt):
        """Load a file (gt or tracker) in the MOT Challenge 2D box format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [gt_extras] : list (for each timestep) of dicts (for each extra) of 1D NDArrays (for each det).

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes, tracker_confidences] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        """
        # File location
        if is_gt:
            file = os.path.join(self.gt_fol, seq, 'gt/gt.txt')
        else:
            file = os.path.join(self.tracker_fol, tracker, seq + '.txt')
            if not os.path.exists(file):
                file = file[:-4]

        # Load raw data from text file
        read_data, ignore_data = self._load_simple_text_file(file, is_zipped=False)

        # Convert data to required format
        num_timesteps = self.seq_lengths[seq]
        data_keys = ['ids', 'classes', 'dets' ]
        if is_gt:
            data_keys += ['gt_crowd_ignore_regions', 'gt_extras']
        else:
            data_keys += ['tracker_confidences']
        raw_data = {key: [None] * num_timesteps for key in data_keys}

        # Check for any extra time keys
        current_time_keys = [str( t+ 1) for t in range(num_timesteps)]
        extra_time_keys = [x for x in read_data.keys() if x not in current_time_keys]
        if len(extra_time_keys) > 0:
            if is_gt:
                text = 'Ground-truth'
            else:
                text = 'Tracking'
            raise TrackEvalException(
                text + ' data contains the following invalid timesteps in seq %s: ' % seq + ', '.join(
                    [str(x) + ', ' for x in extra_time_keys]))

        for t in range(num_timesteps):
            time_key = str(t+1)
            if time_key in read_data.keys():
                try:
                    time_data = np.asarray(read_data[time_key], dtype=np.float)
                except ValueError:
                    if is_gt:
                        raise TrackEvalException(
                            'Cannot convert gt data for sequence %s to float. Is data corrupted?' % seq)
                    else:
                        raise TrackEvalException(
                            'Cannot convert tracking data from tracker %s, sequence %s to float. Is data corrupted?' % (
                                tracker, seq))
                try:
                    raw_data['dets'][t] = np.atleast_2d(time_data[:, 2:6])
                    raw_data['ids'][t] = np.atleast_1d(time_data[:, 1]).astype(int)
                except IndexError:
                    if is_gt:
                        err = 'Cannot load gt data from sequence %s, because there is not enough ' \
                              'columns in the data.' % seq
                        raise TrackEvalException(err)
                    else:
                        err = 'Cannot load tracker data from tracker %s, sequence %s, because there is not enough ' \
                              'columns in the data.' % (tracker, seq)
                        raise TrackEvalException(err)
                if time_data.shape[1] >= 8:
                    raw_data['classes'][t] = np.atleast_1d(time_data[:, 7]).astype(int)
                else:
                    if not is_gt:
                        raw_data['classes'][t] = np.ones_like(raw_data['ids'][t])
                    else:
                        raise TrackEvalException(
                            'GT data is not in a valid format, there is not enough rows in seq %s, timestep %i.' % (
                                seq, t))
                if is_gt:
                    gt_extras_dict = {'zero_marked': np.atleast_1d(time_data[:, 6].astype(int))}
                    raw_data['gt_extras'][t] = gt_extras_dict
                else:
                    raw_data['tracker_confidences'][t] = np.atleast_1d(time_data[:, 6])
            else:
                raw_data['dets'][t] = np.empty((0, 4))
                raw_data['ids'][t] = np.empty(0).astype(int)
                raw_data['classes'][t] = np.empty(0).astype(int)
                if is_gt:
                    gt_extras_dict = {'zero_marked': np.empty(0)}
                    raw_data['gt_extras'][t] = gt_extras_dict
                else:
                    raw_data['tracker_confidences'][t] = np.empty(0)
            if is_gt:
                raw_data['gt_crowd_ignore_regions'][t] = np.empty((0, 4))

        if is_gt:
            key_map = {'ids': 'gt_ids',
                       'classes': 'gt_classes',
                       'dets': 'gt_dets'}
        else:
            key_map = {'ids': 'tracker_ids',
                       'classes': 'tracker_classes',
                       'dets': 'tracker_dets'}
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)
        raw_data['num_timesteps'] = num_timesteps
        raw_data['seq'] = seq
        return raw_data

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        """ Preprocess data for a single sequence for a single class ready for evaluation.
        Inputs:
             - raw_data is a dict containing the data for the sequence already read in by get_raw_seq_data().
             - cls is the class to be evaluated.
        Outputs:
             - data is a dict containing all of the information that metrics need to perform evaluation.
                It contains the following fields:
                    [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets] : integers.
                    [gt_ids, tracker_ids, tracker_confidences]: list (for each timestep) of 1D NDArrays (for each det).
                    [gt_dets, tracker_dets]: list (for each timestep) of lists of detections.
                    [similarity_scores]: list (for each timestep) of 2D NDArrays.
        Notes:
            General preprocessing (preproc) occurs in 4 steps. Some datasets may not use all of these steps.
                1) Extract only detections relevant for the class to be evaluated (including distractor detections).
                2) Match gt dets and tracker dets. Remove tracker dets that are matched to a gt det that is of a
                    distractor class, or otherwise marked as to be removed.
                3) Remove unmatched tracker dets if they fall within a crowd ignore region or don't meet a certain
                    other criteria (e.g. are too small).
                4) Remove gt dets that were only useful for preprocessing and not for actual evaluation.
            After the above preprocessing steps, this function also calculates the number of gt and tracker detections
                and unique track ids. It also relabels gt and tracker ids to be contiguous and checks that ids are
                unique within each timestep.

        MOT Challenge:
            In MOT Challenge, the 4 preproc steps are as follow:
                1) There is only one class (pedestrian) to be evaluated, but all other classes are used for preproc.
                2) Predictions are matched against all gt boxes (regardless of class), those matching with distractor
                    objects are removed.
                3) There is no crowd ignore regions.
                4) All gt dets except pedestrian are removed, also removes pedestrian gt dets marked with zero_marked.
        """
        # Check that input data has unique ids
        self._check_unique_ids(raw_data)

        cls_id = self.class_name_to_class_id[cls]

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'tracker_confidences', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0
        for t in range(raw_data['num_timesteps']):

            # Get all data
            gt_ids = raw_data['gt_ids'][t]
            gt_dets = raw_data['gt_dets'][t]
            gt_classes = raw_data['gt_classes'][t]
            gt_zero_marked = raw_data['gt_extras'][t]['zero_marked']

            tracker_ids = raw_data['tracker_ids'][t]
            tracker_dets = raw_data['tracker_dets'][t]
            tracker_classes = raw_data['tracker_classes'][t]
            tracker_confidences = raw_data['tracker_confidences'][t]
            similarity_scores = raw_data['similarity_scores'][t]


            # Evaluation is ONLY valid for pedestrian class
            if len(tracker_classes) > 0 and np.max(tracker_classes) > 1:
                raise TrackEvalException(
                    'Evaluation is only valid for pedestrian class. Non pedestrian class (%i) found in sequence %s at '
                    'timestep %i.' % (np.max(tracker_classes), raw_data['seq'], t))

            # Match tracker and gt dets (with hungarian algorithm) and remove tracker dets which match with gt dets
            # which are labeled as belonging to a distractor class.
            to_remove_tracker = np.array([], np.int)
            if gt_ids.shape[0] > 0 and tracker_ids.shape[0] > 0:

                # Check all classes are valid:
                invalid_classes = np.setdiff1d(np.unique(gt_classes), self.valid_class_numbers)
                if len(invalid_classes) > 0:
                    print(' '.join([str(x) for x in invalid_classes]))
                    raise(TrackEvalException('Attempting to evaluate using invalid gt classes. '
                                             'This warning only triggers if preprocessing is performed, '
                                             'e.g. not for MOT15 or where prepropressing is explicitly disabled. '
                                             'Please either check your gt data, or disable preprocessing. '
                                             'The following invalid classes were found in timestep ' + str(t) + ': ' +
                                             ' '.join([str(x) for x in invalid_classes])))

              
                matching_scores = similarity_scores.copy()
                matching_scores[matching_scores < 0.5 - np.finfo('float').eps] = 0
                match_rows, match_cols = linear_sum_assignment(-matching_scores)
                actually_matched_mask = matching_scores[match_rows, match_cols] > 0 + np.finfo('float').eps
                match_rows = match_rows[actually_matched_mask]
                match_cols = match_cols[actually_matched_mask]

                # remove unmatched detections, if they intersect ignore regions 
                seq = raw_data['seq']
                ignore_regions = self.seq_ignore_regions[seq]
                fr_idx = list(ignore_regions.keys())[t]
                fr_ignore_regions = ignore_regions[fr_idx]

                if len(match_cols) < similarity_scores.shape[1]:
                    if len(fr_ignore_regions) > 0:
                        num_dets = len(tracker_dets)
                        unmatched_det_idxs = [i for i in range(num_dets) if i not in match_cols]
                        unmatched_boxes = []
                        for unmatched_idx in unmatched_det_idxs:
                            det = tracker_dets[unmatched_idx]
                            if len(det.shape)  == 2:
                                det = det[0]

                            x1 = det[0]
                            y1 = det[1]
                            x2 = x1 + det[2]
                            y2 = y1 + det[3]

                            box_poly = Polygon([
                                [x1, y1],
                                [x2, y1],
                                [x2, y2],
                                [x1, y2],
                                [x1, y1],
                            ])
                            unmatched_boxes.append(box_poly)
                        region_ious = np.zeros([len(fr_ignore_regions), len(unmatched_boxes)])

                        for i in range(len(fr_ignore_regions)):
                            for j in range(len(unmatched_boxes)):
                                if fr_ignore_regions[i].is_valid:
                                    poly_intersection = fr_ignore_regions[i].intersection(unmatched_boxes[j]).area
                                    poly_union = fr_ignore_regions[i].union(unmatched_boxes[j]).area
                                else:
                                    multi_poly = fr_ignore_regions[i].buffer(0)
                                    poly_intersection = 0
                                    poly_union = 0

                                    if isinstance(multi_poly, Polygon):
                                        poly_intersection = multi_poly.intersection(unmatched_boxes[j]).area
                                        poly_union = multi_poly.union(unmatched_boxes[j]).area
                                    else:
                                        multi_poly = fr_ignore_regions[i].buffer(0)
                                        poly_intersection = 0
                                        poly_union = 0

                                        if isinstance(multi_poly, Polygon):
                                            poly_intersection = multi_poly.intersection(unmatched_boxes[j]).area
                                            poly_union = multi_poly.union(unmatched_boxes[j]).area
                                        else:
                                            for poly in multi_poly:
                                                poly_intersection += poly.intersection(unmatched_boxes[j]).area
                                                poly_union += poly.union(unmatched_boxes[j]).area
                                                
                                region_ious[i, j] = poly_intersection / poly_union

                        if len(ignore_regions) > 0 and len(unmatched_boxes) > 0:
                            ignore_candidates = np.where((region_ious > 0.1))[1].tolist()

                        if len(ignore_candidates) > 0:
                            dets_to_remove = [] 
                            for remove_idx in ignore_candidates:
                                det_idx = unmatched_det_idxs[remove_idx]
                                dets_to_remove.append(det_idx)
                            
                            to_remove_tracker = np.array(dets_to_remove, dtype=np.int)
                  
            # Apply preprocessing to remove all unwanted tracker dets.
            data['tracker_ids'][t] = np.delete(tracker_ids, to_remove_tracker, axis=0)
            data['tracker_dets'][t] = np.delete(tracker_dets, to_remove_tracker, axis=0)
            data['tracker_confidences'][t] = np.delete(tracker_confidences, to_remove_tracker, axis=0)
            similarity_scores = np.delete(similarity_scores, to_remove_tracker, axis=1)
            
            data['gt_ids'][t] = gt_ids
            data['gt_dets'][t] = gt_dets
            data['similarity_scores'][t] = similarity_scores

            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
            num_tracker_dets += len(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])

        # Re-label IDs such that there are no empty IDs
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(np.int)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(np.int)

        # Record overview statistics.
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_timesteps'] = raw_data['num_timesteps']
        data['seq'] = raw_data['seq']

        # Ensure again that ids are unique per timestep after preproc.
        self._check_unique_ids(data, after_preproc=True)

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_box_ious(gt_dets_t, tracker_dets_t, box_format='xywh')
        return similarity_scores

