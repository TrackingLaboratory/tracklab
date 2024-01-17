import os
import csv
import configparser
import numpy as np
from scipy.optimize import linear_sum_assignment
from trackeval.datasets._base_dataset import _BaseDataset

# from ._base_dataset import _BaseDataset
from trackeval import utils
from trackeval import _timing
from trackeval.utils import TrackEvalException


class SoccerNet2DBox(_BaseDataset):
    """Dataset class for SoccerNet Challenge 2D bounding box tracking"""

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            'GT_FOLDER': os.path.join(code_path, 'data/gt/mot_challenge/'),  # Location of GT data
            'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/mot_challenge/'),  # Trackers location
            'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
            'CLASSES_TO_EVAL': ['pedestrian'],  # Valid: ['pedestrian']
            'BENCHMARK': 'MOT17',  # Valid: 'MOT17', 'MOT16', 'MOT20', 'MOT15'
            'SPLIT_TO_EVAL': 'train',  # Valid: 'train', 'test', 'all'
            'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
            'PRINT_CONFIG': True,  # Whether to print current config
            'DO_PREPROC': True,  # Whether to perform preprocessing (never done for MOT15)
            'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
            'SEQMAP_FOLDER': None,  # Where seqmaps are found (if None, GT_FOLDER/seqmaps)
            'SEQMAP_FILE': None,  # Directly specify seqmap file (if none use seqmap_folder/benchmark-split_to_eval)
            'SEQ_INFO': None,  # If not None, directly specify sequences to eval and their number of timesteps
            'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',  # '{gt_folder}/{seq}/gt/gt.txt'
            'SKIP_SPLIT_FOL': False,  # If False, data is in GT_FOLDER/BENCHMARK-SPLIT_TO_EVAL/ and in
                                      # TRACKERS_FOLDER/BENCHMARK-SPLIT_TO_EVAL/tracker/
                                      # If True, then the middle 'benchmark-split' folder is skipped for both.
        }
        return default_config

    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()
        # Fill non-given config values with defaults
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())

        self.benchmark = self.config['BENCHMARK']
        gt_set = self.config['BENCHMARK'] + '-' + self.config['SPLIT_TO_EVAL']
        self.gt_set = gt_set
        if not self.config['SKIP_SPLIT_FOL']:
            split_fol = gt_set
        else:
            split_fol = ''
        self.gt_fol = os.path.join(self.config['GT_FOLDER'], split_fol)
        self.tracker_fol = os.path.join(self.config['TRACKERS_FOLDER'], split_fol)
        self.should_classes_combine = True
        self.use_super_categories = True  # Create supercategories for players, referee, goalkeeper, ball, other
        self.data_is_zipped = self.config['INPUT_AS_ZIP']
        self.do_preproc = self.config['DO_PREPROC']

        self.output_fol = self.config['OUTPUT_FOLDER']
        if self.output_fol is None:
            self.output_fol = self.tracker_fol

        self.tracker_sub_fol = self.config['TRACKER_SUB_FOLDER']
        self.output_sub_fol = self.config['OUTPUT_SUB_FOLDER']

        # Get classes to eval
        self.valid_classes = ['pedestrian']
        self.class_list = [cls.lower() if cls.lower() in self.valid_classes else None
                           for cls in self.config['CLASSES_TO_EVAL']]
        if not all(self.class_list):
            raise TrackEvalException('Attempted to evaluate an invalid class. Only pedestrian class is valid.')
        self.class_name_to_class_id = {'pedestrian': 1, 'person_on_vehicle': 2, 'car': 3, 'bicycle': 4, 'motorbike': 5,
                                       'non_mot_vehicle': 6, 'static_person': 7, 'distractor': 8, 'occluder': 9,
                                       'occluder_on_ground': 10, 'occluder_full': 11, 'reflection': 12, 'crowd': 13}
        self.valid_class_numbers = list(self.class_name_to_class_id.values())

        # Get sequences to eval and check gt files exist
        self.seq_list, self.seq_lengths = self._get_seq_info()
        if len(self.seq_list) < 1:
            raise TrackEvalException('No sequences are selected to be evaluated.')

        # Check gt files exist
        for seq in self.seq_list:
            if not self.data_is_zipped:
                curr_file = self.config["GT_LOC_FORMAT"].format(gt_folder=self.gt_fol, seq=seq)
                if not os.path.isfile(curr_file):
                    print('GT file not found ' + curr_file)
                    raise TrackEvalException('GT file not found for sequence: ' + seq)
        if self.data_is_zipped:
            curr_file = os.path.join(self.gt_fol, 'data.zip')
            if not os.path.isfile(curr_file):
                print('GT file not found ' + curr_file)
                raise TrackEvalException('GT file not found: ' + os.path.basename(curr_file))

        # Get trackers to eval
        if self.config['TRACKERS_TO_EVAL'] is None:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            self.tracker_list = self.config['TRACKERS_TO_EVAL']

        if self.config['TRACKER_DISPLAY_NAMES'] is None:
            self.tracker_to_disp = dict(zip(self.tracker_list, self.tracker_list))
        elif (self.config['TRACKERS_TO_EVAL'] is not None) and (
                len(self.config['TRACKER_DISPLAY_NAMES']) == len(self.tracker_list)):
            self.tracker_to_disp = dict(zip(self.tracker_list, self.config['TRACKER_DISPLAY_NAMES']))
        else:
            raise TrackEvalException('List of tracker files and tracker display names do not match.')

        for tracker in self.tracker_list:
            if self.data_is_zipped:
                curr_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol + '.zip')
                if not os.path.isfile(curr_file):
                    print('Tracker file not found: ' + curr_file)
                    raise TrackEvalException('Tracker file not found: ' + tracker + '/' + os.path.basename(curr_file))
            else:
                for seq in self.seq_list:
                    curr_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '.txt')
                    if not os.path.isfile(curr_file):
                        print('Tracker file not found: ' + curr_file)
                        raise TrackEvalException(
                            'Tracker file not found: ' + tracker + '/' + self.tracker_sub_fol + '/' + os.path.basename(
                                curr_file))

    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def _get_seq_info(self):
        seq_list = []
        seq_lengths = {}
        if self.config["SEQ_INFO"]:
            seq_list = list(self.config["SEQ_INFO"].keys())
            seq_lengths = self.config["SEQ_INFO"]

            # If sequence length is 'None' tries to read sequence length from .ini files.
            for seq, seq_length in seq_lengths.items():
                if seq_length is None:
                    ini_file = os.path.join(self.gt_fol, seq, 'seqinfo.ini')
                    if not os.path.isfile(ini_file):
                        raise TrackEvalException('ini file does not exist: ' + seq + '/' + os.path.basename(ini_file))
                    ini_data = configparser.ConfigParser()
                    ini_data.read(ini_file)
                    seq_lengths[seq] = int(ini_data['Sequence']['seqLength'])

        else:
            if self.config["SEQMAP_FILE"]:
                seqmap_file = self.config["SEQMAP_FILE"]
            else:
                if self.config["SEQMAP_FOLDER"] is None:
                    seqmap_file = os.path.join(self.config['GT_FOLDER'], 'seqmaps', self.gt_set + '.txt')
                else:
                    seqmap_file = os.path.join(self.config["SEQMAP_FOLDER"], self.gt_set + '.txt')
            if not os.path.isfile(seqmap_file):
                print('no seqmap found: ' + seqmap_file)
                raise TrackEvalException('no seqmap found: ' + os.path.basename(seqmap_file))
            with open(seqmap_file) as fp:
                reader = csv.reader(fp)
                for i, row in enumerate(reader):
                    if i == 0 or row[0] == '':
                        continue
                    seq = row[0]
                    seq_list.append(seq)
                    ini_file = os.path.join(self.gt_fol, seq, 'seqinfo.ini')
                    if not os.path.isfile(ini_file):
                        raise TrackEvalException('ini file does not exist: ' + seq + '/' + os.path.basename(ini_file))
                    ini_data = configparser.ConfigParser()
                    ini_data.read(ini_file)
                    seq_lengths[seq] = int(ini_data['Sequence']['seqLength'])
        return seq_list, seq_lengths

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
        if self.data_is_zipped:
            if is_gt:
                zip_file = os.path.join(self.gt_fol, 'data.zip')
            else:
                zip_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol + '.zip')
            file = seq + '.txt'
        else:
            zip_file = None
            if is_gt:
                file = self.config["GT_LOC_FORMAT"].format(gt_folder=self.gt_fol, seq=seq)
            else:
                file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '.txt')

        # Load raw data from text file
        read_data, ignore_data = self._load_simple_text_file(file, is_zipped=self.data_is_zipped, zip_file=zip_file)

        # Convert data to required format
        num_timesteps = self.seq_lengths[seq]
        data_keys = ['ids', 'classes', 'dets']
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
                    time_data = np.asarray(read_data[time_key], dtype=float)
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

        BDD100K:
            In BDD100K, the 4 preproc steps are as follow:
                1) There are eight classes (pedestrian, rider, car, bus, truck, train, motorcycle, bicycle)
                    which are evaluated separately.
                2) For BDD100K there is no removal of matched tracker dets.
                3) Crowd ignore regions are used to remove unmatched detections.
                4) No removal of gt dets.
        """
        cls_id = self.class_name_to_class_id[cls]

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0
        for t in range(raw_data['num_timesteps']):

            # Only extract relevant dets for this class for preproc and eval (cls)
            gt_class_mask = np.atleast_1d(raw_data['gt_classes'][t] == cls_id)
            gt_class_mask = gt_class_mask.astype(bool)
            gt_ids = raw_data['gt_ids'][t][gt_class_mask]
            gt_dets = raw_data['gt_dets'][t][gt_class_mask]

            tracker_class_mask = np.atleast_1d(raw_data['tracker_classes'][t] == cls_id)
            tracker_class_mask = tracker_class_mask.astype(bool)
            tracker_ids = raw_data['tracker_ids'][t][tracker_class_mask]
            tracker_dets = raw_data['tracker_dets'][t][tracker_class_mask]
            similarity_scores = raw_data['similarity_scores'][t][gt_class_mask, :][:, tracker_class_mask]

            # Match tracker and gt dets (with hungarian algorithm)
            unmatched_indices = np.arange(tracker_ids.shape[0])
            if gt_ids.shape[0] > 0 and tracker_ids.shape[0] > 0:
                matching_scores = similarity_scores.copy()
                matching_scores[matching_scores < 0.5 - np.finfo('float').eps] = 0
                match_rows, match_cols = linear_sum_assignment(-matching_scores)
                actually_matched_mask = matching_scores[match_rows, match_cols] > 0 + np.finfo('float').eps
                match_cols = match_cols[actually_matched_mask]
                unmatched_indices = np.delete(unmatched_indices, match_cols, axis=0)

            # For unmatched tracker dets, remove those that are greater than 50% within a crowd ignore region.
            unmatched_tracker_dets = tracker_dets[unmatched_indices, :]
            crowd_ignore_regions = raw_data['gt_crowd_ignore_regions'][t]
            intersection_with_ignore_region = self._calculate_box_ious(unmatched_tracker_dets, crowd_ignore_regions,
                                                                       box_format='x0y0x1y1', do_ioa=True)
            is_within_crowd_ignore_region = np.any(intersection_with_ignore_region > 0.5 + np.finfo('float').eps,
                                                   axis=1)

            # Apply preprocessing to remove unwanted tracker dets.
            to_remove_tracker = unmatched_indices[is_within_crowd_ignore_region]
            data['tracker_ids'][t] = np.delete(tracker_ids, to_remove_tracker, axis=0)
            data['tracker_dets'][t] = np.delete(tracker_dets, to_remove_tracker, axis=0)
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
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(int)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(int)

        # Record overview statistics.
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_timesteps'] = raw_data['num_timesteps']

        # Ensure that ids are unique per timestep.
        self._check_unique_ids(data)

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_box_ious(gt_dets_t, tracker_dets_t, box_format='xywh')
        return similarity_scores

train_classes = [
  {
    "id": 1,
    "name": "ball_1",
    "supercategory": "person"
  },
  {
    "id": 2,
    "name": "ball_2",
    "supercategory": "person"
  },
  {
    "id": 3,
    "name": "ball_none",
    "supercategory": "person"
  },
  {
    "id": 4,
    "name": "goalkeeper_left",
    "supercategory": "person"
  },
  {
    "id": 5,
    "name": "goalkeeper_left_1",
    "supercategory": "person"
  },
  {
    "id": 6,
    "name": "goalkeeper_left_18",
    "supercategory": "person"
  },
  {
    "id": 7,
    "name": "goalkeeper_right",
    "supercategory": "person"
  },
  {
    "id": 8,
    "name": "goalkeeper_right_1",
    "supercategory": "person"
  },
  {
    "id": 9,
    "name": "goalkeeper_right_18",
    "supercategory": "person"
  },
  {
    "id": 10,
    "name": "goalkeeper_right_2",
    "supercategory": "person"
  },
  {
    "id": 11,
    "name": "other_1",
    "supercategory": "person"
  },
  {
    "id": 12,
    "name": "other_2",
    "supercategory": "person"
  },
  {
    "id": 13,
    "name": "other_3",
    "supercategory": "person"
  },
  {
    "id": 14,
    "name": "player_left",
    "supercategory": "person"
  },
  {
    "id": 15,
    "name": "player_left_10",
    "supercategory": "person"
  },
  {
    "id": 16,
    "name": "player_left_11",
    "supercategory": "person"
  },
  {
    "id": 17,
    "name": "player_left_13",
    "supercategory": "person"
  },
  {
    "id": 18,
    "name": "player_left_14",
    "supercategory": "person"
  },
  {
    "id": 19,
    "name": "player_left_15",
    "supercategory": "person"
  },
  {
    "id": 20,
    "name": "player_left_16",
    "supercategory": "person"
  },
  {
    "id": 21,
    "name": "player_left_17",
    "supercategory": "person"
  },
  {
    "id": 22,
    "name": "player_left_20",
    "supercategory": "person"
  },
  {
    "id": 23,
    "name": "player_left_21",
    "supercategory": "person"
  },
  {
    "id": 24,
    "name": "player_left_22",
    "supercategory": "person"
  },
  {
    "id": 25,
    "name": "player_left_23",
    "supercategory": "person"
  },
  {
    "id": 26,
    "name": "player_left_24",
    "supercategory": "person"
  },
  {
    "id": 27,
    "name": "player_left_25",
    "supercategory": "person"
  },
  {
    "id": 28,
    "name": "player_left_26",
    "supercategory": "person"
  },
  {
    "id": 29,
    "name": "player_left_27",
    "supercategory": "person"
  },
  {
    "id": 30,
    "name": "player_left_28",
    "supercategory": "person"
  },
  {
    "id": 31,
    "name": "player_left_29",
    "supercategory": "person"
  },
  {
    "id": 32,
    "name": "player_left_3",
    "supercategory": "person"
  },
  {
    "id": 33,
    "name": "player_left_30",
    "supercategory": "person"
  },
  {
    "id": 34,
    "name": "player_left_31",
    "supercategory": "person"
  },
  {
    "id": 35,
    "name": "player_left_32",
    "supercategory": "person"
  },
  {
    "id": 36,
    "name": "player_left_33",
    "supercategory": "person"
  },
  {
    "id": 37,
    "name": "player_left_34",
    "supercategory": "person"
  },
  {
    "id": 38,
    "name": "player_left_35",
    "supercategory": "person"
  },
  {
    "id": 39,
    "name": "player_left_36",
    "supercategory": "person"
  },
  {
    "id": 40,
    "name": "player_left_4",
    "supercategory": "person"
  },
  {
    "id": 41,
    "name": "player_left_40",
    "supercategory": "person"
  },
  {
    "id": 42,
    "name": "player_left_44",
    "supercategory": "person"
  },
  {
    "id": 43,
    "name": "player_left_5",
    "supercategory": "person"
  },
  {
    "id": 44,
    "name": "player_left_50",
    "supercategory": "person"
  },
  {
    "id": 45,
    "name": "player_left_55",
    "supercategory": "person"
  },
  {
    "id": 46,
    "name": "player_left_56",
    "supercategory": "person"
  },
  {
    "id": 47,
    "name": "player_left_6",
    "supercategory": "person"
  },
  {
    "id": 48,
    "name": "player_left_62",
    "supercategory": "person"
  },
  {
    "id": 49,
    "name": "player_left_7",
    "supercategory": "person"
  },
  {
    "id": 50,
    "name": "player_left_8",
    "supercategory": "person"
  },
  {
    "id": 51,
    "name": "player_left_9",
    "supercategory": "person"
  },
  {
    "id": 52,
    "name": "player_left_93",
    "supercategory": "person"
  },
  {
    "id": 53,
    "name": "player_right",
    "supercategory": "person"
  },
  {
    "id": 54,
    "name": "player_right_10",
    "supercategory": "person"
  },
  {
    "id": 55,
    "name": "player_right_11",
    "supercategory": "person"
  },
  {
    "id": 56,
    "name": "player_right_14",
    "supercategory": "person"
  },
  {
    "id": 57,
    "name": "player_right_15",
    "supercategory": "person"
  },
  {
    "id": 58,
    "name": "player_right_16",
    "supercategory": "person"
  },
  {
    "id": 59,
    "name": "player_right_17",
    "supercategory": "person"
  },
  {
    "id": 60,
    "name": "player_right_19",
    "supercategory": "person"
  },
  {
    "id": 61,
    "name": "player_right_20",
    "supercategory": "person"
  },
  {
    "id": 62,
    "name": "player_right_21",
    "supercategory": "person"
  },
  {
    "id": 63,
    "name": "player_right_22",
    "supercategory": "person"
  },
  {
    "id": 64,
    "name": "player_right_23",
    "supercategory": "person"
  },
  {
    "id": 65,
    "name": "player_right_24",
    "supercategory": "person"
  },
  {
    "id": 66,
    "name": "player_right_25",
    "supercategory": "person"
  },
  {
    "id": 67,
    "name": "player_right_26",
    "supercategory": "person"
  },
  {
    "id": 68,
    "name": "player_right_27",
    "supercategory": "person"
  },
  {
    "id": 69,
    "name": "player_right_28",
    "supercategory": "person"
  },
  {
    "id": 70,
    "name": "player_right_29",
    "supercategory": "person"
  },
  {
    "id": 71,
    "name": "player_right_3",
    "supercategory": "person"
  },
  {
    "id": 72,
    "name": "player_right_30",
    "supercategory": "person"
  },
  {
    "id": 73,
    "name": "player_right_31",
    "supercategory": "person"
  },
  {
    "id": 74,
    "name": "player_right_33",
    "supercategory": "person"
  },
  {
    "id": 75,
    "name": "player_right_34",
    "supercategory": "person"
  },
  {
    "id": 76,
    "name": "player_right_35",
    "supercategory": "person"
  },
  {
    "id": 77,
    "name": "player_right_36",
    "supercategory": "person"
  },
  {
    "id": 78,
    "name": "player_right_38",
    "supercategory": "person"
  },
  {
    "id": 79,
    "name": "player_right_4",
    "supercategory": "person"
  },
  {
    "id": 80,
    "name": "player_right_40",
    "supercategory": "person"
  },
  {
    "id": 81,
    "name": "player_right_44",
    "supercategory": "person"
  },
  {
    "id": 82,
    "name": "player_right_5",
    "supercategory": "person"
  },
  {
    "id": 83,
    "name": "player_right_50",
    "supercategory": "person"
  },
  {
    "id": 84,
    "name": "player_right_55",
    "supercategory": "person"
  },
  {
    "id": 85,
    "name": "player_right_6",
    "supercategory": "person"
  },
  {
    "id": 86,
    "name": "player_right_62",
    "supercategory": "person"
  },
  {
    "id": 87,
    "name": "player_right_7",
    "supercategory": "person"
  },
  {
    "id": 88,
    "name": "player_right_75",
    "supercategory": "person"
  },
  {
    "id": 89,
    "name": "player_right_8",
    "supercategory": "person"
  },
  {
    "id": 90,
    "name": "player_right_9",
    "supercategory": "person"
  },
  {
    "id": 91,
    "name": "referee_main",
    "supercategory": "person"
  },
  {
    "id": 92,
    "name": "referee_side_bottom",
    "supercategory": "person"
  },
  {
    "id": 93,
    "name": "referee_side_top",
    "supercategory": "person"
  }
]

val_classes = [
  {
    "id": 1,
    "name": "ball_1",
    "supercategory": "person"
  },
  {
    "id": 2,
    "name": "ball_2",
    "supercategory": "person"
  },
  {
    "id": 3,
    "name": "ball_3",
    "supercategory": "person"
  },
  {
    "id": 4,
    "name": "ball_none",
    "supercategory": "person"
  },
  {
    "id": 5,
    "name": "goalkeeper_left",
    "supercategory": "person"
  },
  {
    "id": 6,
    "name": "goalkeeper_left_1",
    "supercategory": "person"
  },
  {
    "id": 7,
    "name": "goalkeeper_left_25",
    "supercategory": "person"
  },
  {
    "id": 8,
    "name": "goalkeeper_left_30",
    "supercategory": "person"
  },
  {
    "id": 9,
    "name": "goalkeeper_left_32",
    "supercategory": "person"
  },
  {
    "id": 10,
    "name": "goalkeeper_right",
    "supercategory": "person"
  },
  {
    "id": 11,
    "name": "goalkeeper_right_1",
    "supercategory": "person"
  },
  {
    "id": 12,
    "name": "goalkeeper_right_18",
    "supercategory": "person"
  },
  {
    "id": 13,
    "name": "goalkeeper_right_30",
    "supercategory": "person"
  },
  {
    "id": 14,
    "name": "goalkeeper_right_32",
    "supercategory": "person"
  },
  {
    "id": 15,
    "name": "other_1",
    "supercategory": "person"
  },
  {
    "id": 16,
    "name": "other_2",
    "supercategory": "person"
  },
  {
    "id": 17,
    "name": "other_3",
    "supercategory": "person"
  },
  {
    "id": 18,
    "name": "other_4",
    "supercategory": "person"
  },
  {
    "id": 19,
    "name": "player_left",
    "supercategory": "person"
  },
  {
    "id": 20,
    "name": "player_left_10",
    "supercategory": "person"
  },
  {
    "id": 21,
    "name": "player_left_11",
    "supercategory": "person"
  },
  {
    "id": 22,
    "name": "player_left_12",
    "supercategory": "person"
  },
  {
    "id": 23,
    "name": "player_left_14",
    "supercategory": "person"
  },
  {
    "id": 24,
    "name": "player_left_15",
    "supercategory": "person"
  },
  {
    "id": 25,
    "name": "player_left_16",
    "supercategory": "person"
  },
  {
    "id": 26,
    "name": "player_left_17",
    "supercategory": "person"
  },
  {
    "id": 27,
    "name": "player_left_2",
    "supercategory": "person"
  },
  {
    "id": 28,
    "name": "player_left_20",
    "supercategory": "person"
  },
  {
    "id": 29,
    "name": "player_left_22",
    "supercategory": "person"
  },
  {
    "id": 30,
    "name": "player_left_23",
    "supercategory": "person"
  },
  {
    "id": 31,
    "name": "player_left_24",
    "supercategory": "person"
  },
  {
    "id": 32,
    "name": "player_left_25",
    "supercategory": "person"
  },
  {
    "id": 33,
    "name": "player_left_26",
    "supercategory": "person"
  },
  {
    "id": 34,
    "name": "player_left_27",
    "supercategory": "person"
  },
  {
    "id": 35,
    "name": "player_left_3",
    "supercategory": "person"
  },
  {
    "id": 36,
    "name": "player_left_30",
    "supercategory": "person"
  },
  {
    "id": 37,
    "name": "player_left_31",
    "supercategory": "person"
  },
  {
    "id": 38,
    "name": "player_left_33",
    "supercategory": "person"
  },
  {
    "id": 39,
    "name": "player_left_34",
    "supercategory": "person"
  },
  {
    "id": 40,
    "name": "player_left_36",
    "supercategory": "person"
  },
  {
    "id": 41,
    "name": "player_left_4",
    "supercategory": "person"
  },
  {
    "id": 42,
    "name": "player_left_43",
    "supercategory": "person"
  },
  {
    "id": 43,
    "name": "player_left_44",
    "supercategory": "person"
  },
  {
    "id": 44,
    "name": "player_left_5",
    "supercategory": "person"
  },
  {
    "id": 45,
    "name": "player_left_50",
    "supercategory": "person"
  },
  {
    "id": 46,
    "name": "player_left_55",
    "supercategory": "person"
  },
  {
    "id": 47,
    "name": "player_left_56",
    "supercategory": "person"
  },
  {
    "id": 48,
    "name": "player_left_59",
    "supercategory": "person"
  },
  {
    "id": 49,
    "name": "player_left_60",
    "supercategory": "person"
  },
  {
    "id": 50,
    "name": "player_left_62",
    "supercategory": "person"
  },
  {
    "id": 51,
    "name": "player_left_69",
    "supercategory": "person"
  },
  {
    "id": 52,
    "name": "player_left_7",
    "supercategory": "person"
  },
  {
    "id": 53,
    "name": "player_left_8",
    "supercategory": "person"
  },
  {
    "id": 54,
    "name": "player_left_9",
    "supercategory": "person"
  },
  {
    "id": 55,
    "name": "player_left_93",
    "supercategory": "person"
  },
  {
    "id": 56,
    "name": "player_left_99",
    "supercategory": "person"
  },
  {
    "id": 57,
    "name": "player_right",
    "supercategory": "person"
  },
  {
    "id": 58,
    "name": "player_right_10",
    "supercategory": "person"
  },
  {
    "id": 59,
    "name": "player_right_11",
    "supercategory": "person"
  },
  {
    "id": 60,
    "name": "player_right_14",
    "supercategory": "person"
  },
  {
    "id": 61,
    "name": "player_right_15",
    "supercategory": "person"
  },
  {
    "id": 62,
    "name": "player_right_16",
    "supercategory": "person"
  },
  {
    "id": 63,
    "name": "player_right_17",
    "supercategory": "person"
  },
  {
    "id": 64,
    "name": "player_right_2",
    "supercategory": "person"
  },
  {
    "id": 65,
    "name": "player_right_20",
    "supercategory": "person"
  },
  {
    "id": 66,
    "name": "player_right_21",
    "supercategory": "person"
  },
  {
    "id": 67,
    "name": "player_right_22",
    "supercategory": "person"
  },
  {
    "id": 68,
    "name": "player_right_23",
    "supercategory": "person"
  },
  {
    "id": 69,
    "name": "player_right_24",
    "supercategory": "person"
  },
  {
    "id": 70,
    "name": "player_right_25",
    "supercategory": "person"
  },
  {
    "id": 71,
    "name": "player_right_26",
    "supercategory": "person"
  },
  {
    "id": 72,
    "name": "player_right_27",
    "supercategory": "person"
  },
  {
    "id": 73,
    "name": "player_right_29",
    "supercategory": "person"
  },
  {
    "id": 74,
    "name": "player_right_3",
    "supercategory": "person"
  },
  {
    "id": 75,
    "name": "player_right_30",
    "supercategory": "person"
  },
  {
    "id": 76,
    "name": "player_right_31",
    "supercategory": "person"
  },
  {
    "id": 77,
    "name": "player_right_33",
    "supercategory": "person"
  },
  {
    "id": 78,
    "name": "player_right_34",
    "supercategory": "person"
  },
  {
    "id": 79,
    "name": "player_right_36",
    "supercategory": "person"
  },
  {
    "id": 80,
    "name": "player_right_38",
    "supercategory": "person"
  },
  {
    "id": 81,
    "name": "player_right_4",
    "supercategory": "person"
  },
  {
    "id": 82,
    "name": "player_right_43",
    "supercategory": "person"
  },
  {
    "id": 83,
    "name": "player_right_44",
    "supercategory": "person"
  },
  {
    "id": 84,
    "name": "player_right_45",
    "supercategory": "person"
  },
  {
    "id": 85,
    "name": "player_right_5",
    "supercategory": "person"
  },
  {
    "id": 86,
    "name": "player_right_50",
    "supercategory": "person"
  },
  {
    "id": 87,
    "name": "player_right_53",
    "supercategory": "person"
  },
  {
    "id": 88,
    "name": "player_right_55",
    "supercategory": "person"
  },
  {
    "id": 89,
    "name": "player_right_56",
    "supercategory": "person"
  },
  {
    "id": 90,
    "name": "player_right_59",
    "supercategory": "person"
  },
  {
    "id": 91,
    "name": "player_right_6",
    "supercategory": "person"
  },
  {
    "id": 92,
    "name": "player_right_62",
    "supercategory": "person"
  },
  {
    "id": 93,
    "name": "player_right_7",
    "supercategory": "person"
  },
  {
    "id": 94,
    "name": "player_right_76",
    "supercategory": "person"
  },
  {
    "id": 95,
    "name": "player_right_78",
    "supercategory": "person"
  },
  {
    "id": 96,
    "name": "player_right_8",
    "supercategory": "person"
  },
  {
    "id": 97,
    "name": "player_right_9",
    "supercategory": "person"
  },
  {
    "id": 98,
    "name": "player_right_93",
    "supercategory": "person"
  },
  {
    "id": 99,
    "name": "referee_main",
    "supercategory": "person"
  },
  {
    "id": 100,
    "name": "referee_side_bottom",
    "supercategory": "person"
  },
  {
    "id": 101,
    "name": "referee_side_top",
    "supercategory": "person"
  }
]