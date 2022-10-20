import os
import csv
import configparser
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_dataset import _BaseDataset
from .. import utils
from .. import _timing
from ..utils import TrackEvalException, count_valid_joints
import json
from shapely import geometry
from pathlib import Path


class PoseTrackReID(_BaseDataset):
    """Dataset class for MOT Challenge 2D bounding box tracking"""

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            'GT_FOLDER': os.path.join(code_path, 'data/gt/PoseTrackReid/posetrack_data/annotations/val/'),  # Location of GT data
            'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/PoseTrackReid/'),  # Trackers location
            'PRINT_CONFIG': False,  # Whether to print current config
            'ASSURE_SINGLE_TRACKER': False
        }
        return default_config

    def get_output_fol(self, tracker):
        return os.path.join(self.output_fol, self.tracker_to_disp[tracker], self.output_sub_fol)

    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()

        self.n_joints = 15
        self.n_raw_joints = 17
        self.joint_names = ['Nose', 'Neck', 'Head', 'LS', 'RS', 'LE', 'RE', 'LW', 'RW', 'LH', 'RH', 'LK', 'RK', 'LA', 'RA']
        # Fill non-given config values with defaults
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())

        self.gt_fol = self.config['GT_FOLDER']
        self.tracker_fol = self.config['TRACKERS_FOLDER']
        self.output_sub_fol = ''
        self.should_classes_combine = False
        self.use_super_categories = False

        self.output_fol = None
        if self.output_fol is None:
            self.output_fol = self.tracker_fol

        # # Get classes to eval
        self.valid_classes = ['person']
        self.class_list = [cls for cls in self.valid_classes]
        # if track folder is 'sequences' folder, have a single tracker
        # otherwise, each subfolder must contain a tracker_name which contains a sequences_folder
        # tracker_fol
        # |
        # |---tracker_1
        # |   |---sequences
        # |---tracker_2
        # |   |---sequences

        tracker_fol_json_files = [file for file in os.listdir(self.tracker_fol) if '.json' in file and '_test' in file]

        if os.path.basename(os.path.normpath(self.tracker_fol)) == 'sequences':
            curr_path = Path(self.tracker_fol)
            parent_path = curr_path.parent.absolute()
            self.tracker_fol = parent_path 
            self.tracker_list = ['sequences']
            self.tracker_to_disp = {folder: '' for folder in self.tracker_list}
        elif len(tracker_fol_json_files) > 0:
            curr_path = Path(self.tracker_fol)
            parent_path = curr_path.parent.absolute()
            self.tracker_fol = parent_path 
            self.tracker_list = [os.path.basename(curr_path)]
            self.tracker_to_disp = {folder: '' for folder in self.tracker_list}
        else:
            self.tracker_list = []
            self.tracker_to_disp = dict()
            for folder in os.listdir(self.tracker_fol):
                folder_path = os.path.join(self.tracker_fol, folder)
                if not os.path.isdir(folder_path):
                    continue

                sub_files = os.listdir(folder_path)
                if 'sequences' in sub_files:
                    self.tracker_list.append(f'{folder}/sequences') 
                    self.tracker_to_disp[f'{folder}/sequences'] = folder 

        # Get sequences to eval and check gt files exist
        self.seq_list, self.seq_lengths = self._get_seq_info()
        if len(self.seq_list) < 1:
            raise TrackEvalException('No sequences are selected to be evaluated.')

        # Check gt files exist
        if config['ASSURE_SINGLE_TRACKER']:
            assert len(self.tracker_list) == 1

        for tracker in self.tracker_list:
            for seq in self.seq_list:

                det_file = os.path.join(self.tracker_fol, tracker,seq)

                if not os.path.isfile(det_file):
                    print(f"DET file {det_file} not found for tracker {tracker}")
                    raise TrackEvalException(f"DET file {det_file} not found for tracker {tracker}")

    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def _get_seq_info(self):
        sequence_files = os.listdir(self.gt_fol)
        seq_lengths = dict()

        # reading sequence lengths
        for seq in sequence_files:
            seq_path = os.path.join(self.gt_fol, seq)

            with open(seq_path, 'r') as f:
                seq_data = json.load(f)

            annotated_images = [img for img in seq_data['images']]
            seq_lengths[seq] = len(annotated_images)

        return sequence_files, seq_lengths

    def _get_head_size(self, x1, y1, x2, y2):
        head_size = 0.6 * np.linalg.norm(np.subtract([x2, y2], [x1, y1]))
        return head_size

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
            file_path = os.path.join(self.gt_fol, seq)
        else:
            file_path = os.path.join(self.tracker_fol, tracker, seq)
        with open(file_path, 'r') as f:
            read_data = json.load(f)

        image_data = {img['id']: {**img, 'annotations': []} for img in read_data['images']}
        image_ids = list(image_data.keys())

        for ann in read_data['annotations']:
            im_id = ann['image_id']
            image_data[im_id]['annotations'].append(ann)

        # Convert data to required format
        num_timesteps = self.seq_lengths[seq]
        data_keys = ['ids', 'classes', 'dets', 'image_ids']
        if is_gt:
            data_keys += ['gt_crowd_ignore_regions', 'gt_extras', 'head_sizes', 'ignore_regions', 'is_labeled']
        else:
            data_keys += ['tracker_confidences', 'keypoint_detected']
        raw_data = {key: [None] * num_timesteps for key in data_keys}

        for t in range(num_timesteps):
            im_id = image_ids[t]
            frame_data = image_data[im_id]

            raw_data['image_ids'][t] = im_id
            # add ignore regions
            if is_gt:
                raw_data['ignore_regions'][t] = (frame_data['ignore_regions_x'], frame_data['ignore_regions_y'])
                raw_data['is_labeled'][t] = frame_data['is_labeled']

            frame_annotations = frame_data['annotations']
            person_ids = []
            keypoints = []
            scores = []
            head_sizes = []
            for p in frame_annotations:

                person_ids.append(p['person_id'])
                keypoints.append(p['keypoints'])

                if 'scores' not in p:
                    #TODO: Warning that scores are not present!
                    p_score = [-9999 for _ in range(self.n_raw_joints)]
                    scores.append(p_score)
                else:
                    scores.append(p['scores'])

                if is_gt:
                    head_bb = p['bbox_head']
                    x1, y1, w, h = head_bb
                    x2, y2 = x1 + w, y1 + h
                    head_size = self._get_head_size(x1, y1, x2, y2)
                    head_sizes.append(head_size)

            person_ids = np.array(person_ids)
            keypoints = np.array(keypoints).reshape([-1, self.n_raw_joints, 3])
            keypoints = np.delete(keypoints, [3, 4], axis=1)

            scores = np.array(scores).reshape([-1, self.n_raw_joints])
            scores = np.delete(scores, [3, 4], axis=1)

            if not is_gt:
                keypoints[:, :, 2] = scores

                raw_data['keypoint_detected'][t] = (keypoints[:, :, 0] > 0) & \
                                                   (keypoints[:, :, 1] > 0) & \
                                                   (keypoints[:, :, 2] > 0)

            if len(frame_annotations) > 0:
                raw_data['dets'][t] = keypoints
                raw_data['ids'][t] = person_ids
                raw_data['classes'][t] = np.ones_like(raw_data['ids'][t])   # we only have one class

                if not is_gt:
                    raw_data['tracker_confidences'][t] = scores
                else:
                    raw_data['gt_extras'][t] = {}   # ToDO
                    raw_data['head_sizes'][t] = head_sizes

            else:
                raw_data['dets'][t] = np.empty((0, self.n_joints, 3))
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
                       'dets': 'gt_dets',
                       'head_sizes': 'head_sizes',
                       'ignore_regions': 'ignore_regions',
                       'image_ids': 'image_ids',
                       'is_labeled': 'is_labeled'}
        else:
            key_map = {'ids': 'tracker_ids',
                       'classes': 'tracker_classes',
                       'dets': 'tracker_dets',
                       'image_ids': 'image_ids',
                       'keypoint_detected': 'keypoint_detected'}
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

        cls_id = 1  # we only have class person

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'tracker_confidences', 'similarity_scores', 'keypoint_distances',
                     'keypoint_matches', 'original_gt_ids', 'original_tracker_ids']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0

        num_gt_joints = np.zeros([self.n_joints], dtype=int)
        num_tracker_joints = np.zeros([self.n_joints], dtype=int)

        for t in range(raw_data['num_timesteps']):
            tracker_classes = raw_data['tracker_classes'][t]

            # evaluation is only valid for pedestrian class
            if len(tracker_classes) > 0 and np.max(tracker_classes) > 1:
                raise trackevalexception(
                    'evaluation is only valid for persons class. non person class (%i) found in sequence %s at '
                    'timestep %i.' % (np.max(tracker_classes), raw_data['seq'], t))

            # for now, do not perform pre-processing and copy data!
            for k in data_keys:
                if k in raw_data:
                    data[k][t] = raw_data[k][t]

            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
            num_tracker_dets += len(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])

        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['original_gt_ids'][t] = data['gt_ids'][t].copy()

                    gt_dets = data['gt_dets'][t]
                    num_gt_joints += count_valid_joints(gt_dets)

        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['original_tracker_ids'][t] = data['tracker_ids'][t].copy()

                    tracker_dets = data['tracker_dets'][t]
                    num_tracker_joints += count_valid_joints(tracker_dets)

        # record overview statistics.
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_timesteps'] = raw_data['num_timesteps']
        data['seq'] = raw_data['seq']
        data['num_gt_joints'] = num_gt_joints
        data['num_tracker_joints'] = num_tracker_joints

        # Ensure again that ids are unique per timestep after preproc.
        self._check_unique_ids(data, after_preproc=True)

        return data

    def remove_empty_frames(self, raw_data, is_labeled):
        new_data = {}
        for k, v in raw_data.items():

            if isinstance(v, list):
                new_data[k] = []
                assert len(is_labeled) == len(v)

                for t in range(len(v)):
                    if is_labeled[t]:
                        new_data[k].append(v[t])
            else:
                new_data[k] = v
        return new_data

    def create_empty_data(self, raw_data):
        new_data = {}
        for k, v in raw_data.items():

            if isinstance(v, list):
                new_data[k] = [None for _ in range(len(v))]
            else:
                new_data[k] = v
        return new_data

    def remove_ignored_points(self, dets, poly_list):
        remove_idxs = np.zeros([dets.shape[0], dets.shape[1]], dtype=bool)
        for pidx in range(len(dets)):
            points = dets[pidx]
            for j in range(len(points)):
                pt = geometry.Point(points[j, 0], points[j, 1])
                b_ignore = False

                for poidx in range(len(poly_list)):
                    poly = poly_list[poidx]
                    if poly.contains(pt):
                        b_ignore = True
                        break
                if b_ignore:
                    remove_idxs[pidx, j] = True

        return remove_idxs

    @_timing.time
    def get_raw_seq_data(self, tracker, seq):
        """ Loads raw data (tracker and ground-truth) for a single tracker on a single sequence.
        Raw data includes all of the information needed for both preprocessing and evaluation, for all classes.
        A later function (get_processed_seq_data) will perform such preprocessing and extract relevant information for
        the evaluation of each class.

        This returns a dict which contains the fields:
        [num_timesteps]: integer
        [gt_ids, tracker_ids, gt_classes, tracker_classes, tracker_confidences]:
                                                                list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, tracker_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [similarity_scores]: list (for each timestep) of 2D NDArrays.
        [gt_extras]: dict (for each extra) of lists (for each timestep) of 1D NDArrays (for each det).

        gt_extras contains dataset specific information used for preprocessing such as occlusion and truncation levels.

        Note that similarities are extracted as part of the dataset and not the metric, because almost all metrics are
        independent of the exact method of calculating the similarity. However datasets are not (e.g. segmentation
        masks vs 2D boxes vs 3D boxes).
        We calculate the similarity before preprocessing because often both preprocessing and evaluation require it and
        we don't wish to calculate this twice.
        We calculate similarity between all gt and tracker classes (not just each class individually) to allow for
        calculation of metrics such as class confusion matrices. Typically the impact of this on performance is low.
        """

        # Load raw data.
        raw_gt_data = self._load_raw_file(tracker, seq, is_gt=True)
        raw_tracker_data = self._load_raw_file(tracker, seq, is_gt=False)


        if len(raw_gt_data['image_ids']) != len(raw_tracker_data['image_ids']):
            raise TrackEvalException("The number of frames does not match ground truth")

        # 1) Remove frames that do not contain gt annotations
        raw_gt_data_new = self.remove_empty_frames(raw_gt_data, raw_gt_data['is_labeled'])
        raw_tracker_data_new = self.remove_empty_frames(raw_tracker_data, raw_gt_data['is_labeled'])

        # 1.2) update timesteps
        raw_gt_data_new['num_timesteps'] = len(raw_gt_data_new['image_ids'])
        raw_tracker_data_new['num_timesteps'] = len(raw_tracker_data_new['image_ids'])

        # 2) Remove detected keypoints from ignore regions
        for t, (ignore_regions_t, gt_dets_t, tracker_dets_t, image_id_t1, image_id_t2) in enumerate(zip(raw_gt_data_new['ignore_regions'],
                                                                                                        raw_gt_data_new['gt_dets'],
                                                                                                        raw_tracker_data_new['tracker_dets'],
                                                                                                        raw_gt_data_new['image_ids'],
                                                                                                        raw_tracker_data_new['image_ids'])):

            assert image_id_t1 == image_id_t2
            regions_x, regions_y = ignore_regions_t
            if len(regions_x) == 0:
                continue

            poly_list = []
            for ridx in range(len(regions_x)):
                region_x = regions_x[ridx]
                region_y = regions_y[ridx]

                point_list = []
                for pidx in range(len(region_x)):
                    pt = geometry.Point(region_x[pidx], region_y[pidx])
                    point_list.append(pt)
                poly = geometry.Polygon([[p.x, p.y] for p in point_list])
                poly_list.append(poly)

            remove_det_idxs = self.remove_ignored_points(tracker_dets_t, poly_list)
            remove_gt_idxs  = self.remove_ignored_points(gt_dets_t, poly_list)

            # remove from detections
            tracker_confidences = []
            tracker_ids = []
            tracker_classes = []
            tracker_dets = []
            for pidx in range(len(remove_det_idxs)):
                # check if person should be removed completely:
                if np.sum(remove_det_idxs[pidx]) == remove_det_idxs[pidx].shape[0]:
                    continue

                confidences = raw_tracker_data_new['tracker_confidences'][t][pidx]
                track_id = raw_tracker_data_new['tracker_ids'][t][pidx]
                classe = raw_tracker_data_new['tracker_classes'][t][pidx]
                det = raw_tracker_data_new['tracker_dets'][t][pidx]

                for j in range(len(remove_det_idxs[pidx])):
                    if remove_det_idxs[pidx][j]:
                        confidences[j] = 0
                        det[j] = [0, 0, 0]

                invalid = (det[:, 0] <= 0) & (det[:, 1] <= 0)
                if np.sum(invalid) == det.shape[0]:
                    continue

                tracker_confidences.append(confidences)
                tracker_ids.append(track_id)
                tracker_classes.append(classe)
                tracker_dets.append(det)

            # 2.2) update tracker info
            raw_tracker_data_new['tracker_confidences'][t] = np.array(tracker_confidences).reshape([-1, self.n_joints])
            raw_tracker_data_new['tracker_ids'][t] = np.array(tracker_ids)
            raw_tracker_data_new['tracker_classes'][t] = np.array(tracker_classes)
            raw_tracker_data_new['tracker_dets'][t] = np.array(tracker_dets).reshape([-1, self.n_joints, 3])


            gt_ids = []
            gt_classes = []
            gt_dets = []
            head_sizes = []

            for pidx in range(len(remove_gt_idxs)):
                # check if person should be removed completely:
                if np.sum(remove_gt_idxs[pidx]) == remove_gt_idxs[pidx].shape[0]:
                    continue

                gt_id = raw_gt_data_new['gt_ids'][t][pidx]
                head_size = raw_gt_data_new['head_sizes'][t][pidx]
                gt_class = raw_gt_data_new['gt_classes'][t][pidx]
                gt_pose = raw_gt_data_new['gt_dets'][t][pidx]

                for j in range(len(remove_gt_idxs[pidx])):
                    if remove_gt_idxs[pidx][j]:
                        gt_pose[j] = [0, 0, 0]

                invalid = (gt_pose[:, 0] <= 0) & (gt_pose[:, 1] <= 0)
                if np.sum(invalid) == gt_pose.shape[0]:
                    continue

                gt_ids.append(gt_id)
                gt_classes.append(gt_class)
                gt_dets.append(gt_pose)
                head_sizes.append(head_size)

            # 2.3) update gt data
            raw_gt_data_new['gt_ids'][t] = np.array(gt_ids)
            raw_gt_data_new['gt_classes'][t] = np.array(gt_classes)
            raw_gt_data_new['gt_dets'][t] = np.array(gt_dets)
            raw_gt_data_new['head_sizes'][t] = np.array(head_sizes)

        raw_data = {**raw_tracker_data_new, **raw_gt_data_new}  # Merges dictionaries

        # Calculate similarities for each timestep.
        similarity_scores = []
        keypoint_distances = []
        keypoint_matches = []
        for t, (gt_dets_t, tracker_dets_t, head_sizes_t) in enumerate(zip(raw_data['gt_dets'], raw_data['tracker_dets'], raw_data['head_sizes'])):

            pckhs, distances, matches = self._calculate_p_similarities(gt_dets_t, tracker_dets_t, head_sizes_t)
            similarity_scores.append(pckhs)
            keypoint_distances.append(distances)
            keypoint_matches.append(matches)

        raw_data['similarity_scores'] = similarity_scores
        raw_data['keypoint_distances'] = keypoint_distances
        raw_data['keypoint_matches'] = keypoint_matches
        return raw_data

    def _calculate_pckh(self, gt_dets_t, tracker_dets_t, head_sizes_t, dist_thres=0.5):
        assert len(gt_dets_t) == len(head_sizes_t)

        dist = np.full((len(gt_dets_t), len(tracker_dets_t), self.n_joints), np.inf)

        joint_has_gt = (gt_dets_t[:, :, 0] > 0) & (gt_dets_t[:, :, 1] > 0)

        joint_has_pr = (tracker_dets_t[:, :, 0] > 0) & (tracker_dets_t[:, :, 1] > 0)

        for gt_i in range(len(gt_dets_t)):
            head_size_i = head_sizes_t[gt_i]

            for det_i in range(len(tracker_dets_t)):
                for j in range(self.n_joints):
                    if joint_has_gt[gt_i, j] and joint_has_pr[det_i, j]:
                        gt_point = gt_dets_t[gt_i, j, :2]
                        det_point = tracker_dets_t[det_i, j, :2]

                        dist[gt_i, det_i, j] = np.linalg.norm(np.subtract(gt_point, det_point)) / head_size_i

        # number of annotated joints
        nGTp = np.sum(joint_has_gt, axis=1)
        match = dist <= dist_thres
        pck = 1.0 * np.sum(match, axis=2)
        for i in range(joint_has_gt.shape[0]):
            for j in range(joint_has_pr.shape[0]):
                if nGTp[i] > 0:
                    pck[i, j] = pck[i, j] / nGTp[i]

        return pck, dist, match

    def _calculate_p_similarities(self, gt_dets_t, tracker_dets_t, head_sizes_t):
        similarity_scores, distances, matches = self._calculate_pckh(gt_dets_t, tracker_dets_t, head_sizes_t)
        return similarity_scores, distances, matches

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        raise NotImplementedError("Not implemented")

