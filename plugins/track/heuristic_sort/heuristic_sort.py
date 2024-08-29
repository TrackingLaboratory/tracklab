import numpy as np

from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

import cv2
import torch
from torch.nn import functional as F
from torchreid.metrics.distance import compute_distance_matrix_using_bp_features

import logging

log = logging.getLogger(__name__)

np.random.seed(0)

INIT_STD = 0.1


class Detection:
    def __init__(self, keypoints, apparence_features, pbtrack_id):
        self.keypoints = keypoints
        self.center = keypoints_to_center(keypoints)
        self.apparence_features = apparence_features
        self.pbtrack_id = int(pbtrack_id)


def linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def convert_center_to_z(center):
    z = center.reshape((2, 1))
    return z


def keypoints_to_center(keypoints):
    vis_keypoints = keypoints[keypoints[:, 2] == 1.0]
    assert vis_keypoints.shape[0] > 0
    center = np.mean(vis_keypoints[:, :2], axis=0).reshape(-1)
    return center


def convert_x_to_center(x):
    return x[:2].reshape(-1)


class Tracklet(object):
    count = 0

    def __init__(self, detection):
        """
        Initialises a tracker using initial keypoints.
        """

        # Kalfman filter over center position
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        # state x = [x, y, v_x, v_y]
        # observation z = [x, y]

        self.kf.F = np.array(  # state-transition model
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(  # observation model
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ]
        )

        # FIXME tune these parameters
        self.kf.R *= 10.0  # covariance of the observation noise  # {1.; 10.0}
        self.kf.P[  # a posteriori estimate covariance matrix
            2:, 2:
        ] *= 1000.0  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0  # a posteriori estimate covariance matrix  # {1.; 10.0}
        self.kf.Q[2:, 2:] *= 0.01  # covariance of the process noise

        self.kf.x[:2] = convert_center_to_z(detection.center)
        self.id = Tracklet.count
        Tracklet.count += 1

        # tracklet management
        self.age = 0
        self.hits = 0

        self.time_wo_hits = 0
        self.hit_streak = 0

        # observations
        self.last_center = detection.center
        self.last_keypoints = detection.keypoints

        # predictions of KF
        self.pred_center = np.empty((2,))

        # updates of KF
        self.update_center = np.empty((2,))

        self.apparence_features = detection.apparence_features
        self.ema_alpha = 0.9

        self.state = "init"

        self.pbtrack_id = detection.pbtrack_id

    def update(self, detection):
        # tracklet management
        self.hits += 1
        self.hit_streak += 1
        self.time_wo_hits = 0
        self.kf.update(convert_center_to_z(detection.center))

        # observations
        self.last_center = detection.center
        self.last_keypoints = detection.keypoints

        # updates of KF
        self.update_center = convert_x_to_center(self.kf.x)

        # update apparence features
        detection_features = detection.apparence_features["reid_features"]
        detection_vis_scores = detection.apparence_features["visibility_scores"]

        tracklet_features = self.apparence_features["reid_features"]
        tracklet_vis_scores = self.apparence_features["visibility_scores"]

        xor = np.logical_xor(tracklet_vis_scores, detection_vis_scores)
        ema_scores_tracklet = (tracklet_vis_scores * detection_vis_scores) * np.float32(
            self.ema_alpha
        ) + xor * tracklet_vis_scores
        ema_scores_detection = (
            tracklet_vis_scores * detection_vis_scores
        ) * np.float32(1 - self.ema_alpha) + xor * detection_vis_scores
        smooth_feat = (
            np.expand_dims(ema_scores_tracklet, 1) * tracklet_features
            + np.expand_dims(ema_scores_detection, 1) * detection_features
        )
        smooth_visibility_scores = np.maximum(tracklet_vis_scores, detection_vis_scores)
        smooth_feat[
            np.logical_and(ema_scores_tracklet == 0.0, ema_scores_detection == 0.0)
        ] = 1
        # smooth_feat /= np.linalg.norm(smooth_feat, axis=-1, keepdims=True)  # TODO can cause div by 0 + check if norm is already performed in compute_distance_matrix_using_bp_features
        self.apparence_features = {
            "reid_features": smooth_feat,
            "visibility_scores": smooth_visibility_scores,
        }

        self.pbtrack_id = detection.pbtrack_id

    def predict(self):
        """
        Advances the state vector and returns the predicted pose estimates
        """
        self.age += 1
        if self.time_wo_hits > 0:
            self.hit_streak = 0
        self.time_wo_hits += 1
        self.kf.predict()

        # predictions of KF
        self.pred_center = convert_x_to_center(self.kf.x)

    def compense_camera_motion(self, warp_matrix):
        center = convert_x_to_center(self.kf.x)
        center = np.array([center[0], center[1], 1.0])
        new_center = warp_matrix @ center
        self.kf.x[:2] = convert_center_to_z(new_center)


def align_skeletons(kps1, kps2):
    offset = np.mean(kps2 - kps1, axis=0)
    aligned_kps1 = kps1 + offset
    return aligned_kps1


def compute_sim_pose(detection, tracklet, aligned=False):
    kp1 = tracklet.last_keypoints
    kp2 = detection.keypoints

    vis_kp = np.logical_and(kp1[:, 2] == 1.0, kp2[:, 2] == 1.0)
    if vis_kp.sum() < 3:
        return 0.5

    kp1 = kp1[vis_kp, :2]
    kp2 = kp2[vis_kp, :2]
    if aligned:
        kp2 = align_skeletons(kp2, kp1)

    scale = np.sqrt(
        (kp1[:, 0].max() - kp1[:, 0].min()) * (kp1[:, 1].max() - kp1[:, 1].min())
    )  # sqrt(area)
    scale = max(1.0, scale)  # avoid division by 0

    sigma = INIT_STD

    dist = np.linalg.norm(kp1 - kp2, axis=1)
    oks = np.exp(-0.5 * dist**2 / (scale**2 * sigma**2))
    oks = np.mean(oks)
    return oks


def compute_sim_loc(detection, tracklet, img_size):
    det_center = detection.center
    trk_center = tracklet.pred_center

    scale = np.sqrt(img_size[0] ** 2 + img_size[1] ** 2)
    scale = max(1.0, scale)  # avoid division by 0

    sigma = INIT_STD

    dist = np.linalg.norm(det_center - trk_center)
    sim_loc = np.exp(-0.5 * dist**2 / (scale**2 * sigma**2))
    return sim_loc


def compute_reid_sim(detection, tracklet, normalize_features=True):
    x_features = torch.from_numpy(detection.apparence_features["reid_features"])
    x_visibility_scores = torch.from_numpy(
        detection.apparence_features["visibility_scores"]
    )
    y_features = torch.from_numpy(tracklet.apparence_features["reid_features"])
    y_visibility_scores = torch.from_numpy(
        tracklet.apparence_features["visibility_scores"]
    )
    if normalize_features:
        """Features are normalized here, which is not the most efficient since a given tracklet features will be normalized multiple times.
        It is not done before because we want to do the exponential moving average (EMA) aggregation of tracklets features on the un-normalized features
        and then normalize after. If normalization was done before EMA, the averaged features would not be normalized
        anymore and we would therefore pass un-normalized features to 'compute_distance_matrix_using_bp_features'.
        """
        x_features = F.normalize(x_features, p=2, dim=-1)
        y_features = F.normalize(y_features, p=2, dim=-1)
    distances = compute_distance_matrix_using_bp_features(
        y_features.unsqueeze(0),
        x_features.unsqueeze(0),
        y_visibility_scores.unsqueeze(0),
        x_visibility_scores.unsqueeze(0),
        use_gpu=False,
        use_logger=False,
    )
    distances = (
        distances[0] / 2
    )  # When feature are normalized, the above function returns distances within [0, 2]
    # TODO handle invalid distances
    reid_sim = 1 - distances.mean()
    return reid_sim


def detect_change_of_view(prev_frame, actual_frame):
    # Perform frame differencing
    diff = cv2.absdiff(prev_frame, actual_frame)

    # Apply thresholding to highlight significant differences
    _, threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

    # Calculate the percentage of changed pixels
    changed_pixels = np.sum(threshold) / 255
    total_pixels = threshold.shape[0] * threshold.shape[1]
    change_percentage = changed_pixels / total_pixels

    return change_percentage > 0.5


def compute_camera_motion(prev_frame, actual_frame):
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    warp_mode = cv2.MOTION_EUCLIDEAN
    try:
        (_, warp_matrix) = cv2.findTransformECC(
            prev_frame, actual_frame, warp_matrix, warp_mode, criteria
        )
    except cv2.error as e:
        warp_matrix = None
    return warp_matrix


class HeuristicSORT(object):
    def __init__(
        self,
        min_hits,
        max_wo_hits,
        use_pose_and_loc,
        use_appearance,
        max_pose_and_loc_wo_hits,
        pose_threshold,
        loc_threshold,
        app_threshold,
        sim_threshold,
        camera_motion_correction,
        min_vis_keypoints,
        vis_keypoint_threshold,
    ):
        self.min_hits = min_hits
        self.max_wo_hits = max_wo_hits

        self.use_pose_and_loc = use_pose_and_loc
        self.use_appearance = use_appearance
        assert use_pose_and_loc or use_appearance, (
            "At least one of 'use_pose_and_loc' or " "'use_appearance' must be True"
        )
        if use_pose_and_loc and use_appearance:
            self.max_pose_and_loc_wo_hits = max_pose_and_loc_wo_hits
        else:
            self.max_pose_and_loc_wo_hits = 0

        self.pose_threshold = pose_threshold
        self.loc_threshold = loc_threshold
        self.app_threshold = app_threshold
        self.sim_threshold = sim_threshold

        self.camera_motion_correction = camera_motion_correction

        self.min_vis_keypoints = min_vis_keypoints
        self.vis_keypoint_threshold = vis_keypoint_threshold

        self.tracklets = []
        self.frame_count = 0

    def compute_spatio_temp_matrix(self, detections, tracklets):
        spatio_temp_matrix = np.zeros((len(detections), len(tracklets)))
        for i, det in enumerate(detections):
            for j, trk in enumerate(tracklets):
                sim_loc = compute_sim_loc(det, trk, self.img_size)
                if sim_loc < self.loc_threshold:
                    continue

                sim_pose = compute_sim_pose(det, trk, True)
                if sim_pose < self.pose_threshold:
                    continue
                spatio_temp_matrix[i, j] = np.sqrt(sim_loc * sim_pose)
        return spatio_temp_matrix

    def compute_app_matrix(self, detections, tracklets):
        app_matrix = np.zeros((len(detections), len(tracklets)))
        for i, det in enumerate(detections):
            for j, trk in enumerate(tracklets):
                reid_sim = compute_reid_sim(det, trk)
                if reid_sim < self.app_threshold:
                    continue
                app_matrix[i, j] = reid_sim
        return app_matrix

    def compute_alpha_matrix(self, detections, tracklets):
        alpha_matrix = np.zeros((len(detections), len(tracklets)))
        if self.max_pose_and_loc_wo_hits != 0:
            for j, trk in enumerate(tracklets):
                alpha_matrix[:, j] = np.maximum(
                    0.0,
                    1.0
                    - (trk.time_wo_hits - 1) / self.max_pose_and_loc_wo_hits,
                )
        return alpha_matrix

    def associate_dets_to_trks(self, detections):
        if len(self.tracklets) == 0:
            return (
                np.empty((0, 2)),
                np.arange(len(detections)),
                np.empty((0,)),
            )

        if self.use_pose_and_loc and self.use_appearance:
            spatio_temp_matrix = self.compute_spatio_temp_matrix(
                detections, self.tracklets
            )
            app_matrix = self.compute_app_matrix(detections, self.tracklets)
            alpha_matrix = self.compute_alpha_matrix(detections, self.tracklets)
            sim_matrix = np.sqrt(
                spatio_temp_matrix**alpha_matrix * app_matrix ** (2 - alpha_matrix)
            )
        elif self.use_pose_and_loc:
            sim_matrix = self.compute_spatio_temp_matrix(detections, self.tracklets)
        elif self.use_appearance:
            sim_matrix = self.compute_app_matrix(detections, self.tracklets)
        sim_matrix[sim_matrix < self.sim_threshold] = 0.0

        if min(sim_matrix.shape) > 0:
            a = (sim_matrix > 0).astype(int)
            # if only one match, no association problem to solve
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            # Hungarian algorithm
            else:
                matched_indices = linear_assignment(-sim_matrix)
        else:
            matched_indices = np.empty((0, 2))

        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t in range(len(self.tracklets)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        # filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if sim_matrix[m[0], m[1]] <= 0.0:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def handle_camera_motion(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if hasattr(self, "prev_frame") and len(self.tracklets) > 0:
            has_changed = detect_change_of_view(self.prev_frame, frame)
            if has_changed:
                # in case of POV change, set all the tracklets to rely only on reid
                for trk in self.tracklets:
                    if trk.time_wo_hits < self.max_pose_and_loc_wo_hits:
                        trk.time_wo_hits = self.max_pose_and_loc_wo_hits
            else:
                # compute camera motion compensation
                warp_matrix = compute_camera_motion(self.prev_frame, frame)
                if warp_matrix is not None:
                    for trk in self.tracklets:
                        trk.compense_camera_motion(warp_matrix)
        self.prev_frame = frame

    def update(self, keypoints, reid_features, visibility_scores, pbtrack_ids, image):
        self.frame_count += 1
        self.img_size = image.shape[:2]

        if self.camera_motion_correction:
            self.handle_camera_motion(image)

        # get detections
        assert (
            len(keypoints)
            == len(reid_features)
            == len(visibility_scores)
            == len(pbtrack_ids)
        )
        detections = []
        for i, kpts in enumerate(keypoints):
            kpts[:, 2] = np.where(kpts[:, 2] > self.vis_keypoint_threshold, 1.0, 0.0)
            is_valid = kpts[:, 2].sum() >= self.min_vis_keypoints
            if is_valid:
                detections.append(
                    Detection(
                        kpts,
                        {
                            "reid_features": reid_features[i],
                            "visibility_scores": visibility_scores[i],
                        },
                        pbtrack_ids[i],
                    )
                )

        # propagated poses from tracklets
        for tracklet in self.tracklets:
            tracklet.predict()

        (
            matched,
            unmatched_dets,
            unmatched_trks,
        ) = self.associate_dets_to_trks(detections)

        # update matched tracklets with assigned detections
        for m in matched:
            self.tracklets[m[1]].update(detections[m[0]])

        # create and initialise new tracklets for unmatched detections
        for i in unmatched_dets:
            trk = Tracklet(detections[i])
            self.tracklets.append(trk)

        # handle tracklets outputs and cleaning
        actives = []
        for trk in self.tracklets:
            # get active tracklets
            self.update_state(trk)
            if trk.state == "active":
                actives.append(
                    {
                        "last_keypoints_xyc": trk.last_keypoints,
                        "last_center": trk.last_center,
                        "pred_center": trk.pred_center,
                        "update_center": trk.update_center,
                        "pbtrack_id": trk.pbtrack_id,
                        "track_id": trk.id + 1,  # MOT benchmark requires positive
                    }
                )
        self.tracklets = [trk for trk in self.tracklets if trk.state != "dead"]
        return actives

    def update_state(self, tracklet):
        s = tracklet.state
        if s == "init":
            if tracklet.time_wo_hits >= 1:
                new_state = "dead"
            elif tracklet.hit_streak >= self.min_hits:
                new_state = "active"
            else:
                new_state = "init"
        elif s == "active":
            if tracklet.time_wo_hits == 0:
                new_state = "active"
            else:
                new_state = "recover"
        elif s == "recover":
            if tracklet.time_wo_hits == 0:
                new_state = "active"
            elif tracklet.time_wo_hits >= self.max_wo_hits:
                new_state = "dead"
            else:
                new_state = "recover"
        elif s == "dead":
                new_state = "dead"
        else:
            raise ValueError(f"tracklet state is in undefined state {s}.")
        tracklet.state = new_state
