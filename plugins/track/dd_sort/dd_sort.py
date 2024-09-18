import cv2
import torch
import numpy as np
import logging

log = logging.getLogger(__name__)


class Detection:
    def __init__(self, image_id, features, pbtrack_id, frame_idx):
        for k, v in features.items():
            setattr(self, k, v)
        self.pbtrack_id = pbtrack_id
        self.image_id = image_id
        self.frame_idx = torch.tensor(frame_idx)
        self.similarity_with_tracklet = None
        self.similarities = None


class Tracklet(object):
    # MOT benchmark requires positive:
    count = 1  # FIXME not thread safe

    def __init__(self, detection, max_gallery_size):
        self.last_detection = detection
        # self.token = detection.token
        self.detections = [detection]
        self.state = "init"
        self.id = Tracklet.count
        Tracklet.count += 1

        self.age = 0
        self.hits = 0
        self.hit_streak = 0
        self.time_wo_hits = 0

        self.max_gallery_size = max_gallery_size

    def forward(self):
        self.age += 1
        self.time_wo_hits += 1
        if self.time_wo_hits > 1:
            self.hit_streak = 0

    def update(self, detection):
        self.detections.append(detection)
        self.detections = self.detections[-self.max_gallery_size:]
        # tracklet management
        self.hits += 1
        self.hit_streak += 1
        self.time_wo_hits = 0
        self.last_detection = detection

    def padded_features(self, name, size):
        features = torch.stack([getattr(det, name) for det in reversed(self.detections)])
        if features.shape[0] < size:
            features = torch.cat(
                [features,
                 torch.zeros(size - features.shape[0], *features.shape[1:], device=features.device) + float('nan')]
            )
        return features


def detect_change_of_view(prev_frame, actual_frame):  # TODO move elsewhere
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


@torch.no_grad()
class DDSORT(object):
    def __init__(
            self,
            simformer,
            min_hits,
            max_wo_hits,
            max_gallery_size=50,
            camera_motion_correction=False,  # fixme
            **kwargs,
    ):
        self.simformer = simformer.eval()
        self.min_hits = min_hits
        self.max_wo_hits = max_wo_hits
        self.max_gallery_size = max_gallery_size
        self.camera_motion_correction = camera_motion_correction  # fixme
        self.tracklets = []
        self.frame_count = 0

    def update(self, features, pbtrack_ids, image_id, image):
        self.frame_count += 1

        # camera motion compensation
        if self.camera_motion_correction:
            self.handle_camera_motion(image)

        # build detections from features
        detections = []
        for i in range(len(pbtrack_ids)):
            features_i = {k: v[0, i] for k, v in features.items()}
            detections.append(
                Detection(
                    image_id,
                    features_i,
                    pbtrack_ids[i],
                    frame_idx=self.frame_count - 1
                )
            )

        # advance state of tracklets
        for track in self.tracklets:
            track.forward()

        # associate detections to tracklets
        (
            matched,
            unmatched_trks,
            unmatched_dets,
            td_sim_matrix,
        ) = self.associate_dets_to_trks(self.tracklets, detections, image)

        # update matched tracklets with assigned detections
        for m in matched:
            tracklet = self.tracklets[m[0]]
            detection = detections[m[1]]
            tracklet.update(detection)
            similarity = td_sim_matrix[m[0], m[1]]
            detection.similarity_with_tracklet = similarity
            detection.similarities = td_sim_matrix[:, m[1]]

        # create and initialise new tracklets for unmatched detections
        for i in unmatched_dets:
            trk = Tracklet(detections[i], self.max_gallery_size)
            self.tracklets.append(trk)

        # handle tracklets outputs and cleaning
        actives = []
        for trk in self.tracklets:
            # get active tracklets
            self.update_state(trk)
            if trk.state == "active":
                actives.append(
                    {
                        "pbtrack_id": trk.last_detection.pbtrack_id.item(),
                        "track_id": trk.id,
                        "hits": trk.hits,
                        "age": trk.age,
                        "matched_with": ("S", trk.last_detection.similarity_with_tracklet.cpu().numpy()),
                        "time_since_update": trk.time_wo_hits,
                        "state": trk.state,
                        "costs": {
                            "S": {self.tracklets[j].id: sim for j, sim in
                                  enumerate(trk.last_detection.similarities.cpu().numpy())},
                            "St": self.simformer.sim_threshold,
                        }
                    }
                )
        self.tracklets = [trk for trk in self.tracklets if trk.state != "dead"]
        return actives

    def associate_dets_to_trks(self, tracklets, detections, image):
        if len(tracklets) == 0:
            return (
                np.empty((0, 2)),
                np.empty((0,)),
                np.arange(len(detections)),
                np.empty((0,)),
            )
        if len(detections) == 0:
            return (
                np.empty((0, 2)),
                np.arange(len(self.tracklets)),
                np.empty((0,)),
                np.empty((0,)),
            )

        batch = self.build_simformer_batch(tracklets, detections, self.simformer.device, image)
        association_matrix, association_result, td_sim_matrix = self.simformer.predict_step(batch, self.frame_count)
        matched = association_result[0]["matched_td_indices"]
        unmatched_trks = association_result[0]["unmatched_trackers"]
        unmatched_dets = association_result[0]["unmatched_detections"]
        return matched, unmatched_trks, unmatched_dets, td_sim_matrix.squeeze(0)

    def matched_td_indices(self, tracklet_detection_matrix):
        # Convert the input matrix to a PyTorch tensor
        matrix_tensor = torch.tensor(tracklet_detection_matrix, dtype=torch.float32)

        # Find matched pairs (tracklet, detection)
        matched_pairs = []
        while matrix_tensor.sum() > 0:
            max_val, max_idx = torch.max(matrix_tensor, dim=1)
            tracklet_idx = torch.argmax(max_val)
            detection_idx = max_idx[tracklet_idx].item()

            if max_val[tracklet_idx] > 0:
                matched_pairs.append((tracklet_idx, detection_idx))
                matrix_tensor[tracklet_idx, :] = 0
                matrix_tensor[:, detection_idx] = 0

        # Find unmatched detections and tracklets
        unmatched_dets = torch.nonzero(matrix_tensor.sum(dim=0)).squeeze().tolist()
        unmatched_trks = torch.nonzero(matrix_tensor.sum(dim=1)).squeeze().tolist()

        return matched_pairs, unmatched_dets, unmatched_trks

    def build_simformer_batch(self, tracklets, detections, device, image):  # TODO UGLY - refactor
        T_max = np.array([len(t.detections) for t in tracklets]).max()
        batch = {
            "images": [torch.from_numpy(image)],
            'image_id': detections[0].image_id,  # int
            "det_feats": {
                'visibility_scores': torch.stack([det.visibility_scores for det in detections]).unsqueeze(1).unsqueeze(
                    0).to(device=device),  # [1, N, 1, 7]
                'embeddings': torch.stack([det.embeddings for det in detections]).unsqueeze(1).unsqueeze(
                    0).to(device=device),  # [1, N, 1, 7*D]
                'index': torch.IntTensor([det.pbtrack_id for det in detections]).unsqueeze(1).unsqueeze(0).to(
                    device=device),  # [1, N, 1]
                'bbox_conf': torch.stack([det.bbox_conf for det in detections]).unsqueeze(1).unsqueeze(1).unsqueeze(
                    0).to(device=device),  # [1, N, 1, 1]
                'bbox_ltwh': torch.stack([det.bbox_ltwh for det in detections]).unsqueeze(1).unsqueeze(0).to(
                    device=device),  # [1, N, 1, 4]
                'keypoints_xyc': torch.stack([det.keypoints_xyc for det in detections]).unsqueeze(1).unsqueeze(0).to(
                    device=device),  # [1, N, 1, 17, 3]
                'age': torch.zeros(len(detections), device=device).unsqueeze(1).unsqueeze(1).unsqueeze(0),
                # [B, N, 1, 17, 3]
            },
            'det_masks': torch.ones((1, len(detections), 1), device=device, dtype=torch.bool),  # [1, N, 1]
            "track_feats": {
                'visibility_scores': torch.stack(
                    [t.padded_features("visibility_scores", T_max) for t in tracklets]).unsqueeze(0).to(device=device),
                # [1, N, T, 7]
                'embeddings': torch.stack(
                    [t.padded_features("embeddings", T_max) for t in
                     tracklets]).unsqueeze(0).to(device=device),  # [1, N, T, 7*D]
                'index': torch.stack([t.padded_features("pbtrack_id", T_max) for t in tracklets]).unsqueeze(0).to(
                    device=device),  # [1, N, T]
                'bbox_conf': torch.stack([t.padded_features("bbox_conf", T_max) for t in tracklets]).unsqueeze(
                    2).unsqueeze(0).to(device=device),  # [1, N, T, 1]
                'bbox_ltwh': torch.stack([t.padded_features("bbox_ltwh", T_max) for t in tracklets]).unsqueeze(0).to(
                    device=device),  # [1, N, T, 4]
                'keypoints_xyc': torch.stack([t.padded_features("keypoints_xyc", T_max) for t in tracklets]).unsqueeze(
                    0).to(device=device),  # [1, N, T, 17, 3]
                'age': self.frame_count - 1 - torch.stack(
                    [t.padded_features("frame_idx", T_max) for t in tracklets]).unsqueeze(2).unsqueeze(0).to(
                    device=device),  # [1, N, T, 17, 3]
            },
            'track_masks': torch.stack([torch.cat([torch.ones(len(t.detections), dtype=torch.bool),
                                                   torch.zeros(T_max - len(t.detections), dtype=torch.bool)]) for t in
                                        tracklets]).unsqueeze(0).to(device=device),  # [1, N, T]
        }
        return batch

    def handle_camera_motion(self, frame):
        raise NotImplementedError()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if hasattr(self, "prev_frame") and len(self.tracklets) > 0:
            has_changed = detect_change_of_view(self.prev_frame, frame)
            if has_changed:
                pass
            else:
                pass
        self.prev_frame = frame

    def update_state(self, tracklet):
        s = tracklet.state
        if s == "init":
            if tracklet.hit_streak >= self.min_hits:
                new_state = "active"
            elif tracklet.time_wo_hits >= 1:
                new_state = "dead"
            else:
                new_state = "init"
        elif s == "active":
            if tracklet.time_wo_hits == 0:
                new_state = "active"
            elif tracklet.time_wo_hits < self.max_wo_hits:
                new_state = "lost"
            else:
                new_state = "dead"
        elif s == "lost":
            if tracklet.time_wo_hits == 0:
                new_state = "active"
            elif tracklet.time_wo_hits < self.max_wo_hits:
                new_state = "lost"
            else:
                new_state = "dead"
        elif s == "dead":
            new_state = "dead"
        else:
            raise ValueError(f"tracklet state is in undefined state {s}.")
        tracklet.state = new_state
