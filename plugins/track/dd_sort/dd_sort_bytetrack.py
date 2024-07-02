import torch
import logging

import numpy as np


log = logging.getLogger(__name__)


class Detection:
    def __init__(self, image_id, features, score, pbtrack_id, frame_id):
        for k, v in features.items():
            setattr(self, k, v)
        self.score = score
        self.pbtrack_id = pbtrack_id
        self.image_id = image_id
        self.frame_id = torch.tensor(frame_id)
        self.similarity_with_tracklet = None
        self.similarities = None


class TrackState(object):
    Init = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class Tracklet(object):
    _count = 0

    def __init__(self, detection, frame_id):
        """Init state"""
        self.last_det = detection
        self.detections = [detection]
        self.score = detection.score
        self.start_frame = frame_id
        self.frame_id = frame_id
        self.state = TrackState.Init
        self.track_id = -1

    def activate(self, detection, frame_id):
        """from Init to Tracked state"""
        self.track_id = self.next_id()
        self.update(detection, frame_id)

    def update(self, detection, frame_id):
        """Stays in Tracked state"""
        self.last_det = detection
        self.detections.append(detection)
        self.score = detection.score
        self.frame_id = frame_id

        self.state = TrackState.Tracked

    def re_activate(self, detection, frame_id, new_id=False):
        """from Lost to Tracked state"""
        self.last_det = detection
        self.detections.append(detection)
        self.score = detection.score
        self.frame_id = frame_id

        self.state = TrackState.Tracked
        if new_id:
            self.track_id = self.next_id()

    def padded_features(self, name, size):
        features = torch.stack([getattr(det, name) for det in reversed(self.detections)])
        if features.shape[0] < size:
            features = torch.cat(
                [
                    features,
                    torch.zeros(
                        size - features.shape[0], *features.shape[1:], device=features.device
                    )
                    + float("nan"),
                ]
            )
        return features

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        Tracklet._count += 1
        return Tracklet._count

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    def __repr__(self):
        return f"ID{self.track_id}-(t:{self.start_frame}->{self.end_frame})"


class DDSORTBYTETracker(object):
    def __init__(
        self,
        simformer,
        det_high_thresh=0.4,
        det_low_thresh=0.1,
        max_time_lost=30,
        simformer_call_strat: str = "3call",
        **kwargs,
    ):
        self.init = []  # type: list[Tracklet]
        self.tracks = []  # type: list[Tracklet]
        self.lost = []  # type: list[Tracklet]
        self.removed = []  # type: list[Tracklet]

        self.simformer = simformer.eval()
        self.det_high_thresh = det_high_thresh
        self.det_low_thresh = det_low_thresh
        self.max_time_lost = max_time_lost

        if simformer_call_strat == "3call":
            self.simformer_call_strat = self.three_call_simformer
        elif simformer_call_strat == "1call":
            self.simformer_call_strat = self.one_call_simformer
        else:
            raise NotImplementedError(f"Unknown simformer_call_strat: {simformer_call_strat}")

        self.frame_id = 0

    def update(self, features, pbtrack_ids, image_id, image):
        self.frame_id += 1
        # fixme
        # bytetrack does not use Init tracks for the first frame (only!, after it is normal)
        # they are directly used as actived tracks

        # classify detections into 2 categories : high and low score
        # for the baseline use `bbox_conf` as in ByteTrack
        # fixme check if conf from cls head is better
        dets_high = []
        dets_low = []
        for i in range(len(pbtrack_ids)):
            features_i = {k: v[0, i] for k, v in features.items()}
            score = features_i["bbox_conf"]
            if score >= self.det_high_thresh:
                dets_high.append(
                    Detection(
                        image_id, features_i, score, pbtrack_ids[i], frame_id=self.frame_id - 1
                    )
                )
            elif score >= self.det_low_thresh:
                dets_low.append(
                    Detection(
                        image_id, features_i, score, pbtrack_ids[i], frame_id=self.frame_id - 1
                    )
                )

        # sanity check on tracklets
        self.check_tracks()

        self.simformer_call_strat(dets_high, dets_low, image)

        # Update and clean the lists of tracks
        self.clean_tracks()

        # Output results
        tracks = []
        for trk in self.tracks:
            tracks.append(
                {
                    "pbtrack_id": trk.last_det.pbtrack_id.item(),
                    "track_id": trk.track_id,
                    "hits": len(trk.detections),
                    "age": self.frame_id - trk.start_frame,
                    "matched_with": (
                        "S",
                        trk.last_det.similarity_with_tracklet.cpu().numpy()
                        # if trk.last_det.similarity_with_tracklet
                        # else 0,
                    ),
                    "time_since_update": self.frame_id - trk.end_frame,
                    "state": trk.state,
                    "costs": {
                        "S": {
                            track_id: sim
                            for track_id, sim in zip(
                                trk.last_det.similarities_track_idx,
                                trk.last_det.similarities.cpu().numpy(),
                            )
                        },
                        "St": self.simformer.sim_threshold,
                    },
                }
            )
        return tracks

    def three_call_simformer(self, dets_high, dets_low, image):
        # First association: dets with high score and active tracks + lost
        first_tracks_pool = self.tracks + self.lost
        (
            matches,
            u_track,
            u_dets_high,
            td_sim_matrix,
        ) = self.associate_dets_to_trks(first_tracks_pool, dets_high, image)  # bytetrack use high threshold (0.7-0.9)

        similarities_track_idx = [t.track_id for t in first_tracks_pool]
        for itracked, idet in matches:
            track = first_tracks_pool[itracked]
            det = dets_high[idet]
            det.similarity_with_tracklet = td_sim_matrix[itracked, idet]
            det.similarities = td_sim_matrix[:, idet]
            det.similarities_track_idx = similarities_track_idx
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
            elif track.state == TrackState.Lost:
                track.re_activate(det, self.frame_id, new_id=False)
                self.tracks.append(track)
            else:
                raise ValueError(f"Unexpected track state: `{track.state}`")
        # Second association: dets low with remaining active (not lost) tracklets from last pool
        second_tracks_pool = [
            first_tracks_pool[i]
            for i in u_track
            if first_tracks_pool[i].state == TrackState.Tracked
        ]
        (
            matches,
            u_track,
            u_dets_low,
            td_sim_matrix,
        ) = self.associate_dets_to_trks(second_tracks_pool, dets_low, image)  # bytetrack use low threshold (0.5)

        similarities_track_idx = [t.track_id for t in second_tracks_pool]
        for itrack, idet in matches:
            track = second_tracks_pool[itrack]
            det = dets_low[idet]
            det.similarity_with_tracklet = td_sim_matrix[itrack, idet]
            det.similarities = td_sim_matrix[:, idet]
            det.similarities_track_idx = similarities_track_idx
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
            elif track.state == TrackState.Lost:
                track.re_activate(det, self.frame_id, new_id=False)
                self.tracks.append(track)
            else:
                raise ValueError(f"Unexpected track state: `{track.state}`")
        # Mark as lost tracks that were not matched during first and second round association
        for itrack in u_track:
            track = second_tracks_pool[itrack]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                self.lost.append(track)
        # Third association: dets high remaining with init tracks
        init_tracks_pool = self.init
        dets_remain = [dets_high[i] for i in u_dets_high]
        (
            matches,
            u_unconfirmed,
            u_dets_remain,
            td_sim_matrix,
        ) = self.associate_dets_to_trks(init_tracks_pool, dets_remain, image)  # bytetrack use high threshold (0.7)

        for itrack, idet in matches:
            track = init_tracks_pool[itrack]
            det = dets_remain[idet]
            det.similarity_with_tracklet = td_sim_matrix[itrack, idet]
            det.similarities = td_sim_matrix[:, idet]
            det.similarities_track_idx = [t.track_id for t in init_tracks_pool]
            track.activate(det, self.frame_id)
            self.tracks.append(track)
        # Init tracks not matched with any detection are marked as removed
        for itrack in u_unconfirmed:
            track = init_tracks_pool[itrack]
            track.mark_removed()
            # self.removed.append(track)  # fixme do we want to log the not confirmed tracks?
        # Remaining high score detections are used to create new tracks in init state
        for idet in u_dets_remain:
            track = Tracklet(dets_remain[idet], self.frame_id)
            self.init.append(track)
        # Remove tracks that has not been matched for too long
        for track in self.lost:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                self.removed.append(track)

    def one_call_simformer(self, dets_high, dets_low, image):
        tracklets = self.tracks + self.lost + self.init
        detections = dets_high + dets_low

        # get general sim_matrix
        if len(tracklets) > 0 and len(detections) > 0:
            batch = build_simformer_batch(
                tracklets, detections, self.simformer.device, image, self.frame_id
            )
            tracks, dets = self.simformer.predict_preprocess(batch)
            _, _, td_sim_matrix = self.simformer.forward(tracks, dets)
        else:
            td_sim_matrix = torch.empty((1, len(tracklets), len(detections)), device=self.simformer.device)

        first_tracks_mask = torch.cat(
            (
                torch.ones(1, len(self.tracks)),
                torch.ones(1, len(self.lost)),
                torch.zeros(1, len(self.init)),
            ),
            dim=1,
        ).to(dtype=torch.bool, device=self.simformer.device)
        second_tracks_mask = torch.cat(
            (
                torch.ones(1, len(self.tracks)),
                torch.zeros(1, len(self.lost)),
                torch.zeros(1, len(self.init)),
            ),
            dim=1,
        ).to(dtype=torch.bool, device=self.simformer.device)
        third_tracks_mask = torch.cat(
            (
                torch.zeros(1, len(self.tracks)),
                torch.zeros(1, len(self.lost)),
                torch.ones(1, len(self.init)),
            ),
            dim=1,
        ).to(dtype=torch.bool, device=self.simformer.device)
        first_dets_mask = torch.cat(
            (torch.ones(1, len(dets_high)), torch.zeros(1, len(dets_low))), dim=1
        ).to(dtype=torch.bool, device=self.simformer.device)
        second_dets_mask = torch.cat(
            (torch.zeros(1, len(dets_high)), torch.ones(1, len(dets_low))), dim=1
        ).to(dtype=torch.bool, device=self.simformer.device)
        third_dets_mask = torch.cat(
            (torch.ones(1, len(dets_high)), torch.zeros(1, len(dets_low))), dim=1
        ).to(dtype=torch.bool, device=self.simformer.device)
        threshold = (
            self.simformer.sim_threshold
            if self.simformer.sim_threshold
            else self.simformer.computed_sim_threshold
        )

        # First association: dets with high score and active tracks + lost
        _, association_result = self.simformer.association(
            td_sim_matrix, first_tracks_mask, first_dets_mask, sim_threshold=threshold
        )
        first_matches = association_result[0]["matched_td_indices"]
        u_track = association_result[0]["unmatched_trackers"]
        u_dets_high = association_result[0]["unmatched_detections"]

        similarities_track_idx = [t.track_id for t in self.tracks + self.lost]
        for itracked, idet in first_matches:
            track = tracklets[itracked]
            det = detections[idet]
            det.similarity_with_tracklet = td_sim_matrix[0, itracked, idet]
            det.similarities = td_sim_matrix[0, :, idet]
            det.similarities_track_idx = similarities_track_idx
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
            elif track.state == TrackState.Lost:
                track.re_activate(det, self.frame_id, new_id=False)
                self.tracks.append(track)
            else:
                raise ValueError(f"Unexpected track state: `{track.state}`")
            second_tracks_mask[0, itracked] = False
            third_dets_mask[0, idet] = False

        # Second association: dets low with remaining active (not lost) tracklets from last pool
        _, association_result = self.simformer.association(
            td_sim_matrix, second_tracks_mask, second_dets_mask, sim_threshold=threshold
        )
        second_matches = association_result[0]["matched_td_indices"]
        u_track = association_result[0]["unmatched_trackers"]
        u_dets_low = association_result[0]["unmatched_detections"]
        for itrack, idet in second_matches:
            track = tracklets[itrack]
            det = detections[idet]
            det.similarity_with_tracklet = td_sim_matrix[0, itrack, idet]
            det.similarities = td_sim_matrix[0, :, idet]
            det.similarities_track_idx = similarities_track_idx
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
            elif track.state == TrackState.Lost:
                track.re_activate(det, self.frame_id, new_id=False)
                self.tracks.append(track)
            else:
                raise ValueError(f"Unexpected track state: `{track.state}`")
        # Mark as lost tracks that were not matched during first and second round association
        for itrack in u_track:
            track = tracklets[itrack]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                self.lost.append(track)

        # Third association: dets high remaining with init tracks
        _, association_result = self.simformer.association(
            td_sim_matrix, third_tracks_mask, third_dets_mask, sim_threshold=threshold
        )
        third_matches = association_result[0]["matched_td_indices"]
        u_unconfirmed = association_result[0]["unmatched_trackers"]
        u_dets_remain = association_result[0]["unmatched_detections"]
        for itrack, idet in third_matches:
            track = tracklets[itrack]
            det = detections[idet]
            det.similarity_with_tracklet = td_sim_matrix[0, itrack, idet]
            det.similarities = td_sim_matrix[0, :, idet]
            det.similarities_track_idx = [t.track_id for t in self.init]
            track.activate(det, self.frame_id)
            self.tracks.append(track)
        # Init tracks not matched with any detection are marked as removed
        for itrack in u_unconfirmed:
            track = tracklets[itrack]
            track.mark_removed()
            # self.removed.append(track)  # fixme do we want to log the not confirmed tracks?
        # Remaining high score detections are used to create new tracks in init state
        for idet in u_dets_remain:
            track = Tracklet(detections[idet], self.frame_id)
            self.init.append(track)
        # Remove tracks that has not been matched for too long
        for track in self.lost:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                self.removed.append(track)

    def check_tracks(self):
        for track in self.tracks:
            assert track.state == TrackState.Tracked, f"Unexpected track state: `{track.state}`"
        for track in self.init:
            assert track.state == TrackState.Init, f"Unexpected track state: `{track.state}`"
        for track in self.lost:
            assert track.state == TrackState.Lost, f"Unexpected track state: `{track.state}`"
        for track in self.removed:
            assert track.state == TrackState.Removed, f"Unexpected track state: `{track.state}`"

    def clean_tracks(self):
        self.init = [t for t in self.init if t.state == TrackState.Init]
        self.tracks = [t for t in self.tracks if t.state == TrackState.Tracked]
        self.lost = [t for t in self.lost if t.state == TrackState.Lost]
        self.removed = [t for t in self.removed if t.state == TrackState.Removed]

    def associate_dets_to_trks(self, tracklets, detections, image):
        """
        fixme: for the moment the association is done in 3 stages (high, low, init) to be as in the baseline bytetrack
        we should add a one stage inference on simformer (high + low + init) than handle detections
        as would bytetrack do in 3 stages based on the same rules (high, low, init)
        """
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
                np.arange(len(tracklets)),
                np.empty((0,)),
                np.empty((0,)),
            )

        batch = build_simformer_batch(tracklets, detections, self.simformer.device, image, self.frame_id)
        association_matrix, association_result, td_sim_matrix = self.simformer.predict_step(
            batch, self.frame_id
        )
        matched = association_result[0]["matched_td_indices"]
        unmatched_trks = association_result[0]["unmatched_trackers"]
        unmatched_dets = association_result[0]["unmatched_detections"]
        return matched, unmatched_trks, unmatched_dets, td_sim_matrix.squeeze(0)


def build_simformer_batch(tracklets, detections, device, image, frame_count):
    T_max = np.array([len(t.detections) for t in tracklets]).max()
    batch = {
        "images": [torch.from_numpy(image)],
        "image_id": detections[0].image_id,  # int
        "det_feats": {
            "visibility_scores": torch.stack([det.visibility_scores for det in detections])
            .unsqueeze(1)
            .unsqueeze(0)
            .to(device=device),  # [1, N, 1, 7]
            "embeddings": torch.stack([det.embeddings.flatten() for det in detections])  # FIXME no flatten
            .unsqueeze(1)
            .unsqueeze(0)
            .to(device=device),  # [1, N, 1, 7*D]
            "index": torch.IntTensor([det.pbtrack_id for det in detections])
            .unsqueeze(1)
            .unsqueeze(0)
            .to(device=device),  # [1, N, 1]
            "bbox_conf": torch.stack([det.bbox_conf for det in detections])
            .unsqueeze(1)
            .unsqueeze(1)
            .unsqueeze(0)
            .to(device=device),  # [1, N, 1, 1]
            "bbox_ltwh": torch.stack([det.bbox_ltwh for det in detections])
            .unsqueeze(1)
            .unsqueeze(0)
            .to(device=device),  # [1, N, 1, 4]
            "keypoints_xyc": torch.stack([det.keypoints_xyc for det in detections])
            .unsqueeze(1)
            .unsqueeze(0)
            .to(device=device),  # [B, N, 1, 17, 3]
            'age': torch.zeros(len(detections), device=device).unsqueeze(1).unsqueeze(1).unsqueeze(0),
            # [B, N, 1, 17, 3]
        },
        "det_masks": torch.ones(
            (1, len(detections), 1), device=device, dtype=torch.bool
        ),  # [1, N, 1]
        "track_feats": {
            "visibility_scores": torch.stack(
                [t.padded_features("visibility_scores", T_max) for t in tracklets]
            )
            .unsqueeze(0)
            .to(device=device),  # [1, N, T, 7]
            "embeddings": torch.stack(
                [
                    t.padded_features("embeddings", T_max).flatten(start_dim=1, end_dim=2)
                    for t in tracklets
                ]
            )
            .unsqueeze(0)
            .to(device=device),  # [1, N, T, 7*D]
            "index": torch.stack([t.padded_features("pbtrack_id", T_max) for t in tracklets])
            .unsqueeze(0)
            .to(device=device),  # [1, N, T]
            "bbox_conf": torch.stack([t.padded_features("bbox_conf", T_max) for t in tracklets])
            .unsqueeze(2)
            .unsqueeze(0)
            .to(device=device),  # [1, N, T, 1]
            "bbox_ltwh": torch.stack([t.padded_features("bbox_ltwh", T_max) for t in tracklets])
            .unsqueeze(0)
            .to(device=device),  # [1, N, T, 4]
            "keypoints_xyc": torch.stack(
                [t.padded_features("keypoints_xyc", T_max) for t in tracklets]
            )
            .unsqueeze(0)
            .to(device=device),  # [B, N, T, 17, 3]
            'age': frame_count - torch.stack([t.padded_features("frame_id", T_max) for t in tracklets]).unsqueeze(2).unsqueeze(0).to(device=device),  # [1, N, T, 17, 3]
        },
        "track_masks": torch.stack(
            [
                torch.cat(
                    [
                        torch.ones(len(t.detections), dtype=torch.bool),
                        torch.zeros(T_max - len(t.detections), dtype=torch.bool),
                    ]
                )
                for t in tracklets
            ]
        )
        .unsqueeze(0)
        .to(device=device),  # [1, N, T]
    }
    return batch
