import numpy as np
import torch
import cv2 as cv

from scipy.optimize import linear_sum_assignment
from corrtrack.tracking.track_manager import TrackManager
from corrtrack.tracking.track_utils import get_reid_box_embeddings

def bbx_match_score(warped_corr_pose, 
                    curr_bbx, 
                    valid_corr_pose_joints, 
                    corr_threshold, 
                    num_valid_corr_pose_joints):
    match_score = 0
    points_inside = 0
    for j in range(17):
        if not bool(valid_corr_pose_joints[j]):
            continue

        if curr_bbx[0] <= warped_corr_pose[0, j] <= curr_bbx[2] \
        and curr_bbx[1] <= warped_corr_pose[1, j] <= curr_bbx[3]:
            if warped_corr_pose[2, j] >= corr_threshold:
                match_score += warped_corr_pose[2, j]
                points_inside += 1

    if points_inside > 0:
        match_score /= points_inside

    return match_score

def build_querie(pose):
    queries_u = np.zeros((1, 17, 3))
    queries_u[0, :, 0] = (pose[1, :] / 4).astype(np.uint)
    queries_u[0, :, 1] = (pose[0, :] / 4).astype(np.uint)
    queries_u[0, :, 2] = pose[2, :]

    return queries_u

def correlation_score(curr_pose, 
                      correlations, 
                      joint_threshold, 
                      corr_threshold):

    person_query = np.zeros((3, 17))
    person_query[:2] = curr_pose[:2] / 4
    person_query[2] = curr_pose[2]

    match_score = 0
    joint_count = 0
    for j in range(17):
        if person_query[2, j] >= joint_threshold:
            x, y = np.round(person_query[:2, j]).astype(int)

            x = min(x, correlations.shape[3] - 1)
            y = min(y, correlations.shape[2] - 1)

            corr = correlations[0, j, y, x].item()

            if corr > corr_threshold:
                match_score += corr
                joint_count += 1

    return match_score, joint_count

def compute_OKS(gt, 
                dt, 
                valid_joints, 
                bb, 
                area, 
                joint_threshold):
    sigmas = np.array([.26, .79, .79, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    vars = (sigmas * 2) ** 2
    k = len(sigmas)

    xg = gt[:, 0]
    yg = gt[:, 1]
    vg = valid_joints.astype(np.uint8)

    k1 = np.count_nonzero(vg > 0)

    x0 = bb[0] - bb[2];
    x1 = bb[0] + bb[2] * 2
    y0 = bb[1] - bb[3];
    y1 = bb[1] + bb[3] * 2

    xd = dt[:, 0]
    yd = dt[:, 1]

    if k1 > 0:
        # measure the per-keypoint distance if keypoints visible
        dx = xd - xg
        dy = yd - yg
    else:
        # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
        z = np.zeros((k))
        dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
        dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)
    e = (dx ** 2 + dy ** 2) / vars / (area.item() + np.spacing(1)) / 2
    if k1 > 0:
        e = e[vg > 0]

    OKS = np.sum(np.exp(-e)) / e.shape[0]
    joint_oks = np.zeros(17)
    joint_oks[vg > 0] = np.exp(-e)

    return OKS, joint_oks


def pose_match_score(curr_bbx, 
                     curr_pose, 
                     warped_corr_pose, 
                     corr_threshold, 
                     joint_threshold):
    area = (curr_bbx[2] - curr_bbx[0]) * (curr_bbx[3] - curr_bbx[1])
    pose_det = np.ones((17, 3))

    for j in range(17):
        if warped_corr_pose[2, j] > corr_threshold:
            pose_det[j, 0] = warped_corr_pose[0, j]
            pose_det[j, 1] = warped_corr_pose[1, j]
        else:
            pose_det[j, 0] = 10000
            pose_det[j, 1] = 10000

    valid_joints = (warped_corr_pose[2, :] > corr_threshold) & (curr_pose[2, :] >= joint_threshold)

    if np.count_nonzero(valid_joints) > 0:
        OKS, joint_oks = compute_OKS(curr_pose.transpose([1, 0]), 
                                     pose_det, 
                                     valid_joints, 
                                     curr_bbx, 
                                     area,
                                     joint_threshold)
    else:
        OKS = 0
        joint_oks = 0

    return OKS, joint_oks

def perform_tracking(tracks, 
                     affinities, 
                     affinities_clean, 
                     votes, 
                     frame_data, 
                     features, 
                     joint_threshold, 
                     ann_ids,
                     min_keypoints, 
                     f, 
                     max_detections_per_image, 
                     duplicate_ratio, 
                     break_tracks,
                     seq_net, 
                     similarity_threshold, 
                     min_refinement_track_len,
                     min_consecutive_len,
                     images=None):

    track_manager = TrackManager.get_instance()
    frame_kpts = frame_data['kpts']
    frame_anno_ids = frame_data['anno_ids']
    frame_bboxes = frame_data['bbs']

    tracked = np.zeros(len(tracks))
    assert break_tracks and "Ensure that we break tracks here"

    # extract reid features for all detections in the current frame
    bbox_embeddings, valid_boxes_inds = get_reid_box_embeddings(frame_bboxes, 
                                                                seq_net)
    inactive_tracks_before = None 

    for k in range(max_detections_per_image):
        best_corr = 0
        best_t = -1
        ann_id = int(frame_anno_ids[k].item())

        if ann_ids[ann_id] == 1:
            continue

        # find track with highest score
        for t_idx in range(len(tracks)):
            if break_tracks and tracks[t_idx]['continue'] == 0:
                continue

            corr = affinities_clean[t_idx, k]
            if corr > best_corr:
                if tracked[t_idx] == 0:
                    best_corr = corr
                    best_t = t_idx

        # update best matching track
        if best_t >= 0 and best_corr > 0:
            tracked[best_t] = 1
            ann_ids[ann_id] = 1

            queries_u = build_querie(frame_kpts[k])

            embedding_idx = np.argwhere(valid_boxes_inds[:, 0] == k)[0]
            box_embedding = bbox_embeddings[embedding_idx]

            # update matched track
            track_manager.update_track(best_t,
                                       queries_u,
                                       frame_kpts[k],
                                       ann_id,
                                       features[f],
                                       f,
                                       np.count_nonzero(frame_kpts[k][2] >= joint_threshold),
                                       frame_bboxes[k],
                                       box_embedding)

    # deactivate all unmatched tracks
    for t_idx in range(len(tracks)):
        if tracked[t_idx] == 0:
            tracks[t_idx]['continue'] = 0

    # check for duplicates
    duplicates = np.zeros(max_detections_per_image)

    if np.count_nonzero(np.count_nonzero(votes, axis=1) > 1):
        # get all track idxs for which we might have duplicates
        track_idxs = np.argwhere(np.count_nonzero(votes, axis=1) > 1)

        for track_idx in track_idxs:
            track_votes = votes[track_idx[0]]

            suspicious_poses = np.argwhere(track_votes > 0)[:, 0]

            best_k = np.argmax(affinities_clean[track_idx])
            best_pose_score = affinities[2, track_idx[0], best_k]
            best_bbx_score = affinities[0, track_idx[0], best_k]
            best_corr_score = affinities[1, track_idx[0], best_k]

            for pose_idx in suspicious_poses:
                suspicious_ann_id = int(frame_anno_ids[pose_idx].item())
                if ann_ids[suspicious_ann_id] == 1:
                    continue

                pose_score = affinities[2, track_idx[0], pose_idx]
                bbx_score = affinities[0, track_idx[0], pose_idx]
                corr_score = affinities[1, track_idx[0], pose_idx]

                pose_ratio = 0
                bbx_ratio = 0
                corr_ratio = 0

                if best_pose_score > 0:
                    pose_ratio = pose_score / best_pose_score
                if best_corr_score > 0:
                    corr_ratio = corr_score / best_corr_score
                avg_ratio = (pose_ratio + corr_ratio)

                # We have a duplicate
                if avg_ratio > duplicate_ratio:
                    duplicates[pose_idx] = 1

    ############################
    # Get unmatched detections #
    ############################

    unmatched_detections = []
    for k in range(max_detections_per_image):
        if duplicates[k] == 1:
            continue

        ann_id = int(frame_anno_ids[k].item())
        if ann_ids[ann_id] == 1:
            continue

        if frame_kpts[k][2].sum() == -17:
            continue

        num_valid_keypoints = np.count_nonzero(frame_kpts[k][2] >= joint_threshold)
        if num_valid_keypoints < min_keypoints:
            continue

        unmatched_detections.append(k)

    ##############################
    # Match with inactive tracks #
    ##############################
    if len(unmatched_detections) > 0:
        inactive_tracks = track_manager.get_inactive_tracks(min_track_length=min_refinement_track_len)

        # Try to associate inactive tracks with unmatched detections
        if len(inactive_tracks) > 0:
            inactive_track_features = []
            track_features = [] 
            for t_idx, t in inactive_tracks:
                t_features = torch.cat(list(t['reid_features']))
                t_features = t_features.mean(dim=0)
                inactive_track_features.append(t_features)

            inactive_track_features = torch.stack(inactive_track_features)
            detection_features = []
            for k in unmatched_detections:
                embedding_idx = np.argwhere(valid_boxes_inds[:, 0] == k)[0]
                detection_features.append(bbox_embeddings[embedding_idx])
            detection_features = torch.cat(detection_features)

            inactive_sim = torch.einsum('mc,nc->mn', detection_features, inactive_track_features)
            inactive_sim = inactive_sim.cpu().numpy()

            source_idxs, target_idxs = linear_sum_assignment(inactive_sim, maximize=True)

            detections_to_remove = []
            for s_idx, tgt_idx in zip(source_idxs, target_idxs):
                if inactive_sim[s_idx, tgt_idx] > similarity_threshold:

                    k = unmatched_detections[s_idx]
                    track_idx, track = inactive_tracks[tgt_idx]
                    detections_to_remove.append(k)

                    queries_u = build_querie(frame_kpts[k])
                    ann_id = int(frame_anno_ids[k].item())
                    ann_ids[ann_id] = 1
                    
                    embedding_idx = np.argwhere(valid_boxes_inds[:, 0] == k)[0]
                    box_embedding = bbox_embeddings[embedding_idx]

                    # update matched track
                    track_manager.update_track(track_idx,
                                               queries_u,
                                               frame_kpts[k],
                                               ann_id,
                                               features[f],
                                               f,
                                               np.count_nonzero(frame_kpts[k][2] >= joint_threshold),
                                               frame_bboxes[k],
                                               box_embedding)
                    tracked[track_idx] = 1
            
            for k in detections_to_remove:
                unmatched_detections.remove(k)

    #####################
    # Create new tracks #
    #####################
    if len(unmatched_detections) > 0:
        # if we still have unmatched detections, initialize new tracks
        for k in unmatched_detections:
            queries_u = build_querie(frame_kpts[k])
            ann_id = int(frame_anno_ids[k].item())
            ann_ids[ann_id] = 1    

            embedding_idx = np.argwhere(valid_boxes_inds[:, 0] == k)[0]
            box_embedding = bbox_embeddings[embedding_idx]

            track = track_manager.new_track(queries=queries_u,
                                            kpts=frame_kpts[k],
                                            ann_id=ann_id,
                                            features= features[f],
                                            curr_frame=f,
                                            num_kpts=np.count_nonzero(frame_kpts[k][2] >= joint_threshold),
                                            bbx=frame_bboxes[k],
                                            reid_features=box_embedding)
            track_manager.add(track)

    track_manager.increment_inactive_counter_and_kill()
    tracks = track_manager.get_tracks()    
    return track_manager.get_tracks()


def get_pose_from_correlations(correlations, 
                               frame_warps, 
                               res_h, 
                               res_w, 
                               corr_threshold):
    correlations_vec = correlations.view(17, res_h * res_w)

    val_k, ind = correlations_vec.topk(1, dim=1)
    xs = ind % res_w
    ys = (ind / res_w).long()
    val_k, xs, ys = val_k.numpy(), xs.numpy(), ys.numpy()

    corr_pose = np.ones([3, 17])
    corr_pose[0] = xs[:, 0] * 4
    corr_pose[1] = ys[:, 0] * 4

    warped_corr_pose = np.matmul(frame_warps, corr_pose)

    corr_pose[2] = val_k[:, 0]
    warped_corr_pose[2] = val_k[:, 0]

    invalid_corr_joints = val_k[:, 0] < corr_threshold

    warped_corr_pose[:, invalid_corr_joints] = 0
    corr_pose[:, invalid_corr_joints] = 0

    return warped_corr_pose, corr_pose


def get_affinities(tracks, 
                   features, 
                   model, 
                   frame_data,
                   fr_idx,
                   break_tracks, 
                   joint_threshold, 
                   corr_threshold, 
                   oks_threshold,
                   res_h, 
                   res_w, 
                   min_keypoints, 
                   max_detections_per_image, 
                   images=None):
    vis = False

    frame_kpts = frame_data['kpts']
    original_frame_kpts = frame_data['original_kpts']

    frame_warps = frame_data['warps']
    frame_bboxes = frame_data['bbs']

    affinities = np.zeros([3, len(tracks), max_detections_per_image])
    affinities_clean = np.zeros([len(tracks), max_detections_per_image])
    votes = np.zeros([len(tracks), max_detections_per_image])

    bbox_affinities = np.zeros((len(tracks), max_detections_per_image))
    pose_affinities = np.zeros((len(tracks), max_detections_per_image))
    corr_affinities = np.zeros((len(tracks), max_detections_per_image, 2))

    for t_idx, track in enumerate(tracks):
        # don't track if track is broken
        if break_tracks and track['continue'] == 0:
            continue

        fA = track['features']
        fA = fA.view(1, 32, res_h, res_w)
        queries = track['queries']

        invalid_query_joints = queries[0, :, 2] < joint_threshold

        with torch.no_grad():
            output = model(fA, features[fr_idx].view(1, 32, res_h, res_w), queries, False, True)
            correlations = torch.sigmoid(output).data.cpu()

            # get pose in original image
            warped_corr_pose, corr_pose = get_pose_from_correlations(correlations, frame_warps, res_h, res_w,
                                                                     corr_threshold)

            # supress invalid correspondences
            warped_corr_pose[:, invalid_query_joints] = 0
            corr_pose[:, invalid_query_joints] = 0

        valid_corr_pose_joints = warped_corr_pose[2] >= corr_threshold
        num_valid_corr_pose_joints = np.count_nonzero(valid_corr_pose_joints)

        for k in range(max_detections_per_image):
            curr_original_pose = original_frame_kpts[k]
            curr_pose = frame_kpts[k]

            # we only measure, if curr pose has enough valid joints
            num_curr_valid_joints = np.count_nonzero(curr_pose[2] >= joint_threshold)
            if num_curr_valid_joints < min_keypoints:
                continue

            # bbox in original image
            curr_bbx = frame_bboxes[k]

            # measure how many correspondence points are located within bounding box of
            # person k

            if num_valid_corr_pose_joints >= min_keypoints:
                bbox_affinities[t_idx, k] = bbx_match_score(warped_corr_pose, curr_bbx,
                                                            valid_corr_pose_joints, corr_threshold,
                                                            num_valid_corr_pose_joints)

            # measure the oks similarity of pose k and valid correspondences
            pose_affinities[t_idx, k], _ = pose_match_score(curr_bbx, curr_original_pose, warped_corr_pose,
                                                            corr_threshold, joint_threshold)

            # measure correspondence affinities based on person k
            corr_affinities[t_idx, k] = correlation_score(curr_pose, correlations, joint_threshold, corr_threshold)

        # we want to find duplicates

        votes[t_idx, bbox_affinities[t_idx] > 0] += 1
        votes[t_idx, corr_affinities[t_idx, :, 0] > 0] += 1
        votes[t_idx, pose_affinities[t_idx] > oks_threshold] += 1
        affinities[1, t_idx] = corr_affinities[t_idx, :, 0]

    for t_idx in range(len(tracks)):

        relevant_poses = np.argwhere(bbox_affinities[t_idx] > corr_threshold)
        max_score = 0
        best_k = -1

        for k in relevant_poses:
            pose_score = pose_affinities[t_idx, k[0]]
            corr_score, num_kpts = corr_affinities[t_idx, k[0]]

            affinities[0, t_idx, k[0]] = bbox_affinities[t_idx, k[0]]
            affinities[2, t_idx, k[0]] = pose_score

            if num_kpts == 0:
                # prevent division by zero
                num_kpts = 1
            score = pose_score + corr_score / num_kpts

            if score > max_score:
                max_score = score
                best_k = k[0]

        if pose_affinities[t_idx, best_k] > oks_threshold and corr_affinities[
            t_idx, best_k, 0] > corr_threshold * min_keypoints:
            affinities_clean[t_idx, best_k] = max_score

    return affinities_clean, affinities, votes
