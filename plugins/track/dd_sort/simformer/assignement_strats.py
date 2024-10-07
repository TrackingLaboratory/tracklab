import numpy as np
import torch

from scipy.optimize import linear_sum_assignment


def hungarian_algorithm(td_sim_matrix, valid_tracks, valid_dets, sim_threshold=0.0):
    """
    apply hungarian algorithm on sim_matrix with the entries in valid_dets and valid_tracks

    td_sim_matrix: float32 tensor [B, T, D]
    valid_tracks: bool tensor [B, T]
        True is valid False otherwise
    valid_dets: bool tensor [B, D]
        True is valid False otherwise
    :return: association_matrix: bool tensor [B, T, D]
        True if the pair value is associated False otherwise
    """
    B, T, D = td_sim_matrix.shape
    association_matrix = torch.zeros_like(td_sim_matrix, dtype=torch.bool)
    association_result = []
    for b in range(B):
        if valid_tracks[b].sum() > 0 and valid_dets[b].sum() > 0:
            sim_matrix_masked = td_sim_matrix[b, valid_tracks[b], :][:, valid_dets[b]]
            # sim_matrix_masked[sim_matrix_masked < sim_threshold] = 0.0  # work less well than below
            sim_matrix_masked[sim_matrix_masked < sim_threshold] = sim_threshold - 1e-5  # it is done like this in BPBreIDStrongSORT
            row_idx, col_idx = linear_sum_assignment(-sim_matrix_masked.cpu())

            valid_rows = torch.nonzero(valid_tracks[b]).squeeze(dim=1).cpu()
            valid_cols = torch.nonzero(valid_dets[b]).squeeze(dim=1).cpu()
            matched_td_indices = np.array(list(zip(valid_rows[row_idx], valid_cols[col_idx])))

            unmatched_trackers = [t.item() for t in valid_rows if t.item() not in set(valid_rows[row_idx].tolist())]
            unmatched_detections = [d.item() for d in valid_cols if d.item() not in set(valid_cols[col_idx].tolist())]

            matches = []
            for m in matched_td_indices:
                if td_sim_matrix[b, m[0], m[1]] < sim_threshold:
                    unmatched_trackers.append(m[0])
                    unmatched_detections.append(m[1])
                else:
                    association_matrix[b, m[0], m[1]] = True
                    matches.append(m)
            matched_td_indices = np.array(matches)
        else:
            matched_td_indices = np.empty((0, 2), dtype=int)
            unmatched_trackers = []
            unmatched_detections = []
            if valid_tracks[b].sum() > 0:
                unmatched_trackers = torch.nonzero(valid_tracks[b]).squeeze(dim=1).tolist()
            elif valid_dets[b].sum() > 0:
                unmatched_detections = torch.nonzero(valid_dets[b]).squeeze(dim=1).tolist()
        association_result.append(
            {
                "matched_td_indices": matched_td_indices,
                "unmatched_trackers": unmatched_trackers,
                "unmatched_detections": unmatched_detections,
            }
        )
    # print("association_matrix")
    # print(association_matrix.cpu().numpy())
    # print("association_result")
    # print(association_result[0]["matched_td_indices"].tolist())
    # print(association_result[0]["unmatched_trackers"])
    # print(association_result[0]["unmatched_detections"])
    return association_matrix, association_result


def greedy_assignment(td_sim_matrix, threshold):
    """
    Greedy assignment method for tracklet-detection association.

    Args:
    - td_sim_matrix (np.ndarray): A matrix where each entry (i,j) represents the similarity
      between tracklet i and detection j.
    - threshold (float): A threshold value above which an association is considered valid.

    Returns:
    - assignments (list of tuples): A list where each tuple (i,j) indicates that
      tracklet i is assigned to detection j.
    """

    # List to store the assignments
    assignments = []

    # Convert matrix to list of tuples (similarity, tracklet_index, detection_index)
    pairs = [(td_sim_matrix[i, j], i, j) for i in range(td_sim_matrix.shape[0]) for j in range(td_sim_matrix.shape[1])]
    # Sort pairs by similarity in descending order
    pairs.sort(key=lambda x: x[0], reverse=True)

    assigned_tracklets = set()
    assigned_detections = set()

    for sim, i, j in pairs:
        if sim < threshold:
            break
        if i not in assigned_tracklets and j not in assigned_detections:
            assignments.append((i, j))
            assigned_tracklets.add(i)
            assigned_detections.add(j)

    return np.array(assignments) if len(assignments) > 0 else np.empty((0, 2), dtype=int)


def argmax_algorithm(td_sim_matrix, valid_tracks, valid_dets, sim_threshold=0.0):
    B, T, D = td_sim_matrix.shape
    association_matrix = torch.zeros_like(td_sim_matrix, dtype=torch.bool)
    association_result = []
    for b in range(B):
        if valid_tracks[b].sum() > 0 and valid_dets[b].sum() > 0:
            sim_matrix_masked = td_sim_matrix[b, valid_tracks[b], :][:, valid_dets[b]]
            sim_matrix_masked[sim_matrix_masked < sim_threshold] = 0.0
            row_idx, col_idx = argmax_assignment(sim_matrix_masked.cpu())

            valid_rows = torch.nonzero(valid_tracks[b]).squeeze(dim=1).cpu()
            valid_cols = torch.nonzero(valid_dets[b]).squeeze(dim=1).cpu()
            matched_td_indices = np.array(list(zip(valid_rows[row_idx], valid_cols[col_idx])))

            unmatched_trackers = [t.item() for t in valid_rows if t.item() not in set(valid_rows[row_idx].tolist())]
            unmatched_detections = [d.item() for d in valid_cols if d.item() not in set(valid_cols[col_idx].tolist())]

            matches = []
            for m in matched_td_indices:
                if td_sim_matrix[b, m[0], m[1]] < sim_threshold:
                    unmatched_trackers.append(m[0])
                    unmatched_detections.append(m[1])
                else:
                    association_matrix[b, m[0], m[1]] = True
                    matches.append(m)
            matched_td_indices = np.array(matches)
        else:
            matched_td_indices = np.empty((0, 2), dtype=int)
            unmatched_trackers = []
            unmatched_detections = []
            if valid_tracks[b].sum() > 0:
                unmatched_trackers = torch.nonzero(valid_tracks[b]).squeeze(dim=1).tolist()
            elif valid_dets[b].sum() > 0:
                unmatched_detections = torch.nonzero(valid_dets[b]).squeeze(dim=1).tolist()
        association_result.append(
            {
                "matched_td_indices": matched_td_indices,
                "unmatched_trackers": unmatched_trackers,
                "unmatched_detections": unmatched_detections,
            }
        )
    return association_matrix, association_result

def argmax_assignment(sim_matrix):
    sim_matrix = sim_matrix.cpu()

    col_idx = sim_matrix.argmax(dim=1).numpy()
    row_idx = torch.arange(sim_matrix.size(0)).numpy()

    assignment = {}
    for i in range(len(row_idx)):
        row = row_idx[i]
        col = col_idx[i]
        cost = sim_matrix[row, col].item()

        if col in assignment:
            prev_row, prev_cost = assignment[col]
            if cost > prev_cost:
                assignment[col] = (row, cost)
        else:
            assignment[col] = (row, cost)

    final_row_idx = []
    final_col_idx = []
    for col, (row, cost) in assignment.items():
        final_row_idx.append(row)
        final_col_idx.append(col)
    return final_row_idx, final_col_idx
