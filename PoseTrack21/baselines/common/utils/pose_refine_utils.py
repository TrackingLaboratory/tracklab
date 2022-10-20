import numpy as np
import torch
import torch.nn as nn


def build_querie(pose):
    queries_u = np.zeros((1, 17, 3))
    queries_u[0, :, 0] = (pose[1, :] / 4).astype(np.uint)
    queries_u[0, :, 1] = (pose[0, :] / 4).astype(np.uint)
    queries_u[0, :, 2] = pose[2, :]

    return queries_u


def create_warp_matrix(output_size, crop_pos, scales, flip=0):


    param = {'rot': 0,
            'scale': scales,  # scale,
            'flip': flip,
            'tx': 0,
            'ty': 0}

    a = param['scale'][0] * np.cos(param['rot'])
    b = param['scale'][1] * np.sin(param['rot'])

    shift_to_upper_left = np.identity(3)
    shift_to_center = np.identity(3)

    t = np.identity(3)
    t[0][0] = a
    if param['flip']:
        t[0][0] = -a

    t[0][1] = -b
    t[1][0] = b
    t[1][1] = a

    shift_to_upper_left[0][2] = -crop_pos[0] + param['tx']
    shift_to_upper_left[1][2] = -crop_pos[1] + param['ty']
    shift_to_center[0][2] = output_size[0] // 2
    shift_to_center[1][2] = output_size[1] // 2
    t_form = np.matmul(t, shift_to_upper_left)
    t_form = np.matmul(shift_to_center, t_form)

    return t_form


def compute_OKS_PoseTrack(gt, dt, thres):

    # changed variance for eyes to variance of shoulders for head_top and neck!
    sigmas = np.array([
        .26, .79, .79, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87,
        .87, .89, .89
    ]) / 10.0

    vars = (sigmas * 2)**2
    k = len(sigmas)
    g = np.array(gt)
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]

    indexes = vg >= thres

    min_x = np.min(xg[indexes])
    max_x = np.max(xg[indexes])
    min_y = np.min(yg[indexes])
    max_y = np.max(yg[indexes])

    bb_h = max_y - min_y
    bb_w = max_x - min_x
    area = bb_w * bb_h

    gt_bb = [min_x, min_y, bb_w, bb_h]

    k1 = np.count_nonzero(indexes)
    x0 = gt_bb[0] - gt_bb[2]
    x1 = gt_bb[0] + gt_bb[2] * 2
    y0 = gt_bb[1] - gt_bb[3]
    y1 = gt_bb[1] + gt_bb[3] * 2

    xd = dt[0::3]
    yd = dt[1::3]

    if k1 > 0:
        # measure the per-keypoint distacorr_pose_gtnce if keypoints visible
        dx = xd - xg
        dy = yd - yg
    else:
        # measure minimum distance to keypoints in (x0,y0) & (x1,y1)

        z = np.zeros((xd.shape[0], k))
        dx = np.maximum(z, x0 - xd) + np.maximum(z, xd - x1)
        dy = np.maximum(z, y0 - yd) + np.maximum(z, yd - y1)

    e = (dx**2 + dy**2) / vars / (area + np.spacing(1)) / 2
    if k1 > 0:
        e = e[vg > 0]
    OKS = np.sum(np.exp(-e)) / e.shape[0]

    return OKS


def compute_OKS_PoseTrack_tracking(gt, dt, thres):

    # changed variance for eyes to variance of shoulders for head_top and neck!
    sigmas = np.array([.26, .79, .79, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0

    vars = (sigmas * 2) ** 2

    ################ gt ################
    g = np.array(gt)

    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]

    indexes = vg >= thres
    if np.count_nonzero(indexes) == 0:
        return 0

    min_x = np.min(xg[indexes])
    max_x = np.max(xg[indexes])
    min_y = np.min(yg[indexes])
    max_y = np.max(yg[indexes])

    bb_h = max_y - min_y
    bb_w = max_x - min_x
    area_gt = bb_w * bb_h

    ###################################
    ############### dt ################
    xd = dt[0::3]
    yd = dt[1::3]
    vd = dt[2::3]

    indexes_d = vd >= thres
    if np.count_nonzero(indexes_d) == 0:
        return 0

    min_x = np.min(xd[indexes_d])
    max_x = np.max(xd[indexes_d])
    min_y = np.min(yd[indexes_d])
    max_y = np.max(yd[indexes_d])
    bb_h = max_y - min_y
    bb_w = max_x - min_x
    area_dt = bb_w * bb_h
    ###################################

    dx = xd - xg
    dy = yd - yg

    e = (dx**2 + dy**2) / vars / ((area_gt + area_dt) / 2 + np.spacing(1)) / 2
    
    ind = list(vg > thres) and list(vd > thres)
    e = e[ind]
    
    OKS = np.sum(np.exp(-e)) / e.shape[0]

    return OKS

def compute_pose_area(pose, scores, joint_threshold):

    indexes = scores >= joint_threshold

    if np.count_nonzero(indexes) == 0:
        return 0

    xg = pose[0::3]
    yg = pose[1::3]

    min_x = np.min(xg[indexes])
    max_x = np.max(xg[indexes])
    min_y = np.min(yg[indexes])
    max_y = np.max(yg[indexes])

    bb_h = max_y - min_y
    bb_w = max_x - min_x
    area = bb_w * bb_h

    return area



def compute_OKS_PoseTrack_iou(gt, dts, thres):
    # Used to perform oks between multiple pose instances at once

    # changed variance for eyes to variance of shoulders for head_top and neck!
    sigmas = np.array([.26, .79, .79, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0

    vars = (sigmas * 2) ** 2

    ################ gt ################
    g = np.array(gt)

    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]

    indexes = vg >= thres

    min_x = np.min(xg[indexes])
    max_x = np.max(xg[indexes])
    min_y = np.min(yg[indexes])
    max_y = np.max(yg[indexes])

    bb_h = max_y - min_y
    bb_w = max_x - min_x
    area_gt = bb_w * bb_h

    ###################################
    ############### dt ################
    ious = np.zeros(dts.shape[0])

    for n_d in range(0, dts.shape[0]):

        xd = dts[n_d][0::3]
        yd = dts[n_d][1::3]
        vd = dts[n_d][2::3]

        indexes_d = vd >= thres
        if np.count_nonzero(indexes_d)  < 2:
            # we remove this pose as it seem to be invalid!
            ious[n_d] = 1
            continue

        min_x = np.min(xd[indexes_d])
        max_x = np.max(xd[indexes_d])
        min_y = np.min(yd[indexes_d])
        max_y = np.max(yd[indexes_d])

        bb_h = max_y - min_y
        bb_w = max_x - min_x
        area_dt = bb_w * bb_h
        ###################################

        dx = xd - xg
        dy = yd - yg

        e = (dx ** 2 + dy ** 2) / vars / ((area_gt + area_dt) / 2 + np.spacing(1)) / 2

        ind = list(vg > thres) and list(vd > thres)
        e = e[ind]

        ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0

    return ious


def get_max_oks(poses, N_ins, gt, jt_th):

    max_oks = 0
    for m in range(N_ins):
        k = poses[m].numpy()
        dt = []
        for j in range(17):
            dt.append(k[0, j])
            dt.append(k[1, j])
            dt.append(1)
        oks = compute_OKS_PoseTrack_tracking(np.array(gt), np.array(dt), jt_th)

        if oks > max_oks:
            max_oks = oks

    return max_oks


def get_pose_from_correspondences(Y, X, conf, queries, jt_th, corr_th, image_res, min_kpts, offsets=None):

    batch_size = Y.shape[0]
    poses = np.zeros((batch_size, 17, 3))
    bbs = np.zeros((batch_size, 4))
    centers = np.zeros((batch_size, 2))
    for n in range(batch_size):

        cx = 0
        cy = 0
        valid_joint_cnt = 0

        for j in range(17):

            if j == 3 or j == 4:
                continue

            if queries[j][2] >= jt_th and conf[n][j] >= corr_th:

                y, x = Y[n][j], X[n][j]
                if y < 0 or x < 0 or y > image_res[1] // 4 - 1 or x > image_res[0] // 4 - 1:
                    continue
                if y == 0 or x == 0:
                    continue
                poses[n][j][0] = x*4
                poses[n][j][1] = y*4
                poses[n][j][2] = conf[n][j]
                cx += x*4
                cy += y*4
                valid_joint_cnt += 1

        if np.count_nonzero(poses[n][:,2] > corr_th) >= min_kpts:
            bbs[n] = bounding_box_from_pose(poses[n], corr_th, image_res)

    return poses, bbs, centers


def get_pose_from_correspondences_w_increased_bb(Y, X, conf,
                                                 queries, jt_th, corr_th,
                                                 image_res, min_kpts,
                                                 increase_ratio=0.1,
                                                 ignore_query_confidence=False):

    N = Y.shape[0]
    poses = np.zeros((N, 17, 3))
    bbs = np.zeros((N, 4))
    centers = np.zeros((N, 2))
    for n in range(N):

        cx = 0
        cy = 0
        N = 0

        for j in range(17):

            if j == 3 or j == 4:
                continue

            if (ignore_query_confidence or queries[j][2] >= jt_th) and conf[n][j] >= corr_th:
                y, x = Y[n][j], X[n][j]
                if y < 0 or x < 0 or y > image_res[1] // 4 - 1 or x > image_res[0] // 4 - 1:
                    continue
                if y == 0 or x == 0:
                    continue
                poses[n][j][0] = x*4 
                poses[n][j][1] = y*4
                poses[n][j][2] = conf[n][j]
                cx += x*4
                cy += y*4
                N += 1

        if np.count_nonzero(poses[n][:,2] > corr_th) >= min_kpts:
            bbs[n] = bounding_box_from_pose_relative_increase(poses[n], corr_th, image_res, size=increase_ratio)

    return poses, bbs, centers


def bounding_box_from_pose(kpts, th, image_res, increase_size=True, size=10):

    enlarge_by_px = size if increase_size else 0

    valid = kpts[:, 2] > th

    if np.count_nonzero(valid) == 0:
        return [0, 0, 0, 0]

    x_min = np.min(kpts[valid, 0])
    y_min = np.min(kpts[valid, 1])

    x_max = np.max(kpts[valid, 0])
    y_max = np.max(kpts[valid, 1])

    x_min = np.maximum(0, x_min-enlarge_by_px)
    y_min = np.maximum(0, y_min-enlarge_by_px)

    x_max = np.minimum(image_res[0] - 1, x_max+enlarge_by_px)
    y_max = np.minimum(image_res[1] - 1, y_max+enlarge_by_px)

    return [x_min, y_min, x_max, y_max]


def bounding_box_from_pose_relative_increase(kpts, th, image_res, size=0.1):

    valid = kpts[:, 2] > th

    if np.count_nonzero(valid) == 0:
        return [0, 0, 0, 0]

    x_min = np.min(kpts[valid, 0])
    y_min = np.min(kpts[valid, 1])

    x_max = np.max(kpts[valid, 0])
    y_max = np.max(kpts[valid, 1])

    width = x_max - x_min
    height = y_max - y_max

    offset_x = size * width / 2
    offset_y = size * height / 2

    x_min = np.maximum(0, x_min - offset_x)
    y_min = np.maximum(0, y_min - offset_y)
    x_max = np.minimum(image_res[0] - 1, x_max + offset_x)
    y_max = np.minimum(image_res[1] - 1, y_max + offset_y)

    # recenter bounding box
    bcx, bcy = (x_min + x_max) / 2, (y_min + y_max) / 2
    cx, cy = np.mean(kpts[valid, 0]), np.mean(kpts[valid, 1])
    diff_x, diff_y = bcx - cx, bcy - cy

    x_min = min(max(0, x_min - diff_x), image_res[0] - 1 - 1)
    y_min = min(max(0, y_min - diff_y), image_res[1] - 1 - 1)
    x_max = min(max(0, x_max - diff_x), image_res[0] - 1 - 1)
    y_max = min(max(0, y_max - diff_x), image_res[1] - 1 - 1)

    return [x_min, y_min, x_max, y_max]


def create_new_anno(im_id, keypoints, scores, tid):

    annotation = {'image_id': im_id,
                'track_id': int(tid.item()),
                'category_id': 1,
                'new_anno': True,
                'keypoints': keypoints,
                'scores': scores,

                }

    return annotation


def get_preds(prs, mat, sr, res_h, res_w):

    pool = nn.MaxPool2d(3, 1, 1).cuda()

    xoff = sr[0:17]
    yoff = sr[17:34]

    o = pool(prs.cuda()).data.cpu()
    maxm = torch.eq(o, prs).float()
    prs = prs * maxm

    prso = prs.view(17, res_w * res_h)
    val_k, ind = prso.topk(1, dim=1)
    xs = ind % res_w
    ys = (ind / res_w).long()

    keypoints = []
    score = 0
    scores = []
    points = torch.ones(3, 17)
    c = 0

    for j in range(17):

        x, y = xs[j][0], ys[j][0]
        dx = xoff[j][int(y)][int(x)]
        dy = yoff[j][int(y)][int(x)]
        points[0][j] = (x * 4) + dx.item()
        points[1][j] = (y * 4) + dy.item()
        score += val_k[j][0]

        scores.append(val_k[j][0].item()/2)
        c += 1

    score /= c

    points_warped = np.matmul(mat, points)

    for j in range(17):

        keypoints.append(float(points_warped[0][j]))
        keypoints.append(float(points_warped[1][j]))
        keypoints.append(1)

    return keypoints, scores, points, points_warped

def create_warps(boxes,  output_size):


    warps = []

    for i, b in enumerate(boxes):

        width, height = b[2] - b[0], b[3] - b[1]

        crop_pos = [(b[0] + b[2])/2, (b[1] + b[3])/2]

        max_d = np.maximum(height, width)
        scales = [float(output_size[0]) / float(max_d), float(output_size[1]) / float(max_d)]
        w = create_warp_matrix(output_size, crop_pos, scales)
        warps.append(w)

    return warps


def get_poses(keypoints, scores, warp, jt_th, image_dim):

    X, Y = keypoints[0:51:3], keypoints[1:51:3]
    pose_im = np.ones((3, 17))
    warp_inv = np.linalg.inv(warp)

    for j in range(17):
        if j == 3 or j == 4:
            scores[j] = 0
            pose_im[0, j], pose_im[1, j] = 0, 0
            continue

        if X[j] > ((image_dim[0] - 1) or Y[j] > (image_dim[1] - 1) or X[j] < 0 or Y[j] < 0):
            scores[j] = 0

        pose_im[0, j], pose_im[1, j] = X[j], Y[j]

    pose_orig_im = np.matmul(warp_inv, pose_im)
    pose_orig_im[2, :] = scores
    pose_im[2, :] = scores

    gt = []
    for j in range(17):
        x, y, s = pose_orig_im[0, j], pose_orig_im[1, j], pose_orig_im[2, j]

        if s == 0:
            gt.append(0)
            gt.append(0)
            gt.append(0)

            pose_im[0, j], pose_im[1, j] = 0, 0
            pose_orig_im[0, j], pose_orig_im[1, j] = 0, 0
        else:
            gt.append(x)
            gt.append(y)
            gt.append(s)
    return pose_im, pose_orig_im, gt
