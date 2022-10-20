import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


def get_preds_for_pose(prs, mat, sr, output_size, joint_scores=False):
    pool = nn.MaxPool2d(3, 1, 1).cuda()

    xoff = sr[0:17]
    yoff = sr[17:34]

    prs2 = prs

    o = pool(Variable(prs.cuda())).data.cpu()
    maxm = torch.eq(o, prs).float()
    prs = prs * maxm

    res_w = output_size[0] // 4
    res_h = output_size[1] // 4

    prso = prs.view(17, res_h * res_w)
    val_k, ind = prso.topk(1, dim=1)
    xs = ind % res_w
    ys = (ind / res_w).long()

    keypoints = []
    score = 0
    scores = []
    points = torch.zeros(17, 2)
    c = 0

    for j in range(17):

        x, y = xs[j][0], ys[j][0]
        dx = xoff[j][int(y)][int(x)]
        dy = yoff[j][int(y)][int(x)]
        points[j][0] = (x * 4) + dx.item()
        points[j][1] = (y * 4) + dy.item()

        score += val_k[j][0]

        if joint_scores:
            scores.append(val_k[j][0].item() / 2.0)
        c += 1

    score /= c

    for j in range(17):
        point = torch.ones(3, 1)
        point[0][0] = points[j][0]
        point[1][0] = points[j][1]

        keypoint = np.matmul(mat, point)
        keypoints.append(float(keypoint[0].item()))
        keypoints.append(float(keypoint[1].item()))
        keypoints.append(1)

    if joint_scores:
        return keypoints, scores
    else:
        return keypoints, score.item() / 2.0


def get_transform(param, crop_pos, output_size, scales):
    shift_to_upper_left = np.identity(3)
    shift_to_center = np.identity(3)

    a = scales[0] * param['scale_x'] * np.cos(param['rot'])
    b = scales[1] * param['scale_y'] * np.sin(param['rot'])

    t = np.identity(3)
    t[0][0] = a
    if param['flip']:
        t[0][0] = -a

    t[0][1] = -b
    t[1][0] = b
    t[1][1] = a

    shift_to_upper_left[0][2] = -crop_pos[0] + param['tx']
    shift_to_upper_left[1][2] = -crop_pos[1] + param['ty']
    shift_to_center[0][2] = output_size[0] / 2
    shift_to_center[1][2] = output_size[1] / 2
    t_form = np.matmul(t, shift_to_upper_left)
    t_form = np.matmul(shift_to_center, t_form)

    return t_form
