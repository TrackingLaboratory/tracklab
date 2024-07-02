import numpy as np
import torch


def posetrack2h36m(x):
    '''
        Input: (B x T x V x C)

        PoseTrack keypoints = [ 'nose',
                                'head_bottom',
                                'head_top',
                                'left_ear',
                                'right_ear',
                                'left_shoulder',
                                'right_shoulder',
                                'left_elbow',
                                'right_elbow',
                                'left_wrist',
                                'right_wrist',
                                'left_hip',
                                'right_hip',
                                'left_knee',
                                'right_knee',
                                'left_ankle',
                                'right_ankle']
        H36M:
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    '''
    y = torch.zeros(x.shape, device=x.device)
    y[:, :, 0, :] = (x[:, :, 11, :] + x[:, :, 12, :]) * 0.5
    y[:, :, 1, :] = x[:, :, 12, :]
    y[:, :, 2, :] = x[:, :, 14, :]
    y[:, :, 3, :] = x[:, :, 16, :]
    y[:, :, 4, :] = x[:, :, 11, :]
    y[:, :, 5, :] = x[:, :, 13, :]
    y[:, :, 6, :] = x[:, :, 15, :]
    y[:, :, 8, :] = x[:, :, 1, :]
    y[:, :, 7, :] = (y[:, :, 0, :] + y[:, :, 8, :]) * 0.5
    y[:, :, 9, :] = x[:, :, 0, :]
    y[:, :, 10, :] = x[:, :, 2, :]
    y[:, :, 11, :] = x[:, :, 5, :]
    y[:, :, 12, :] = x[:, :, 7, :]
    y[:, :, 13, :] = x[:, :, 9, :]
    y[:, :, 14, :] = x[:, :, 6, :]
    y[:, :, 15, :] = x[:, :, 8, :]
    y[:, :, 16, :] = x[:, :, 10, :]
    y[:, :, 0, 2] = torch.minimum(x[:, :, 11, 2], x[:, :, 12, 2])
    y[:, :, 7, 2] = torch.minimum(y[:, :, 0, 2], y[:, :, 8, 2])
    return y
