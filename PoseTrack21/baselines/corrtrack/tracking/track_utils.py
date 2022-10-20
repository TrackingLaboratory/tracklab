import torch
import numpy as np 

def get_reid_box_embeddings(bboxes, seq_net):
    box_sum = bboxes.sum(axis=-1)
    valid_boxes_inds = np.argwhere(box_sum != -4)
    boxes = torch.as_tensor(bboxes[valid_boxes_inds[:, 0]]).float()
    boxes = boxes.cuda()

    if boxes.size(0) > 0:
        box_embeddings, _ = seq_net.get_box_embeddings([boxes])
    else:
        box_embeddings = torch.zeros(0).cuda()
        
    return box_embeddings, valid_boxes_inds

