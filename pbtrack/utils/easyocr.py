import numpy as np

def bbox_easyocr_to_image_ltwh(easy_ocr_bbox, bbox_tlwh):
    p1, p2, p3, p4 = easy_ocr_bbox
    jn_bbox = np.array([p1+bbox_tlwh[0], bbox_tlwh[1]])
    return jn_bbox