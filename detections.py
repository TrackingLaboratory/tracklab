import numpy as np
import torch
import cv2

cmap = [(0,0,255), (0,128,255), (0,255,255), (0,255,128), (0,255,0), (128,255,0),
        (255,255,0), (255,128,0), (255,0,0), (255,0,128), (255,0,255), (128,0,255),
        (128,128,128), (0,0,0), (255,255,255)]

class Detections:
    
    def __init__(self, bboxes=[], poses=[], scores=[], h=None, w=None, **kwargs):
        self.poses = poses
        self.scores = scores
        self.bboxes = bboxes
        self.h = h
        self.w = w
                 
    def update_bboxes(self):
        # update bboxes based on self.poses
        self.bboxes = []
        for pose in self.poses:
            left_top = np.amin(pose, axis=0)
            right_bottom = np.amax(pose, axis=0)
            self.bboxes.append(
                np.array([left_top[0], left_top[1],
                              right_bottom[0], right_bottom[1]])
            )
    
    def update_HW(self, H, W):
        # update and verify attributes with new sizes
        self.H = H
        self.W = W
        self._update_Bboxes()
        self._update_Poses()
            
    def _update_Bboxes(self):
        # scale bboxes (xyxy) to BBoxes (XYXY)
        gain = min(self.h/self.H, self.w/self.W)  # gain  = old / new
        h_pad, w_pad  = (self.h - self.H*gain)/2, (self.w - self.W*gain) / 2  # wh padding
        self.Bboxes = []
        for bbox in self.bboxes:
            BBox = bbox.copy()
            BBox[[0, 2]] -= w_pad
            BBox[[1, 3]] -= h_pad
            BBox /= gain
            self.Bboxes.append(BBox)
        
    def _update_Poses(self):
        # scale keypoints (xys) to Keypoints (XYs)
        gain = min(self.h/self.H, self.w/self.W)  # gain  = old / new
        h_pad, w_pad  = (self.h - self.H*gain)/2, (self.w - self.W*gain) / 2  # wh padding
        self.Poses = []
        for pose in self.poses:
            Pose = pose.copy()
            Pose[:, 0] -= w_pad
            Pose[:, 1] -= h_pad
            Pose[:, :2] /= gain
            self.Poses.append(Pose)
            
    def get_StrongSORT_inputs(self):
        xywhs = []
        for Bbox in self.Bboxes:
            x = (Bbox[0]+Bbox[2])/2
            y = (Bbox[1]+Bbox[3])/2
            w = Bbox[2] - Bbox[0]
            h = Bbox[3] - Bbox[1]
            xywhs.append(np.array([x, y, w, h]))
        
        xywhs = torch.Tensor(np.asarray(xywhs))
        scores = torch.Tensor(self.scores)
        classes = torch.Tensor([0.]*len(self.scores))
        #FIXME extremely slow operation torch.asarray()
        return xywhs, scores, classes
    
    def show_image(self, img):
        # img : Tensor RGB (1, 3, H, W)
        self.img = 255*img[0].cpu().numpy().transpose(1, 2, 0) # -> RGB (H, W, 3)
        
    def show_Bboxes(self):
        for i, Bbox in enumerate(self.Bboxes):
            p1, p2 = (int(Bbox[0]), int(Bbox[1])), (int(Bbox[2]), int(Bbox[3]))
            
            overlay = self.img.copy()
            cv2.rectangle(overlay, p1, p2, color=(0,0,255), thickness=2)
            alpha = self.scores[i]

            self.img = cv2.addWeighted(overlay, alpha, self.img, 1 - alpha, 0)
            
    def show_Poses(self):
        for i, Pose in enumerate(self.Poses):
            for Keypoint in Pose:
                x, y = int(Keypoint[0]), int(Keypoint[1])
                
                overlay = self.img.copy()
                cv2.circle(overlay, (x, y), radius=1, color=(0,0,255), thickness=2)
                alpha = Keypoint[2]

                self.img = cv2.addWeighted(overlay, alpha, self.img, 1 - alpha, 0)
                
    def show_Tracks(self, Tracks):
        for i, Track in enumerate(Tracks):
            p1, p2 = (int(Track[0]), int(Track[1])), (int(Track[2]), int(Track[3]))
            
            overlay = self.img.copy()
            cv2.rectangle(overlay, p1, p2, color=(255,0,0), thickness=2)
            alpha = Track[6]

            self.img = cv2.addWeighted(overlay, alpha, self.img, 1 - alpha, 0)
            self.img = cv2.putText(self.img, f"ID={int(Track[4])}", p1, 
                                   fontFace=2,
                                   fontScale=0.5,
                                   color=(255,0,0))
    
    def get_image(self):
        assert hasattr(self, 'img'), "No image added, you should first load an image"+\
            " using add_img(img)." # TODO add a check of before each call to function
        img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR).astype(np.uint8)
        return img
        
