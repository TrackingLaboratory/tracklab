import numpy as np
import cv2

from reconnaissance.utils.images import overlay_heatmap

cmap = [(0,0,255), (0,128,255), (0,255,255), (0,255,128), (0,255,0), (128,255,0),
        (255,255,0), (255,128,0), (255,0,0), (255,0,128), (255,0,255), (128,0,255),
        (128,128,128), (0,0,0), (255,255,255)]


class Detections:
    
    def __init__(self, bboxes=[], poses=[], scores=[], h=None, w=None, **kwargs):
        self.poses = poses # low resolution
        # poses is a list of np.array of keypoints.
        # An element of pose is thus the keypoints of a detected instance
        self.bboxes = bboxes # low resolution
        self.scores = scores
        self.h = h # low resolution height
        self.w = w # low resolution width
        
    """
        Data part - working in both resolutions
    """
    
    def update_bboxes(self):
        # update bboxes based on self.poses
        self.bboxes = []
        for keypoints in self.poses:
            left_top = np.amin(keypoints, axis=0)
            right_bottom = np.amax(keypoints, axis=0)
            l = left_top[0]
            t = left_top[1]
            r = right_bottom[0]
            b = right_bottom[1]
            w = r - l
            h = b - t
            scale = 0.2
            sl = max(l - w*scale / 2, 0)
            st = max(t - h*scale / 2, 0)
            sr = min(r + w*scale / 2, self.w - 1)
            sb = min(b + h*scale / 2, self.h - 1)
            self.bboxes.append(
                np.array([sl, st, sr, sb])
            )

    def add_reid_features(self, reid_features):
        self.reid_features = reid_features

    def add_visibility_scores(self, visibility_scores):
        self.visibility_scores = visibility_scores

    def add_body_masks(self, body_masks):
        self.body_masks = body_masks

    def add_HW(self, H, W):
        # add high resolution sizes
        self.H = H
        self.W = W
        
    def add_Tracks(self, Bboxes, IDs, scores, Poses):
        self.Tracks = []
        for Bbox, Id, score, Keypoints in zip(Bboxes, IDs, scores, Poses):
            Track = dict(
                ID = Id,
                score = score,
                Bbox = Bbox,
                Keypoints = Keypoints
            )
            self.Tracks.append(Track)
            
    def update_Bboxes(self):
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
        
    def update_Poses(self):
        # scale keypoints (xys) to Keypoints (XYs)
        gain = min(self.h/self.H, self.w/self.W)  # gain  = old / new
        h_pad, w_pad  = (self.h - self.H*gain)/2, (self.w - self.W*gain) / 2  # wh padding
        self.Poses = []
        for keypoints in self.poses:
            Keypoints = keypoints.copy()
            Keypoints[:, 0] -= w_pad
            Keypoints[:, 1] -= h_pad
            Keypoints[:, :2] /= gain
            self.Poses.append(Keypoints)
            
    """
        Visualization part - working in high resolution
    """
            
    def _check_assertions(self):
        assert hasattr(self, 'img'), "No image added, you should first load an image"+\
            " using show_image(image)."
        assert type(self.img) == np.ndarray, ""
        assert self.img.dtype == np.float32, ""
        assert self.img.ndim == 3, ""
        assert self.img.shape[2] == 3, ""
    
    def show_image(self, img):
        # img : Tensor RGB (1, 3, H, W) (0 -> 1)
        img = 255*img[0].cpu().numpy().transpose(1, 2, 0)
        # -> RGB (H, W, 3)
        # OpenCV uses BGR
        self.img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # float 0 -> 255

    def show_Bboxes(self):
        self._check_assertions()
        for i, Bbox in enumerate(self.Bboxes):
            p1, p2 = (int(Bbox[0]), int(Bbox[1])), (int(Bbox[2]), int(Bbox[3]))
            
            overlay = self.img.copy()
            cv2.rectangle(overlay, p1, p2, color=(0, 0, 255), thickness=2)
            alpha = self.scores[i]

            self.img = cv2.addWeighted(overlay, alpha, self.img, 1 - alpha, 0)

    def show_masks(self):
        self._check_assertions()
        for i, Bbox in enumerate(self.Bboxes):
            l, t, r, b = int(Bbox[0]), int(Bbox[1]), int(Bbox[2]), int(Bbox[3])
            body_masks = self.body_masks[i]
            img_crop = self.img[t:b, l:r]
            img_crop_with_mask = overlay_heatmap(img_crop, body_masks[0].numpy(), weight=-1)
            self.img[t:b, l:r] = img_crop_with_mask

    def show_Poses(self):
        self._check_assertions()
        for i, Keypoints in enumerate(self.Poses):
            for Keypoint in Keypoints:
                x, y = int(Keypoint[0]), int(Keypoint[1])
                
                # overlay = self.img.copy()
                cv2.circle(self.img, (x, y), radius=3, color=(252, 93, 215), thickness=2)
                # alpha = Keypoint[2]
                # self.img = cv2.addWeighted(overlay, alpha, self.img, 1 - alpha, 0)
                
    def show_Tracks(self):
        self._check_assertions()
        for i, Track in enumerate(self.Tracks):
            p1 = int(Track['Bbox'][0]), int(Track['Bbox'][1])
            p2 = int(Track['Bbox'][2]), int(Track['Bbox'][3])
            
            overlay = self.img.copy()
            cv2.rectangle(overlay, p1, p2, color=(255,0,0), thickness=2)
            alpha = Track['score']

            self.img = cv2.addWeighted(overlay, alpha, self.img, 1 - alpha, 0)
            self.img = cv2.putText(self.img,
                                   f"ID={Track['ID']}",
                                   p1, 
                                   fontFace=2,
                                   fontScale=0.5,
                                   color=(255,0,0))
    
    def get_image(self):
        self._check_assertions()
        return self.img.astype(np.uint8)
        
