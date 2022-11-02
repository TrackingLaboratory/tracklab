import torch
import numpy as np

from pathlib import Path
from strong_sort.utils.parser import get_config # TODOs in here
from strong_sort.strong_sort import StrongSORT

@torch.no_grad()
class StrongSORT2detections():
    def __init__(self, config_strongsort, device):
        self.cfg = get_config()
        self.cfg.merge_from_file(config_strongsort)
        weights = Path(self.cfg.STRONGSORT.WEIGHTS).resolve()
        
        self.model = StrongSORT(
            max_dist=self.cfg.STRONGSORT.MAX_DIST,
            max_iou_distance=self.cfg.STRONGSORT.MAX_IOU_DISTANCE,
            max_age=self.cfg.STRONGSORT.MAX_AGE,
            n_init=self.cfg.STRONGSORT.N_INIT,
            nn_budget=self.cfg.STRONGSORT.NN_BUDGET,
        )
        # For camera compensation
        # TODO not sure about utility
        self.prev_frame = None
        
    def _image2input(self, image): # Tensor RGB (1, 3, H, W)
        assert 1 == image.shape[0], "Test batch size should be 1"
        input = image[0].cpu().numpy() # -> (3, H, W)
        input = np.transpose(input, (1, 2, 0)) # -> (H, W, 3)
        input = input*255.0
        input = input.astype(np.uint8) # -> to uint8
        return input

    def _camera_compensation(self, curr_frame):
        if self.cfg.STRONGSORT.ECC:  # camera motion compensation
            self.model.tracker.camera_update(self.prev_frame, curr_frame)
            self.prev_frame = curr_frame
            
    def _detections2inputs(self, detections):
        xywhs = []
        for Bbox in detections.Bboxes:
            x = (Bbox[0]+Bbox[2])/2
            y = (Bbox[1]+Bbox[3])/2
            w = Bbox[2] - Bbox[0]
            h = Bbox[3] - Bbox[1]
            xywhs.append(np.array([x, y, w, h]))
        
        xywhs = torch.Tensor(np.asarray(xywhs))
        reid_features = torch.Tensor(detections.reid_features)
        scores = torch.Tensor(detections.scores)
        classes = torch.Tensor([0.]*len(detections.scores))
        return xywhs, reid_features, scores, classes
        
    def run(self, detections, image):
        image = self._image2input(image)
        self._camera_compensation(image)
        
        results = []
         # check if instance(s) has been detected
         # by pose detector
        if detections.scores:
            xywhs, reid_features, scores, classes = self._detections2inputs(detections)
            results = self.model.update(xywhs,
                                        reid_features,
                                        scores,
                                        classes,
                                        image)
        
        detections = self._update_detections(results, detections)
        return detections

    def _update_detections(self, results, detections):
        Bboxes = []
        IDs = []
        scores = []
        Poses = []
        for result in results:
            Keypoints = self._find_Keypoints(result[:4], detections)
            Bboxes.append(result[:4])
            IDs.append(result[4])
            scores.append(result[6])
            Poses.append(Keypoints)
        detections.add_Tracks(Bboxes, IDs, scores, Poses)
        return detections
    
    def _find_Keypoints(self, Bbox_track, detections):
        """
            TODO rendre ça mieux car c'est vraiment degeulasse
            Comme les Bboxes en output de StrongSort sont différentes de celles
            du detecteur, je suis obligé des les parser pour les associer.
            Ici l'algo est clairement sous optimisé O(n) mais on appelle
            cette fonction n fois -> O(n²) in fine, c'est franchement degeux.
            Ca serait cool que dans le framework Re-ID, on ai un matching
            input / output pour éviter ça...
        """
        # compute euclidean distance for each Bbox, select the smallest one
        # to return the Keypoints
        distances = []
        for BBox_detect in detections.Bboxes:
            distances.append(
                np.sum((Bbox_track - BBox_detect)**2)
            )
        argmin = distances.index(min(distances))
        return detections.Poses[argmin]