import yaml
import cv2
import numpy as np
import os

class Vis_Engine():
    def __init__(self, vis_cfg, detections):
        # handle args
        with open(vis_cfg, "r") as yamlfile:
            cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
        self.cfg = cfg
        self.detections = detections
                
        if self.cfg['images']['save']:
            self.save_image_dir = os.path.join('runs', 
                self.cfg['save_dir'],
                self.cfg['images']['name_dir'])
            os.makedirs(self.save_image_dir, exist_ok=True)
            
        if self.cfg['images']['save']:
            self.save_video_name = os.path.join('runs', 
                self.cfg['save_dir'],
                self.cfg['video']['name'])
        
    def process(self, data):
        patch = self._process_img(data['image'])

        patch = self._final_patch(patch)        
        # save images
        if hasattr(self, 'save_image_dir'):
            filepath = os.path.join(
                self.save_image_dir,
                data['filename']
            )
            assert cv2.imwrite(filepath, patch)
        # save video
        if hasattr(self, 'save_video_name'):
            self._update_video(patch, (data['width'], data['height']))
                
    def _update_video(self, patch, size):
        if not hasattr(self, 'video'):
            self.video = cv2.VideoWriter(
                self.save_video_name,
                cv2.VideoWriter_fourcc(*'mp4v'),
                self.cfg['video']['fps'],
                size
            )
        self.video.write(patch)
    
    def _process_img(self, img):
        patch = img.detach().cpu().numpy()
        patch = np.transpose(patch, (1, 2, 0))
        patch = patch*255
        return patch
        
    def _final_patch(self, patch):
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
        patch = patch.astype(np.uint8)
        return patch
        