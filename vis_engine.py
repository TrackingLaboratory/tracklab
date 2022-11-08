import yaml
import cv2
import numpy as np
import os

class Vis_Engine():
    def __init__(self, vis_cfg, tracker):
        # handle args
        with open(vis_cfg, "r") as yamlfile:
            cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
        self.cfg = cfg
        self.tracker = tracker

        self.save_img = self.cfg['images']['save']
        self.save_vid = self.cfg['video']['save']
        
        if self.save_img:
            self.save_image_dir = os.path.join('runs', 
                self.cfg['save_dir'],
                'images')
            os.makedirs(self.save_image_dir, exist_ok=True)
            
        if self.save_vid:
            self.save_video_name = os.path.join('runs', 
                self.cfg['save_dir'],
                'results.mp4')
        
    def process(self, data):
        patch = self._process_img(data['image'])
        
        if self.cfg['detection']['bbox']['show']:
            patch = self._plot_bbox(data, patch)
        if self.cfg['detection']['pose']['show']:
            patch = self._plot_pose(data, patch)
        if self.cfg['tracking']['show']:
            patch = self._plot_track(data, patch)

        patch = self._final_patch(patch)
        if self.save_img:
            filepath = os.path.join(
                self.save_image_dir,
                data['filename']
            )
            assert cv2.imwrite(filepath, patch)
        # save video
        if self.save_vid:
            self._update_video(patch, (data['width'], data['height']))
            
    def _plot_bbox(self, data, patch):
        detections = self.tracker[(self.tracker.filename == data['filename']) & \
                                  (self.tracker.source == 2)]
        bboxes = detections.bbox_xyxy(with_conf=True)
        for bbox in bboxes:
            p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
            cv2.rectangle(patch, p1, p2, 
                          color=self.cfg['detection']['bbox']['color'],
                          thickness=self.cfg['detection']['bbox']['thickness'])
            if self.cfg['detection']['bbox']['print_conf']:
                p = (int(bbox[0]) + 1, int(bbox[1]) - 2)
                cv2.putText(patch, f' {bbox[4]:.2}', p,
                        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale=0.75,
                        color=self.cfg['detection']['bbox']['color'],
                        thickness=1)
        return patch
    
    def _plot_pose(self, data, patch):
        detections = self.tracker[(self.tracker.filename == data['filename'])]
        poses = detections.pose_xy(with_conf=True)
        for pose in poses:
            for kp in pose:
                p = (int(kp[0]), int(kp[1]))
                cv2.circle(patch, p, 
                           radius=self.cfg['detection']['pose']['radius'], 
                           color=self.cfg['detection']['pose']['color'], 
                           thickness=self.cfg['detection']['pose']['thickness'])
                if self.cfg['detection']['pose']['print_conf']:
                    p = (int(kp[0]) + 1, int(kp[1]) - 2)
                    cv2.putText(patch, f'{kp[2]:.2}', p,
                        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale=0.75,
                        color=self.cfg['detection']['pose']['color'],
                        thickness=1)
        return patch
    
    def _plot_track(self, data, patch):
        detections = self.tracker[(self.tracker.filename == data['filename']) & \
                                  (self.tracker.source == 3)]
        bboxes = detections.bbox_xyxy(with_conf=True)
        ids = detections[['person_id']].values
        for bbox, id in zip(bboxes, ids):
            p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
            cv2.rectangle(patch, p1, p2, 
                          color=self.cfg['tracking']['color'],
                          thickness=self.cfg['tracking']['thickness'])
            txt = ''
            if self.cfg['tracking']['print_id']:
                txt += f' ID: {int(id)}'
            if self.cfg['tracking']['print_conf']:
                txt += f' - conf: {bbox[4]:.2}'
            p = (int(bbox[0]) + 1, int(bbox[1]) - 2)
            cv2.putText(patch, txt, p,
                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale=0.75,
                    color=self.cfg['tracking']['color'],
                    thickness=1)
        return patch
                
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
        patch = img.cpu().numpy()
        patch = 255.*patch
        patch = patch.transpose(1, 2, 0)
        patch = np.ascontiguousarray(patch, dtype=np.float32)
        return patch
        
    def _final_patch(self, patch):
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
        patch = patch.astype(np.uint8)
        return patch
        