import os
import cv2
import json
import torch
import torchvision.transforms as T


class PoseTrack(torch.utils.data.Dataset):  # TODO replace by PoseTrack21
    """
        Not for training.
        This dataset is only for inference and testing.
        Annotations agnostic. 
    """
    def __init__(self, path, subset):
        self.path = path
        self.subset = subset
        self.transform = T.ToTensor()
        self.datas = []
        
        posetrack = os.path.join(path, 'posetrack_data', subset)        
        files = os.listdir(posetrack)
        for file in files:
            file_path = os.path.join(posetrack, file)
            with open(file_path) as f:
                metadatas = json.load(f)
                for frame, metadata in enumerate(metadatas['images']):
                    metadata['frame'] = frame + 1
                self.datas.extend(metadatas['images'])
            
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        data = self.datas[index]
        img_path = os.path.join(self.path, data['file_name'])
        img = cv2.imread(img_path) # BGR (H, W, 3)
        assert img is not None, 'Error while reading image'
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # -> RGB (H, W, 3) 
        img = self.transform(img) # -> RGB|float 0 -> 1|(3, H, W)
        return {
            'image': img,
            'file_path': data['file_name'],
            'height': img.shape[1], # transform has not been applied yet
            'width': img.shape[2], # transform has not been applied yet
            'image_id': data['image_id'],
            'file_name': os.path.basename(data['file_name']),
            'video_name': os.path.basename(os.path.dirname(data['file_name'])),
            'frame': data['frame'],
            'nframes': data['nframes'],
        }
        
class ImageFolder(torch.utils.data.Dataset):
    
    def __init__(self, path):
        assert os.path.isdir(path), f"{path} is not a directory. path should point to" +\
            " a folder with images"
        
        self.path = path
        self.transform = T.ToTensor()
        self.images = []
        for image in sorted(os.listdir(self.path)):
            to_lower = image.lower()
            if to_lower.endswith(('.png', '.jpg', '.jpeg')):
                path_to_image = os.path.join(self.path, image)
                self.images.append(path_to_image)
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        path = self.images[index]
        img = cv2.imread(path) # BGR (H, W, 3)
        assert img is not None, 'Error while reading image'
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # -> RGB (H, W, 3) 
        img = self.transform(img) # -> RGB|float 0 -> 1|(3, H, W)
        return {
            'image': img,
            'file_path': path,
            'height': img.shape[1], # transform has not been applied yet
            'width': img.shape[2], # transform has not been applied yet
            'file_name': os.path.basename(path),
            'video_name': os.path.basename(os.path.dirname(path)),
        }
