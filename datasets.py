import os
import cv2
import json

import torch
import torchvision.transforms as T


class PoseTrack(torch.utils.data.Dataset):
    """
        Not for training.
        This dataset is only for inference and testing.
        Annotations agnostic. 
    """
    def __init__(self, path, train=False):
        self.path = path
        #subset = "train" if train else "val" # FIXME
        subset = "tiny_val" # for testing purposes
        self.transform = T.ToTensor()
        self.images = []
        
        posetrack = os.path.join(path, "posetrack_data", subset)        
        files = os.listdir(posetrack)
        for file in files:
            file_path = os.path.join(posetrack, file)
            with open(file_path) as f:
                data = json.load(f)
                file_name = file.split('.')[0]
                for frame, image in enumerate(data['images']):
                    image['folder'] = file_name
                    image['frame'] = frame+1
                self.images.extend(data['images'])
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        data = self.images[index]
        path = os.path.join(self.path, data['file_name'])
        img = cv2.imread(path) # (H, W, 3) BGR
        assert img is not None, "Error while reading image"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # (H, W, 3) RGB
        img = self.transform(img) # (3, H, W) RGB Tensor float 0 -> 1
        return {
            "image": img,
            "folder": data['folder'],
            "file_name": data['file_name'],
            "image_id": data['image_id'],
            "frame": data['frame'],
            "nframes": data['nframes']
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
            if to_lower.endswith((".png", ".jpg", ".jpeg")):
                path_to_image = os.path.join(self.path, image)
                self.images.append(path_to_image)
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        path = self.images[index]
        img = cv2.imread(path) # BGR (H, W, 3)
        assert img is not None, "Error while reading image"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # -> RGB (H, W, 3) 
        img = self.transform(img) # -> (H, W, 3)
        return {
            "image": img,
            ""
            "filename": os.path.basename(path),
            "height": img.shape[1],
            "width": img.shape[2]
        }


if __name__ == '__main__': # testing function
    dataset = ImageFolder("../Yolov5_StrongSORT_OSNet/data/test_images")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )
    
    for i, image in enumerate(dataloader):
        pass