import os
import cv2
import json

import torch
import torchvision.transforms as T

# TODO modify this
class PoseTrack(torch.utils.data.Dataset):
    """
        This dataset is only for inference.
        Annotations agnostic. 
    """
    def __init__(self, path, train=False, im1_max_size=512):
        self.path = path
        subset = "train" if train else "val"
        self.im1_max_size = im1_max_size
        self.transform = T.ToTensor()
        self.metadatas = []
        
        p_posetrack_data = os.path.join(path, "posetrack_data", subset)        
        files = os.listdir(p_posetrack_data)
        for file in files:
            file_path = os.path.join(p_posetrack_data, file)
            with open(file_path) as f:
                metadata = json.load(f)
                file_name = file.split('.')[0]
                for image in metadata['images']:
                    # convert vid_id from str to int
                    image['vid_id'] = int(image['vid_id'])
                    # add file_name to the metadata
                    image['folder'] = file_name
                self.metadatas.extend(metadata['images'])
            
    def __len__(self):
        return len(self.metadatas)
    
    def __getitem__(self, index):
        metadata = self.metadatas[index]
        path = os.path.join(self.path, metadata['file_name'])
        img = cv2.imread(path) # (H, W, 3)
        assert img is not None, "Error while reading image"
        im0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # (H, W, 3)
        H, W = im0.shape[:2]
        if W > H:
            ratio = float(self.im1_max_size)/float(W)
            dim = (self.im1_max_size, int(ratio*H))
            im1 = cv2.resize(im0, dim, interpolation=cv2.INTER_LINEAR) # (h, w, 3)
        else:
            ratio = float(self.im1_max_size)/float(H)
            dim = (int(ratio*W), self.im1_max_size)
            im1 = cv2.resize(im0, dim, interpolation=cv2.INTER_LINEAR) # (h, w, 3)
        im1 = self.transform(im1) # (3, H, W)
        im0 = self.transform(im0) # (3, h, w)
        return {
            "metadata": metadata,
            "im0": im0,
            "im1": im1
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
        img = self.transform(img) # -> (1, H, W, 3)
        return img


if __name__ == '__main__': # testing function
    dataset = ImageFolder("../Yolov5_StrongSORT_OSNet/data/test_images")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )
    
    for i, image in enumerate(dataloader):
        pass