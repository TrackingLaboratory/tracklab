import os
import cv2
import json

import torch.utils.data as data
import torchvision.transforms as transforms

class PoseTrack(data.Dataset):
    """
        This dataset is only for inference.
        Annotations agnostic. 
    """
    def __init__(self, path, train=False, im1_max_size=480):
        self.path = path
        subset = "train" if train else "val"
        self.im1_max_size = im1_max_size
        self.transform = transforms.ToTensor()
        self.metadatas = []
        
        p_posetrack_data = osp.join(path, "posetrack_data", subset)        
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
        path = osp.join(self.path, metadata['file_name'])
        img = cv2.imread(path)
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
    

if __name__ == '__main__': # testing function
    from torch.utils.data import DataLoader as dl
    dataset = PoseTrack("PoseTrack21/data", train=False)
    dataset = dl(dataset, batch_size=1, shuffle=False)
    for i, batch in enumerate(dataset):
        pass
    
    dataset = PoseTrack2("PoseTrack21/data", train=True, im1_max_size=222)
    dataset = dl(dataset, batch_size=1, shuffle=False)
    for i, batch in enumerate(dataset):
        pass