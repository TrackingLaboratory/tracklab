# FILE WILL BE DELETED IN FUTURE

import os
import PIL
import json
import torch
from glob import glob
import torchvision.transforms.functional as F

from pbtrack.datastruct.metadatas import Metadata

class PoseTrack21(torch.utils.data.Dataset):
    def __init__(self, path, subset):
        self.path = path
        self.subset = subset
        self.images = []
        
        posetrack = os.path.join(path, 'posetrack_data', subset)        
        files = os.listdir(posetrack)
        for file in files:
            file_path = os.path.join(posetrack, file)
            with open(file_path) as f:
                metadatas = json.load(f)
                for frame, metadata in enumerate(metadatas['images']):
                    self.images.append(
                        Metadata(
                            id = metadata['id'],
                            video_id = metadata['vid_id'],
                            frame = frame,
                            nframe = metadata['nframes'],
                            file_path = metadata['file_name'],
                            is_labeled = metadata['is_labeled'],
                            ignore_regions_x=metadata['ignore_regions_x'],
                            ignore_regions_y=metadata['ignore_regions_y']
                        )  # type: ignore
                    )
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        metadata = self.images[index]
        image_path = os.path.join(self.path, metadata.file_path)
        image = PIL.Image.open(image_path)
        return dict(
            image = F.pil_to_tensor(image),
            metadata = metadata
            )  # type: ignore
        
class ImageFolder2(torch.utils.data.Dataset):
    def __init__(self, path):
        assert os.path.isdir(path), f"{path} is not a directory. path should point to" +\
            " a folder with images"
        
        self.path = path
        video_id = os.path.basename(os.path.dirname(path))
        
        files = []
        for type in ('*.jpg', '*.jpeg', '*.png'):
            files.extend(glob(os.path.join(path, type)))
        images = []
        for index, file in enumerate(sorted(files)):
            #image_path = os.path.join(self.path, file)
            images.append(
                Image(
                    id = index,
                    video_id = video_id,
                    frame = index,
                    nframe = len(files),
                    file_path = file,
                    is_labeled = False,
                    ignore_regions_x = [],
                    ignore_regions_y = []
                )  # type: ignore
            )
        self.images = images
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        metadata = self.images[index]
        #image_path = os.path.join(self.path, metadata.file_name)
        image = PIL.Image.open(metadata.file_path)
        return dict(
            image = F.pil_to_tensor(image),
            metadata = metadata
            )  # type: ignore
        
import cv2   
class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, path):
        assert os.path.isdir(path), f"{path} is not a directory. path should point to" +\
            " a folder with images"
        
        self.path = path
        video_id = os.path.basename(os.path.dirname(path))
        
        files = []
        for type in ('*.jpg', '*.jpeg', '*.png'):
            files.extend(glob(os.path.join(path, type)))
        images = []
        for index, file in enumerate(sorted(files)):
            #image_path = os.path.join(self.path, file)
            images.append(
                Image(
                    id = index,
                    video_id = video_id,
                    frame = index,
                    nframe = len(files),
                    file_path = file,
                    is_labeled = False,
                    ignore_regions_x = [],
                    ignore_regions_y = []
                )  # type: ignore
            )
        self.images = images
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return self.images[index]
    
    def get_image(self, image_id):
        for image in self.images:
            if image.id == image_id:
                return cv2.imread(image.file_path)
