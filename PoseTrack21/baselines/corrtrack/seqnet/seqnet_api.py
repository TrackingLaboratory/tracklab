import torch
from .seqnet import SeqNet

class SeqNetApi(SeqNet):

    def __init__(self, cfg):
        super().__init__(cfg)

        # stores current images and features 
        self.images = None 
        self.features = None 

    def forward(self, *args, **kwargs):
        assert False and "Forward function is disabled for SeqNet API"

    def load_image(self, image): 
        """
        image: image tensor
        """
        images, _ = self.transform([image], None)
        features = self.backbone(images.tensors)

        self.features = features
        self.images = images 

    def get_box_embeddings(self, boxes): 
        # boxes: list of bbox tensors with [x1, y1, x2, y2]
    
        box_features = self.roi_heads.box_roi_pool(self.features, boxes, self.images.image_sizes)
        box_features = self.roi_heads.reid_head(box_features)
        embeddings, norms = self.roi_heads.embedding_head(box_features)
        
        return embeddings, torch.sigmoid(norms)
