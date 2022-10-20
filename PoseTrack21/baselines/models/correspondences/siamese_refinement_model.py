from __future__ import print_function, division, absolute_import

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from torch.nn import DataParallel
from models.correspondences.siamese_embedding_model import Siamese as LocalCorr
from models.modules import PoseDecoder

model_urls = {
    'url': 'http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-52deb4733.pth'
}


def Refiner(in_ch, out_ch, pretrained=True):
    model = PoseDecoder(in_ch, out_ch)

    if pretrained:
        state_dict = model.state_dict()
        pretrained_state_dict = model_zoo.load_url(model_urls['url'])
        for k, v in state_dict.items():
            if 'pd' in k:
                newk = k[4:]
                if newk in pretrained_state_dict:
                    state_dict[k] = pretrained_state_dict[newk]
            if k  in pretrained_state_dict:
                state_dict[k] = pretrained_state_dict[k]
        print('successfully load '+str(len(state_dict.keys()))+' keys for refiner')
        model.load_state_dict(state_dict)
    return model


class Siamese(nn.Module):

    def __init__(self, res, local_ckpt=None):

        super(Siamese, self).__init__()

        self.res = res
        self.model = LocalCorr()
        #self.model = DataParallel(self.model)
        if local_ckpt is not None:
            import pdb; pdb.set_trace()
            checkpoint = torch.load(local_ckpt)
            pretrained_dict = checkpoint['state_dict']
            state_dict = self.model.state_dict()

            for k, v in pretrained_dict.items():
                newk = k
                if 'module.' in k:
                    newk = k[7:]
                state_dict[newk] = v
        
            self.model.load_state_dict(state_dict)
            for param in self.model.parameters():
                param.requires_grad = False
        self.pd = Refiner(64 + 17, 17)

    def load_pretrained_local_ckpt(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        pretrained_dict = checkpoint['state_dict']
        self.model.load_state_dict(pretrained_dict)
        for param in self.model.parameters():
            param.requires_grad = False

    def get_corr_maps3x3(self, fA, fB, queries):

        res = self.res

        N_batch = queries.shape[0]
        Nsamples = queries.shape[1]
        correlations = torch.zeros(N_batch, Nsamples, res, res)

        for n in range(N_batch):

            fBI = fB[n]
            d1 = fBI * fBI
            e_norms1 = torch.sum(d1, dim=0)
            e_norms1 = e_norms1.repeat([32, 1, 1])
            fBI_norm = fBI / e_norms1
            fBI_norm = fBI_norm.view(1, 32, res, res)

            fAI = fA[n]
            d2 = fAI * fAI
            e_norms2 = torch.sum(d2, dim=0)
            e_norms2 = e_norms2.repeat([32, 1, 1])
            fAI_norm = fAI / e_norms2

            for s in range(Nsamples):

                filter = torch.zeros(32, 3, 3).cuda()
                y, x = int(queries[n, s, 0]), int(queries[n, s, 1])
                if y < 0 or x < 0 or y > res-1 or x > res-1:
                    continue

                miny, maxy = max(0, y - 1), min(y + 1, res-1)
                minx, maxx = max(0, x - 1), min(x + 1, res-1)
                sfy, efy = 0, 2
                sfx, efx = 0, 2

                if y == 0:
                    sfy = 1
                    efy = 2
                if y == res-1:
                    sfy = 0
                    efy = 1
                if x == 0:
                    sfx = 1
                    efx = 2
                if x == res-1:
                    sfx = 0
                    efx = 1

                filter[:, sfy:efy + 1, sfx:efx + 1] = fAI_norm[:, miny:maxy + 1, minx:maxx + 1]
                corr = F.conv2d(fBI_norm, filter.view(1, 32, 3, 3), padding=1)
                corr = corr.view(res, res).reshape(res * res)
                corr = torch.softmax(corr, dim=0)
                correlations[n, s] = corr.reshape(res, res)

        return correlations.cuda()

    def forward(self, x1, x2, queries=None, embeddings_only=False, correlations_only=False):

        if correlations_only:
            correlation_maps = self.get_corr_maps3x3(x1, x2, queries)
            refined = self.pd(torch.cat([x1, x2, correlation_maps], 1))
            return refined

        eA, eB = self.model(x1, x2)

        if embeddings_only:
            return eA, eB

        correlation_maps = self.get_corr_maps3x3(eA, eB, queries)
        refined = self.pd(torch.cat([eA, eB, correlation_maps], 1))

        return refined, correlation_maps
