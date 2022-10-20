from __future__ import print_function, division, absolute_import
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from models.modules import PoseDecoder

__all__ = ['bninception']

model_urls = {
    'url': 'http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-52deb4733.pth'
}

class Inception(nn.Module):
    def __init__(self, out_ch=32):
        super(Inception, self).__init__()

        inplace = True
        self.conv1_7x7_s2 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.conv1_7x7_s2_bn = nn.BatchNorm2d(64, affine=True)
        self.conv1_relu_7x7 = nn.ReLU(inplace)
        self.pool1_3x3_s2 = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.conv2_3x3_reduce = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv2_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.conv2_relu_3x3_reduce = nn.ReLU(inplace)
        self.conv2_3x3 = nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_3x3_bn = nn.BatchNorm2d(192, affine=True)
        self.conv2_relu_3x3 = nn.ReLU(inplace)
        self.pd1 = PoseDecoder(192, out_ch=out_ch)

    def forward(self, x, get_stem_features=False):

        conv1_7x7_s2_out = self.conv1_7x7_s2(x)
        conv1_7x7_s2_bn_out = self.conv1_7x7_s2_bn(conv1_7x7_s2_out)
        conv1_relu_7x7_out = self.conv1_relu_7x7(conv1_7x7_s2_bn_out)
        pool1_3x3_s2_out = self.pool1_3x3_s2(conv1_relu_7x7_out)
        conv2_3x3_reduce_out = self.conv2_3x3_reduce(pool1_3x3_s2_out)
        conv2_3x3_reduce_bn_out = self.conv2_3x3_reduce_bn(conv2_3x3_reduce_out)
        conv2_relu_3x3_reduce_out = self.conv2_relu_3x3_reduce(conv2_3x3_reduce_bn_out)
        conv2_3x3_out = self.conv2_3x3(conv2_relu_3x3_reduce_out)
        conv2_3x3_bn_out = self.conv2_3x3_bn(conv2_3x3_out)
        x = self.conv2_relu_3x3(conv2_3x3_bn_out)
        output1 = self.pd1(x)

        if get_stem_features:
            return output1, x

        return output1


def bninception(pretrained=True, out_ch=32):
    model = Inception(out_ch)
    if pretrained:
        state_dict = model.state_dict()
        pretrained_state_dict = model_zoo.load_url(model_urls['url'])
        for k, v in state_dict.items():
            if 'pd' in k:
                newk = k[4:]
                if newk in pretrained_state_dict:
                    state_dict[k] = pretrained_state_dict[newk]
            if k in pretrained_state_dict:
                state_dict[k] = pretrained_state_dict[k]
        print('successfully load '+str(len(state_dict.keys()))+' keys')
        model.load_state_dict(state_dict)

    return model
