"""
 @Author  : Shuai Liao
"""

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from collections import OrderedDict as odict
import numpy as np
from pytorch_util.torch_v4_feature import LocalResponseNorm # *
from pytorch_util.libtrain import copy_weights, init_weights_by_filling
from pytorch_util.netutil.common_v2.trunk_vgg import get_caffeSrc2Dst


class down(nn.Module):
    def __init__(self, block_cfg=['M', 64, 64], in_channels=3, batch_norm=True):
        super(down, self).__init__()
        #
        layers = []
        for v in block_cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        self.conv = nn.Sequential(*layers)

        if block_cfg[0]=='M':
            self.pool = layers[0]
            self.conv = nn.Sequential(*layers[1:])
        else:
            self.pool = None
            self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.pool is not None:
            x, poolInd = self.pool(x)
            x = self.conv(x)
            return x, poolInd
        else:
            x = self.conv(x)
            return x


class up(nn.Module):
    def __init__(self, block_cfg, in_channels, batch_norm=True):
        super(up, self).__init__()
        layers = []
        for v in block_cfg:
            if v == 'M':
                layers += [nn.MaxUnpool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.ConvTranspose2d(in_channels, v, kernel_size=3, padding=1, bias=True) # bias=False
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        if block_cfg[-1]=='M':
            self.deconv = nn.Sequential(*layers[:-1])
            self.uppool = layers[-1] # maxUnpool2d
        else:
            self.deconv = nn.Sequential(*layers)

    def forward(self, x1, poolInd=None, x2=None): # output_size=None):
        """First deconv and then do uppool,cat if needed."""
        x1 = self.deconv(x1)
        if x2 is not None:
            x1 = self.uppool(x1,poolInd, output_size=x2.size())
            x1 = torch.cat([x2, x1], dim=1)
        return x1



class VGG16_Trunk(nn.Module):

    def __init__(self, init_weights=True):
        super(VGG16_Trunk, self).__init__()
        self.net_arch = 'vgg16' # vgg_type

        self.conv1 = down([      64,  64,     ], in_channels=3     ) # -- 3 -> 64
        self.conv2 = down(['M', 128, 128,     ], in_channels=64    ) # -- 64 -> 128
        self.conv3 = down(['M', 256, 256, 256 ], in_channels=128   ) # -- 128 ->256
        self.conv4 = down(['M', 512, 512, 512 ], in_channels=256   ) # -- 256 -> 512
        self.conv5 = down(['M', 512, 512, 512 ], in_channels=512   ) # -- 512 -> 512
        #
        self.deconv5_abs = up([512, 512, 512, 'M'], in_channels=512    ) # create_deconv_3(512, 512)
        self.deconv4_abs = up([512, 512, 256, 'M'], in_channels=512+512) # create_deconv_3(512+512, 256)
        self.deconv3_abs = up([256, 256, 128, 'M'], in_channels=256+256) # create_deconv_3(256+256, 128)
        self.deconv2_abs = up([128,  64,      'M'], in_channels=128+128) # create_deconv_2(128+128, 64)
        self.deconv1_abs = up([ 64,   3,         ], in_channels=64+64  ) # create_deconv_2(64+64, output_channel)
        #
        self.deconv5_sgc = up([512, 512, 512, 'M'], in_channels=512    ) # create_deconv_3(512, 512)
        self.deconv4_sgc = up([512, 512, 256, 'M'], in_channels=512+512) # create_deconv_3(512+512, 256)
        self.deconv3_sgc = up([256, 256, 128, 'M'], in_channels=256+256) # create_deconv_3(256+256, 128)
        self.deconv2_sgc = up([128,  64,      'M'], in_channels=128+128) # create_deconv_2(128+128, 64)
        self.deconv1_sgc = up([ 64,   4,         ], in_channels=64+64  )
        self.deconv1_abs.deconv = self.deconv1_abs.deconv[:-1]
        self.deconv1_sgc.deconv = self.deconv1_sgc.deconv[:-1]
        if init_weights:
            self.init_weights(None)#(pretrained='caffemodel' if (self.net_arch in ['vgg16', 'vgg19']) else None)

    def forward(self, x):  # In-replace forward_trunk
        """ x is input image data.
            x is of shape:  (batchsize, 3, 224, 224)  """
        # forward convolutional layers
        en1          = self.conv1(x)
        en2,poolInd1 = self.conv2(en1)
        en3,poolInd2 = self.conv3(en2)
        en4,poolInd3 = self.conv4(en3)
        en5,poolInd4 = self.conv5(en4)
        #
        de4_abs      = self.deconv5_abs(en5    , poolInd4, en4) # , output_size=en4.size()
        de3_abs      = self.deconv4_abs(de4_abs, poolInd3, en3) # , output_size=en3.size()
        de2_abs      = self.deconv3_abs(de3_abs, poolInd2, en2) # , output_size=en2.size()
        de1_abs      = self.deconv2_abs(de2_abs, poolInd1, en1) # , output_size=en1.size()
        x_abs        = self.deconv1_abs(de1_abs)
        #
        de4_sgc      = self.deconv5_sgc(en5    , poolInd4, en4) # , output_size=en4.size()
        de3_sgc      = self.deconv4_sgc(de4_sgc, poolInd3, en3) # , output_size=en3.size()
        de2_sgc      = self.deconv3_sgc(de3_sgc, poolInd2, en2) # , output_size=en2.size()
        de1_sgc      = self.deconv2_sgc(de2_sgc, poolInd1, en1) # , output_size=en1.size()
        x_sgc        = self.deconv1_sgc(de1_sgc)
        #
        return x_abs, x_sgc # x

    def init_weights(self, pretrained='caffemodel'):
        """ Two ways to init weights:
            1) by copying pretrained weights.
            2) by filling empirical  weights. (e.g. gaussian, xavier, uniform, constant, bilinear).
        """

        if pretrained is None:
            print('Initializing weights by filling (with gaussian).')
            init_weights_by_filling(self)
        elif pretrained=='caffemodel':
            print("Initializing weights by copying (pretrained caffe weights).")
            src2dsts= odict(conv1_1='conv1.conv.0' , conv1_2='conv1.conv.2' ,
                            conv2_1='conv2.conv.0' , conv2_2='conv2.conv.2' ,
                            conv3_1='conv3.conv.0' , conv3_2='conv3.conv.2', conv3_3='conv3.conv.4',
                            conv4_1='conv4.conv.0' , conv4_2='conv4.conv.2', conv4_3='conv4.conv.4',
                            conv5_1='conv5.conv.0' , conv5_2='conv5.conv.2', conv5_3='conv5.conv.4',)
            copy_weights(self.state_dict(), 'caffemodel.%s'%(self.net_arch), src2dsts=src2dsts)
        elif pretrained=='torchmodel':
            src2dsts= odict([('features.0' ,'conv1.conv.0') , ('features.2' ,'conv1.conv.2'),
                             ('features.5' ,'conv2.conv.0') , ('features.7' ,'conv2.conv.2'),
                             ('features.10','conv3.conv.0') , ('features.12','conv3.conv.2'), ('features.14','conv3.conv.4'),
                             ('features.17','conv4.conv.0') , ('features.19','conv4.conv.2'), ('features.21','conv4.conv.4'),
                             ('features.24','conv5.conv.0') , ('features.26','conv5.conv.2'), ('features.28','conv5.conv.4'),])
            copy_weights(self.state_dict(), 'torchmodel.%s'%(self.net_arch), strict=False, src2dsts=src2dsts)
        else:
            raise NotImplementedError

        for name, m in self.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                print ('>>>>>>>>>>>', name)
                assert m.kernel_size[0] == m.kernel_size[1]
                try:
                    initial_weight = get_upsampling_weight(
                        m.in_channels, m.out_channels, m.kernel_size[0])
                    print (m.weight.shape, initial_weight.shape)
                    m.weight.data.copy_(initial_weight)
                except:
                    print ('pass')
                    pass
            elif isinstance(m, nn.BatchNorm2d): # classname.find('BatchNorm') != -1:
                print ('>>>>>>>>>>>', name)
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
        return self


    def fix_conv1_conv2(self):
        # Fix the layers before conv3:
        for layer in range(10):
            for p in self.features[layer].parameters():
                p.requires_grad = False


# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


if __name__ == '__main__':

    # import numpy as np
    dummy_batch_data  = np.ones((1,3,224,224),dtype=np.float32)
    dummy_batch_data  = torch.from_numpy(dummy_batch_data )

    model = VGG16_Trunk(init_weights=True)
    Pred = model(dummy_batch_data)

