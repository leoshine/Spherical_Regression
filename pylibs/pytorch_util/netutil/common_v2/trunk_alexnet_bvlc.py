import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from basic.common import rdict
import numpy as np
from easydict import EasyDict as edict
import math

from pytorch_util.libtrain import copy_weights, init_weights_by_filling
from pytorch_util.torch_v4_feature import LocalResponseNorm # *
from pytorch_util.torch_3rd_layers import Maskout
from pytorch_util.torch_3rd_funcs  import norm2unit

__all__ = ['AlexNet_Trunk'] # , 'alexnet']


# nr_cate = 3
'''                                                 Name_in_caffe
features.0.weight     -->  features.0.weight            conv1
features.0.bias       -->  features.0.bias
features.3.weight     -->  features.3.weight            conv2
features.3.bias       -->  features.3.bias
features.6.weight     -->  features.6.weight            conv3
features.6.bias       -->  features.6.bias
features.8.weight     -->  features.8.weight            conv4
features.8.bias       -->  features.8.bias
features.10.weight    -->  features.10.weight           conv5
features.10.bias      -->  features.10.bias
classifier.1.weight   -->  classifier.1.weight          fc6
classifier.1.bias     -->  classifier.1.bias
classifier.4.weight   -->  classifier.4.weight          fc7
classifier.4.bias     -->  classifier.4.bias
classifier.6.weight   -->  classifier.6.weight          fc8
classifier.6.bias     -->  classifier.6.bias
'''

class AlexNet_Trunk(nn.Module):
    def __init__(self, init_weights=True):
        super(AlexNet_Trunk, self).__init__()
        self.net_arch = 'alexnet'

        #-- Trunk architecture
        #-- Convolutional layers
        self.Convs = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)   , #[ 0] conv1  Conv1   TO CHECK: alexnet is 96 instead of 64 here.
            nn.ReLU(inplace=True)                                   , #[ 1] relu1
            nn.MaxPool2d(kernel_size=3, stride=2)                   , #[ 2] pool1
            LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=1)        , #[ 3] norm1
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)  , #[ 4] conv2  Conv2
            nn.ReLU(inplace=True)                                   , #[ 5] relu2
            nn.MaxPool2d(kernel_size=3, stride=2)                   , #[ 6] pool2
            LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=1)        , #[ 7] norm2
            nn.Conv2d(256, 384, kernel_size=3, padding=1)           , #[ 8] conv3  Conv3
            nn.ReLU(inplace=True)                                   , #[ 9] relu3
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2) , #[10] conv4  Conv4
            nn.ReLU(inplace=True)                                   , #[11] relu4
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2) , #[12] conv5  Conv5
            nn.ReLU(inplace=True)                                   , #[13] relu5
            nn.MaxPool2d(kernel_size=3, stride=2)                   , #[14] pool5
        )
        #-- Fully connected layers
        self.Fcs = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096)                            , #[0] fc6    Fc6
            nn.ReLU(inplace=True)                                   , #[1] relu6
            nn.Dropout()                                            , #[2] drop6  TO CHECK: alexnet Dropout should follow after Fc
            nn.Linear(4096, 4096)                                   , #[3] fc7    Fc7
            nn.ReLU(inplace=True)                                   , #[4] relu7
            nn.Dropout()                                            , #[5] drop7
        )

        if init_weights==True:  # for legacy
            self.init_weights(pretrained='caffemodel')
        elif isinstance(init_weights, str) or init_weights is None:
            self.init_weights(pretrained=init_weights)
        else:
            raise NotImplementedError

    #@Interface
    def forward(self, x):  # In-replace forward_trunk
        """ x is input image data.
            x is of shape:  (batchsize, 3, 227, 227)
        """
        # forward convolutional layers
        x = self.Convs(x)
        #
        # forward fully connected layers
        batchsize = x.size(0)   # .split(1, dim=1)
        x = x.view(batchsize, 256 * 6 * 6)
        x = self.Fcs(x)
        #
        return x


    def init_weights(self, pretrained='caffemodel'):
        """ Two ways to init weights:
            1) by copying pretrained weights.
            2) by filling empirical  weights. (e.g. gaussian, xavier, uniform, constant, bilinear).
        """
        if pretrained is None:
            print('initialize weights by filling (Fc:gaussian, Conv:kaiming_normal).')
            init_weights_by_filling(self)
        elif pretrained=='caffemodel':
            print("Initializing weights by copying (pretrained caffe weights).")
            src2dsts = dict(conv1='Convs.0' , conv2='Convs.4' , conv3='Convs.8',
                            conv4='Convs.10', conv5='Convs.12', fc6='Fcs.0', fc7='Fcs.3')
            copy_weights(self.state_dict(), 'caffemodel.alexnet', src2dsts=src2dsts)
        elif pretrained=='torchmodel':
            raise NotImplementedError
        else:
            raise NotImplementedError
        return self


    # def _init_weights_by_filling(self):  # _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Linear):
    #             m.weight.data.normal_(0, 0.01)
    #             m.bias.data.zero_()

    def fix_conv1_conv2(self):
        # Fix the layers before conv3:
        for layer in range(8):
            for p in self.Convs[layer].parameters():
                p.requires_grad = False



class Test_AlexNet(nn.Module):

    def __init__(self, nr_cate=3,  _Trunk=AlexNet_Trunk):
        super(Test_AlexNet, self).__init__()
        self.truck = _Trunk(init_weights=True)

        self.nr_cate = nr_cate
        self.maskout = Maskout(nr_cate=nr_cate)


    def forward(self, x, label):
        x = self.truck(x)
        # your code here
        batchsize = x.size(0)

        #-- Head architecture
        self.head_s2 = nn.Sequential(
            nn.Linear(4096, 84),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Linear(84,  self.nr_cate*3),   # 252=3*3
            )
        self.head_s1 = nn.Sequential(
            nn.Linear(4096, 84),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Linear(84,  self.nr_cate*2),
            )
        # self.maskout = Maskout(nr_cate=nr_cate)

        # Note: s1(x,y) is on a circle and (x^2+y^2=1)
        x_s2 = self.maskout(self.head_s2(x).view(batchsize, self.nr_cate, 3) , label)
        x_s1 = self.maskout(self.head_s1(x).view(batchsize, self.nr_cate, 2) , label)
        #-- Normalize coordinate to a unit
        x_s2 = norm2unit(x_s2) #, nr_cate=self.nr_cate)
        x_s1 = norm2unit(x_s1) #, nr_cate=self.nr_cate)

        Pred = edict(s2=x_s2, s1=x_s1)
        return Pred


if __name__ == '__main__':
    model = Test_AlexNet()

    # import numpy as np
    dummy_batch_data  = np.zeros((2,3,227,227),dtype=np.float32)
    dummy_batch_label = np.zeros((2,1),        dtype=np.int64)
    dummy_batch_data  = torch.autograd.Variable( torch.from_numpy(dummy_batch_data ) )
    dummy_batch_label = torch.autograd.Variable( torch.from_numpy(dummy_batch_label) )

    Pred = model(dummy_batch_data, dummy_batch_label)
    print (Pred.s2)


