import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from collections import OrderedDict as odict

from pytorch_util.torch_v4_feature import LocalResponseNorm # *
from pytorch_util.libtrain import copy_weights, init_weights_by_filling



__all__ = ['_VGG_Trunk', 'VGGM_Trunk', 'VGG16_Trunk', 'VGG19_Trunk'] # , 'alexnet']

# __all__ = [
#     'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
#     'vgg19_bn', 'vgg19',
# ]


# model_urls = {
#     'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
#     'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
#     'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
#     'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
#     'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
#     'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
#     'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
#     'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
# }

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


#@Shine: [ref] https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/vggm.py
#              https://gist.github.com/ksimonyan/f194575702fae63b2829#file-vgg_cnn_m_deploy-prototxt
def make_layers_vggm():
    return nn.Sequential(
            nn.Conv2d(3,96,(7, 7),(2, 2)),
            nn.ReLU(inplace=True),
            LocalResponseNorm(5, 0.0005, 0.75, 2),  # note that k=2 (default is k=1, e.g. alexnet)
            nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(96,256,(5, 5),(2, 2),(1, 1)),
            nn.ReLU(inplace=True),
            LocalResponseNorm(5, 0.0005, 0.75, 2), # note that k=2 (default is k=1, e.g. alexnet)
            nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
    )

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

type2cfg = dict(vgg16=cfg['D'], vgg19=cfg['E'])


def get_caffeNaming(cfg_list, learnable_only=True):
    all_names = []
    #---------- Convolution layers ----------
    # All block ending by Max Pooling, so first find indices of pooling
    maxpool_inds = [i for i, x in enumerate(cfg_list) if x == "M"]
    assert len(maxpool_inds)==5
    start,end = 0, 0
    for i in range(5):
        block_id = i+1
        start=end
        end  =maxpool_inds[i]+1
        block_cfg = cfg_list[start:end]
        block_names = []
        for j in range(len(block_cfg)-1):
            conv_id = j+1
            block_names.append('conv{}_{}'.format(block_id, conv_id))
            if not learnable_only:
                block_names.append('relu{}_{}'.format(block_id, conv_id))
        if not learnable_only:
            block_names.append('pool{}'.format(block_id))
        all_names += block_names
        print (block_names)
        print ('--------------------------')

    #---------- Fully connected layers ----------
    if learnable_only:
        all_names += ['fc6', 'fc7']
    else:
        all_names += ['fc6', 'relu6', 'drop6',
                      'fc7', 'relu7', 'drop7',]
                # 'fc8']
    return all_names

#print get_caffeNaming(type2cfg['vgg16'])
#exit()



def get_caffeSrc2Dst(net_arch='vgg16'):
    #@Interface
    if   net_arch=='vgg16':
        return odict(conv1_1='features.0' , conv1_2='features.2' ,
                     conv2_1='features.5' , conv2_2='features.7' ,
                     conv3_1='features.10', conv3_2='features.12', conv3_3='features.14',
                     conv4_1='features.17', conv4_2='features.19', conv4_3='features.21',
                     conv5_1='features.24', conv5_2='features.26', conv5_3='features.28',
                     fc6='classifier.0', fc7='classifier.3')
    elif net_arch=='vgg19':
        return odict(conv1_1='features.0' , conv1_2='features.2' ,
                     conv2_1='features.5' , conv2_2='features.7' ,
                     conv3_1='features.10', conv3_2='features.12', conv3_3='features.14', conv3_4='features.16',
                     conv4_1='features.19', conv4_2='features.21', conv4_3='features.23', conv4_4='features.25',
                     conv5_1='features.28', conv5_2='features.30', conv5_3='features.32', conv5_4='features.34',
                     fc6='classifier.0', fc7='classifier.3')
    elif net_arch in ['vgg16_bn', 'vgg19_bn']:
        print ('No pretrained model for vgg16_bn, vgg19_bn.')
        raise NotImplementedError
    elif net_arch=='vggm':
        return odict(conv1='features.0'  ,   # (96, 3, 7, 7)              0.1 M    (96,)                      0.0 M
                     conv2='features.4'  ,   # (256, 96, 5, 5)            2.3 M    (256,)                     0.0 M
                     conv3='features.8'  ,   # (512, 256, 3, 3)           4.5 M    (512,)                     0.0 M
                     conv4='features.10' ,   # (512, 512, 3, 3)           9.0 M    (512,)                     0.0 M
                     conv5='features.12' ,   # (512, 512, 3, 3)           9.0 M    (512,)                     0.0 M
                     fc6  ='classifier.0',   # (4096, 18432)            288.0 M    (4096,)                    0.0 M
                     fc7  ='classifier.3', ) # (4096, 4096)              64.0 M    (4096,)                    0.0 M
    else:
        raise NotImplementedError

'''
'D':           vgg16                          'E':           vgg19
-------------------------------------         -------------------------------------
conv1_1     64                   [ 0]         conv1_1     64                   [ 0]
relu1_1                          [ 1]         relu1_1                          [ 1]
conv1_2     64                   [ 2]         conv1_2     64                   [ 2]
relu1_2                          [ 3]         relu1_2                          [ 3]
pool1       'M'    (max pooling) [ 4]         pool1       'M'    (max pooling) [ 4]
-------------------------------------         -------------------------------------
conv2_1     128                  [ 5]         conv2_1     128                  [ 5]
relu2_1                          [ 6]         relu2_1                          [ 6]
conv2_2     128                  [ 7]         conv2_2     128                  [ 7]
relu2_2                          [ 8]         relu2_2                          [ 8]
pool2       'M'     (max pooling)[ 9]         pool2       'M'     (max pooling)[ 9]
-------------------------------------         -------------------------------------
conv3_1     256                  [10]         conv3_1     256                  [10]
relu3_1                          [11]         relu3_1                          [11]
conv3_2     256                  [12]         conv3_2     256                  [12]
relu3_2                          [13]         relu3_2                          [13]
conv3_3     256                  [14]         conv3_3     256                  [14]
relu3_3                          [15]         relu3_3                          [15]
pool3       'M'     (max pooling)[16]         conv3_4     256                  [16]
-------------------------------------         relu3_4                          [17]
conv4_1     512                  [17]         pool3       'M'     (max pooling)[18]
relu4_1                          [18]         -------------------------------------
conv4_2     512                  [19]         conv4_1     512                  [19]
relu4_2                          [20]         relu4_1                          [20]
conv4_3     512                  [21]         conv4_2     512                  [21]
relu4_3                          [22]         relu4_2                          [22]
pool4       'M'     (max pooling)[23]         conv4_3     512                  [23]
-------------------------------------         relu4_3                          [24]
conv5_1     512                  [24]         conv4_4     512                  [25]
relu5_1                          [25]         relu4_4                          [26]
conv5_2     512                  [26]         pool4       'M'     (max pooling)[27]
relu5_2                          [27]         -------------------------------------
conv5_3     512                  [28]         conv5_1     512                  [28]
relu5_3                          [29]         relu5_1                          [29]
pool5       'M'     (max pooling)[30]         conv5_2     512                  [30]
-------------------------------------         relu5_2                          [31]
                                              conv5_3     512                  [32]
                                              relu5_3                          [33]
                                              conv5_4     512                  [34]
                                              relu5_4                          [35]
                                              pool5       'M'     (max pooling)[36]
                                              -------------------------------------
'''



class _VGG_Trunk(nn.Module):

    def __init__(self, vgg_type='vgg16', init_weights=True):
        super(_VGG_Trunk, self).__init__()
        self.net_arch = vgg_type

        if vgg_type=='vggm':
            # Convolutional layers
            self.features = make_layers_vggm()    # for vgg16, vgg19, vgg16_bn, vgg19_bn
            # Fully connected layers
            self.classifier = nn.Sequential(
                nn.Linear(512 * 6 * 6, 4096),   # [0]   Note: 512 x 6 x 6 for vgg-m
                nn.ReLU(True),                  # [1]
                nn.Dropout(),                   # [2]
                nn.Linear(4096, 4096),          # [3]
                nn.ReLU(True),                  # [4]
                nn.Dropout(),                   # [5]
                # nn.Linear(4096, num_classes), # @Shine  prob removed
            )
        else:
            # Convolutional layers
            self.features = make_layers(type2cfg[vgg_type])    # for vgg16, vgg19, vgg16_bn, vgg19_bn
            # Fully connected layers
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),   # [0]
                nn.ReLU(True),                  # [1]
                nn.Dropout(),                   # [2]
                nn.Linear(4096, 4096),          # [3]
                nn.ReLU(True),                  # [4]
                nn.Dropout(),                   # [5]
                # nn.Linear(4096, num_classes), # @Shine  prob removed
            )

        #if init_weights:
        #    self.init_weights(pretrained='caffemodel' if (self.net_arch in ['vgg16', 'vgg19']) else None)
        if init_weights==True:  # for legacy
            assert self.net_arch in ['vgg16', 'vgg19']
            self.init_weights(pretrained='caffemodel')
        elif isinstance(init_weights, str) or init_weights is None:
            self.init_weights(pretrained=init_weights)
        else:
            raise NotImplementedError

        #[TODO] build a short cut by sequence: e.g.
        # self.truck_seq = self.sub_seq(start='conv1', end='conv5')

    # [TODO] To implement (see trunk_resnet.py)
    """
    def sub_seq(self, start=None, end=None):   # 'conv1', 'pool5'
        # select sub-sequence from trunk (by creating a shot cut).
        assert start is None or start in self._layer_names, '[Error] %s is not in %s' % (start, self._layer_names)
        assert end   is None or end   in self._layer_names, '[Error] %s is not in %s' % (end  , self._layer_names)
        start_ind = self._layer_names.index(start) if (start is not None) else 0
        end_ind   = self._layer_names.index(end)   if (end   is not None) else len(self._layer_names)-1
        assert start_ind<=end_ind
        self.selected_layer_name = self._layer_names[start_ind:end_ind+1]
        print("Selected sub-sequence: %s" % self.selected_layer_name)
        _seq = nn.Sequential(*[self.__getattr__(x) for x in self.selected_layer_name])
        return _seq     # (self.conv1, self.bn1, self.relu, self.maxpool, self.layer1,self.layer2,self.layer3 )
    """


    #@Interface
    def forward(self, x):  # In-replace forward_trunk
        """ x is input image data.
            x is of shape:  (batchsize, 3, 224, 224)  """
        # forward convolutional layers
        x = self.features(x)
        # forward fully connected layers
        batchsize = x.size(0)   # .split(1, dim=1)
        x = x.view(batchsize, -1)
        x = self.classifier(x)
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
            src2dsts = get_caffeSrc2Dst(self.net_arch)
            copy_weights(self.state_dict(), 'caffemodel.%s'%(self.net_arch), src2dsts=src2dsts)
        elif pretrained=='torchmodel':
            raise NotImplementedError
        else:
            raise NotImplementedError
        return self


    def fix_conv1_conv2(self):
        # Fix the layers before conv3:
        for layer in range(10):
            for p in self.features[layer].parameters():
                p.requires_grad = False


class VGGM_Trunk(_VGG_Trunk):
    def __init__(self, init_weights=True):
        super(VGGM_Trunk, self).__init__(vgg_type='vggm')

class VGG16_Trunk(_VGG_Trunk):
    def __init__(self, init_weights=True):
        super(VGG16_Trunk, self).__init__(vgg_type='vgg16')


class VGG19_Trunk(_VGG_Trunk):
    def __init__(self, init_weights=True):
        super(VGG19_Trunk, self).__init__(vgg_type='vgg19')



#-# def vgg11(pretrained=False, **kwargs):
#-#     """VGG 11-layer model (configuration "A")
#-#
#-#     Args:
#-#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#-#     """
#-#     if pretrained:
#-#         kwargs['init_weights'] = False
#-#     model = VGG_Trunk(make_layers(cfg['A']), **kwargs)
#-#     if pretrained:
#-#         model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
#-#     return model
#-#
#-#
#-# def vgg11_bn(pretrained=False, **kwargs):
#-#     """VGG 11-layer model (configuration "A") with batch normalization
#-#
#-#     Args:
#-#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#-#     """
#-#     if pretrained:
#-#         kwargs['init_weights'] = False
#-#     model = VGG_Trunk(make_layers(cfg['A'], batch_norm=True), **kwargs)
#-#     if pretrained:
#-#         model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
#-#     return model
#-#
#-#
#-# def vgg13(pretrained=False, **kwargs):
#-#     """VGG 13-layer model (configuration "B")
#-#
#-#     Args:
#-#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#-#     """
#-#     if pretrained:
#-#         kwargs['init_weights'] = False
#-#     model = VGG_Trunk(make_layers(cfg['B']), **kwargs)
#-#     if pretrained:
#-#         model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
#-#     return model
#-#
#-#
#-# def vgg13_bn(pretrained=False, **kwargs):
#-#     """VGG 13-layer model (configuration "B") with batch normalization
#-#
#-#     Args:
#-#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#-#     """
#-#     if pretrained:
#-#         kwargs['init_weights'] = False
#-#     model = VGG_Trunk(make_layers(cfg['B'], batch_norm=True), **kwargs)
#-#     if pretrained:
#-#         model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
#-#     return model


# def vgg16(pretrained=False, **kwargs):
#     """VGG 16-layer model (configuration "D")
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     if pretrained:
#         kwargs['init_weights'] = False
#     model = VGG_Trunk(make_layers(cfg['D']), **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
#     return model
#
#
# def vgg16_bn(pretrained=False, **kwargs):
#     """VGG 16-layer model (configuration "D") with batch normalization
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     if pretrained:
#         kwargs['init_weights'] = False
#     model = VGG_Trunk(make_layers(cfg['D'], batch_norm=True), **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
#     return model
#
#
# def vgg19(pretrained=False, **kwargs):
#     """VGG 19-layer model (configuration "E")
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     if pretrained:
#         kwargs['init_weights'] = False
#     model = VGG_Trunk(make_layers(cfg['E']), **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
#     return model
#
#
# def vgg19_bn(pretrained=False, **kwargs):
#     """VGG 19-layer model (configuration 'E') with batch normalization
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     if pretrained:
#         kwargs['init_weights'] = False
#     model = VGG_Trunk(make_layers(cfg['E'], batch_norm=True), **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
#     return model
#

#import torch.nn as nn
#import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from easydict import EasyDict as edict
# import math

from pytorch_util.torch_3rd_layers import Maskout
from pytorch_util.torch_3rd_funcs  import norm2unit

class Test_Net(nn.Module):

    def __init__(self, nr_cate=3,  _Trunk=VGG16_Trunk):
        super(Test_Net, self).__init__()
        self.truck = _Trunk()  #.copy_weights()

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
    model = Test_Net(_Trunk=VGGM_Trunk)  #.copy_weights() # pretrained=True

    # import numpy as np
    dummy_batch_data  = np.zeros((2,3,224,224),dtype=np.float32)
    dummy_batch_label = np.zeros((2,1),        dtype=np.int64)
    dummy_batch_data  = torch.autograd.Variable( torch.from_numpy(dummy_batch_data ) )
    dummy_batch_label = torch.autograd.Variable( torch.from_numpy(dummy_batch_label) )

    print (model)
    exit()
    Pred = model(dummy_batch_data, dummy_batch_label)
    print( Pred.s2)
