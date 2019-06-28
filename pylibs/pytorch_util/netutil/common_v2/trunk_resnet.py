import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from pytorch_util.libtrain import init_weights_by_filling, cfg as pretrained_cfg

# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
__all__ = ['_ResNet_Trunk'] #, 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18' : 'https://download.pytorch.org/models/resnet18-5c106cde.pth' ,
    'resnet34' : 'https://download.pytorch.org/models/resnet34-333f7ec4.pth' ,
    'resnet50' : 'https://download.pytorch.org/models/resnet50-19c8e357.pth' ,
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# def get_func_xNorm(xNorm='batch', affine=True):
def _nn_xNorm(num_channels, xNorm='BN', **kwargs):  # kwargs:  BN: {affine=False},   LN: {elementwise_affine=True}
    #---By Shuai--    num_features = num_channels ?
    """ E.g.
         Fixed BN:   xNorm='BN', affine=False
    """
    # print "   _nn_xNorm: ", xNorm, kwargs
    if   xNorm=='BN':
        return nn.BatchNorm2d(num_channels, **kwargs) # affine=affine)
    elif xNorm=='GN':
        return nn.GroupNorm(num_groups=32, num_channels=num_channels, **kwargs) # affine=affine) # num_channels,
    elif xNorm=='IN':
        return nn.InstanceNorm2d(num_channels, **kwargs) # affine=False) # default is affine=False
    elif xNorm=='LN':
        # TODO  how to calculate normalized_shape?
        # return nn.LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)
        raise NotImplementedError


class nnAdd(nn.Module):
    def forward(self, x0, x1):
        return x0 + x1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, xNorm='BN', xNorm_kwargs={}):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1   = _nn_xNorm(planes, xNorm=xNorm, **xNorm_kwargs) # nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = _nn_xNorm(planes, xNorm=xNorm, **xNorm_kwargs) # nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        #---By Shuai--
        self.merge = nnAdd()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # out += residual
        out = self.merge(out, residual)
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, xNorm='BN', xNorm_kwargs={}):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1   = _nn_xNorm(planes, xNorm=xNorm, **xNorm_kwargs) # nn.BatchNorm2d(planes) # Note: pytorch 'BatchNorm2d' include: BatchNorm+Scale
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = _nn_xNorm(planes, xNorm=xNorm, **xNorm_kwargs) # nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3   = _nn_xNorm(planes * 4, xNorm=xNorm, **xNorm_kwargs) # nn.BatchNorm2d(planes * 4)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        #---By Shuai--
        self.merge = nnAdd()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # out += residual
        out = self.merge(out, residual)
        out = self.relu(out)

        return out



type2blockFunc_layerCfgs = dict(
    resnet18  = (BasicBlock, [2, 2,  2, 2]),
    resnet34  = (BasicBlock, [3, 4,  6, 3]),
    resnet50  = (Bottleneck, [3, 4,  6, 3]),
    resnet101 = (Bottleneck, [3, 4, 23, 3]),
    resnet152 = (Bottleneck, [3, 8, 36, 3]),
    )


class _ResNet_Trunk(nn.Module):

    def __init__(self, res_type='resnet101', pretrained='torchmodel', # True,  # pretrained = {True/False, caffemodel, torchmodel}
                       # init_weights=None,     # [Warning]  deprecated!   for legacy
                       start=None, end=None, xNorm='BN', xNorm_kwargs={}):
        self.inplanes = 64
        super(_ResNet_Trunk, self).__init__()
        self.net_arch = res_type  #@Shine: added

        blockFunc, layerCfgs = type2blockFunc_layerCfgs[res_type]
        #---------------------------------------------------------------[net definition]
        # For cifar:  self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1  = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) # conv1
        self.bn1    = _nn_xNorm(64, xNorm=xNorm, **xNorm_kwargs)   #nn.BatchNorm2d(64) # bn_conv1  + scale_conv1
        self.relu   = nn.ReLU(inplace=True)                                            # conv1_relu
        self.maxpool= nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                 # pool1
        #-#
        self.layer1 = self._make_layer(blockFunc,  64, layerCfgs[0], stride=1, xNorm='BN', xNorm_kwargs={})            # res2{a,b,c}
        self.layer2 = self._make_layer(blockFunc, 128, layerCfgs[1], stride=2, xNorm='BN', xNorm_kwargs={})  # res3{a,b1-b3}  res3b3_relu  [B, 512, 28, 28]
        self.layer3 = self._make_layer(blockFunc, 256, layerCfgs[2], stride=2, xNorm='BN', xNorm_kwargs={})  # res4{a,b1-b22} output  [B, 1024, 14, 14]
        self.layer4 = self._make_layer(blockFunc, 512, layerCfgs[3], stride=2, xNorm='BN', xNorm_kwargs={})  # res5{a,b,c}    output  [B, 2048,  7,  7]
        #-#
        self.pool5 = nn.AvgPool2d(7, stride=1)                        # output  [B, 2048,  1,  1]  @Shine: used to be self.avgpool
        #-# self.fc = nn.Linear(512 * block.expansion, num_classes)   #                            @Shine: fc1000 removed
        #---------------------------------------------------------------
        self._layer_names = [name for name, module in self.named_children()]
        # same as: [ 'conv1', 'bn1', 'relu', 'maxpool',
        #            'layer1','layer2','layer3','layer4','pool5' ]

        if pretrained==True:
            pretrained = 'torchmodel'
        if pretrained==False:
            pretrained = None
        #
        assert pretrained in [None, 'caffemodel', 'torchmodel'], "Unknown pretrained: %s" % pretrained
        print ('[_ResNet_Trunk] init_weights:', pretrained)
        self.init_weights(pretrained=pretrained)

        # build a short cut by sequence
        self.truck_seq = self.sub_seq(start=start, end=end)  # 'conv1', 'pool5'

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
        return _seq


    def _make_layer(self, block, planes, blocks, stride, xNorm, xNorm_kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(planes * block.expansion),
                _nn_xNorm(planes * block.expansion, xNorm=xNorm, **xNorm_kwargs)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,  xNorm=xNorm, xNorm_kwargs=xNorm_kwargs))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, xNorm=xNorm, xNorm_kwargs=xNorm_kwargs))

        return nn.Sequential(*layers)


    #@Interface
    def forward(self, x):  # In-replace forward_trunk
        """ x is input image data.
            x is of shape:  (batchsize, 3, 224, 224)  """
        assert x.size()[1:]==(3,224,224), "resnet need (3,224,224) input data, whereas %s is received."% str(tuple(x.size()[1:]))
        # return self.truck_seq(x)
        if self.selected_layer_name[-1]=='pool5':
            return self.truck_seq(x).view(x.size(0), -1)  # view as (batchsize,2048)
        else:
            return self.truck_seq(x)


    # @staticmethod
    def init_weights(self, pretrained='caffemodel'):
        """ Two ways to init weights:
            1) by copying pretrained weights.
            2) by filling empirical  weights. (e.g. gaussian, xavier, uniform, constant, bilinear).
        """
        if pretrained is None:  # [Warning]  deprecated!   for legacy
            print('initialize weights by filling (Fc:gaussian, Conv:kaiming_normal).')
            init_weights_by_filling(self, silent=False)  # (self, gaussian_std=0.05, kaiming_normal=True)
        elif pretrained=='caffemodel':
            model_path = pretrained_cfg.caffemodel.resnet101.model
            print("Loading caffe pretrained weights from %s" %(model_path))
            state_dict = torch.load(model_path)
            # initialize  weights by copying
            self.load_state_dict({k:v for k,v in state_dict.items() if k in self.state_dict()})
        elif pretrained=='torchmodel':
            print("Loading pytorch pretrained weights from %s" %(model_urls[self.net_arch]))
            state_dict = model_zoo.load_url(model_urls[self.net_arch], progress=True)
            self.load_state_dict(state_dict, strict=False)
        else:
            raise NotImplementedError
        return self


class ResNet18_Trunk(_ResNet_Trunk):
    def __init__(self, **kwargs):
        _ResNet_Trunk.__init__(self, res_type='resnet18', **kwargs)

class ResNet34_Trunk(_ResNet_Trunk):
    def __init__(self, **kwargs):
        _ResNet_Trunk.__init__(self, res_type='resnet34', **kwargs)

class ResNet50_Trunk(_ResNet_Trunk):
    def __init__(self, **kwargs):
        _ResNet_Trunk.__init__(self, res_type='resnet50', **kwargs)

class ResNet101_Trunk(_ResNet_Trunk):
    def __init__(self, **kwargs):
        _ResNet_Trunk.__init__(self, res_type='resnet101', **kwargs)

class ResNet152_Trunk(_ResNet_Trunk):
    def __init__(self, **kwargs):
        _ResNet_Trunk.__init__(self, res_type='resnet152', **kwargs)




import numpy as np
from easydict import EasyDict as edict
from pytorch_util.torch_v4_feature import LocalResponseNorm # *
from pytorch_util.torch_3rd_layers import Maskout
from pytorch_util.torch_3rd_funcs  import norm2unit

class Test_Net(nn.Module):

    def __init__(self, nr_cate=3,  _Trunk=ResNet101_Trunk):
        super(Test_Net, self).__init__()
        self.truck = _Trunk()  # or _Trunk(end='pool5')

        self.nr_cate = nr_cate

        #-- Head architecture
        self.head_s2 = nn.Sequential(
            nn.Linear(2048, 84),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Linear(84,  self.nr_cate*3),   # 252=3*3
            )
        self.head_s1 = nn.Sequential(
            nn.Linear(2048, 84),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Linear(84,  self.nr_cate*2),
            )
        self.maskout = Maskout(nr_cate=nr_cate)

        init_weights_by_filling(self.head_s2)
        init_weights_by_filling(self.head_s1)



    def forward(self, x, label):
        # forward truck
        x = self.truck(x)
        # your code here
        batchsize = x.size(0)
        x = x.view(batchsize, -1)  # viewed as (batchsize, 2048)

        # Note: s1(x,y) is on a circle and (x^2+y^2=1)
        x_s2 = self.maskout(self.head_s2(x).view(batchsize, self.nr_cate, 3) , label)
        x_s1 = self.maskout(self.head_s1(x).view(batchsize, self.nr_cate, 2) , label)
        #-- Normalize coordinate to a unit
        x_s2 = norm2unit(x_s2) #, nr_cate=self.nr_cate)
        x_s1 = norm2unit(x_s1) #, nr_cate=self.nr_cate)

        Pred = edict(s2=x_s2, s1=x_s1)
        return Pred


if __name__ == '__main__':

    model = Test_Net()  #.copy_weights() # pretrained=True

    # import numpy as np
    dummy_batch_data  = np.zeros((2,3,224,224),dtype=np.float32)
    dummy_batch_label = np.zeros((2,1),        dtype=np.int64)
    dummy_batch_data  = torch.autograd.Variable( torch.from_numpy(dummy_batch_data ) )
    dummy_batch_label = torch.autograd.Variable( torch.from_numpy(dummy_batch_label) )

    Pred = model(dummy_batch_data, dummy_batch_label)
    print (Pred.s2)

