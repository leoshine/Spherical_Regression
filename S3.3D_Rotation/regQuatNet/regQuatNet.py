# coding: utf8
"""
 @Author  : Shuai Liao
"""

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from   torch.autograd import Variable
import torch
import torch.nn.functional as F
from   basic.common import rdict
import numpy as np
from   easydict import EasyDict as edict
from   collections import OrderedDict as odict
from   itertools import product

from pytorch_util.netutil.common_v2.trunk_alexnet_bvlc import AlexNet_Trunk
from pytorch_util.netutil.common_v2.trunk_vgg          import VGG16_Trunk
from pytorch_util.netutil.common_v2.trunk_resnet       import ResNet101_Trunk, ResNet50_Trunk

net_arch2Trunk = dict(
    alexnet  = AlexNet_Trunk,
    vgg16    = VGG16_Trunk,
    resnet101= ResNet101_Trunk,
    resnet50 = ResNet50_Trunk
)

from pytorch_util.libtrain import copy_weights, init_weights_by_filling
from pytorch_util.torch_v4_feature import LocalResponseNorm # *
from pytorch_util.torch_3rd_layers import Maskout
from pytorch_util.torch_3rd_funcs  import norm2unit, exp_Normalization


from basic.common import env, add_path
from lib.helper import *


loss_balance = 4.

def cls_pred(output, topk=(1,)):
    maxk = max(topk)
    batch_size = output.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    return pred

def reg2d_pred2tgt(pr_sin, pr_cos):
    theta = torch.atan2(pr_sin, pr_cos)
    return theta


class _BaseReg_Net(nn.Module):
    #
    @staticmethod
    def head_seq(in_size, reg_n_D, nr_cate=12, nr_fc8=334, init_weights=True):  # in_size=4096
        seq = nn.Sequential(
                nn.Linear(in_size, nr_fc8),                             # Fc8
                nn.ReLU(inplace=True),
                #nn.Dropout(),
                nn.Linear(nr_fc8, nr_cate*reg_n_D),                     # Prob
             )
        if init_weights:
            init_weights_by_filling(seq, gaussian_std=0.005, kaiming_normal=True)  # fill weight with gaussian filler
        return seq

    """BVLC alexnet architecture (Note: slightly different from pytorch implementation.)"""
    def __init__(self, nr_cate=12, net_arch='alexnet', init_weights=True):  # AlexNet_Trunk
        super(_BaseReg_Net, self).__init__()
        _Trunk = net_arch2Trunk[net_arch]
        self.trunk = _Trunk(init_weights=init_weights)

        self.nr_cate = nr_cate
        self.top_size = 4096 if not self.trunk.net_arch.startswith('resnet') else 2048

    def forword(self, x, label):
        raise NotImplementedError





#---------------------------------------------------------------------[reg_Direct]
class reg_Direct_Net(_BaseReg_Net):   # No L2 norm at all,
    """ No any L2 normalization to guarantee prediction is on n-sphere, smooth l1 loss is used. """

    """BVLC alexnet architecture (Note: slightly different from pytorch implementation.)"""
    def __init__(self, nr_cate=12, net_arch='alexnet', init_weights=True):
        _BaseReg_Net.__init__(self, nr_cate=nr_cate, net_arch=net_arch, init_weights=init_weights)

        self.nr_cate = nr_cate
        self.reg_n_D = 4

        #-- Head architecture
        # Note: for quaternion, there's only one regression head (instead of 3 Euler angles (a,e,t)).
        #       Thus, nr_fc8=996  (see design.py)
        self.head_quat = self.head_seq(self.top_size, self.reg_n_D, nr_cate=nr_cate, nr_fc8=996, init_weights=init_weights)

        # for maskout specific category
        self.maskout = Maskout(nr_cate=nr_cate)

        # loss module
        self.loss_handler = Smooth_L1_Loss_Handler()
        self.targets = ['quat']


    def forward(self, x, label):
        """label shape (batchsize, )  """
        x = self.trunk(x)           # Forward Conv and Fc6,Fc7
        #
        batchsize = x.size(0)       # .split(1, dim=1)

        # Note: quat(a,b,c,d) is on a 4d sphere and (x^2+y^2=1)
        x_quat = self.maskout(self.head_quat(x).view(batchsize, self.nr_cate, self.reg_n_D) , label)
        #-- Normalize coordinate to a unit
        # x_quat = norm2unit(x_quat) #, nr_cate=self.nr_cate)

        Prob = edict(quat=x_quat)
        return Prob

    def compute_loss(self, Prob, GT):
        Loss = self.loss_handler.compute_loss(self.targets, Prob, GT)
        return Loss

    @staticmethod
    def compute_pred(Prob):
        x_quat = Prob['quat']
        #-- Normalize coordinate to a unit
        x_quat = norm2unit(x_quat) # Note: here we do l2 normalization. Just to make predicted quaternion a unit norm.
        #
        batchsize = x_quat.size(0)
        # Get cpu data.
        batch_data = x_quat.data.cpu().numpy().copy()
        assert batch_data.shape==(batchsize, 4), batch_data.shape
        Pred = edict(quat=batch_data)
        return Pred
#---------------------------------------------------------------------




#---------------------------------------------------------------------[reg_Sflat]
class reg_Sflat_Net(_BaseReg_Net):
    """ L2 normalization activation, with cosine proximity loss. """

    """BVLC alexnet architecture (Note: slightly different from pytorch implementation.)"""
    def __init__(self, nr_cate=12, net_arch='alexnet', init_weights=True):
        _BaseReg_Net.__init__(self, nr_cate=nr_cate, net_arch=net_arch, init_weights=init_weights)

        self.nr_cate = nr_cate
        self.reg_n_D = 4

        #-- Head architecture
        # Note: for quaternion, there's only one regression head (instead of 3 (for a,e,t)).
        #       Thus, nr_fc8=996  (see design.py)
        self.head_quat = self.head_seq(self.top_size, self.reg_n_D, nr_cate=nr_cate, nr_fc8=996, init_weights=init_weights)

        # for maskout a,e,t
        self.maskout = Maskout(nr_cate=nr_cate)

        # loss module
        self.loss_handler  = Cos_Proximity_Loss_Handler()
        self.targets = ['quat']


    def forward(self, x, label):
        """label shape (batchsize, )  """
        x = self.trunk(x)
        #
        batchsize = x.size(0)

        # Note: quat(a,b,c,d) is on a 4d sphere and (x^2+y^2=1)
        x_quat = self.maskout(self.head_quat(x).view(batchsize, self.nr_cate, self.reg_n_D) , label)
        #-- Normalize coordinate to a unit
        x_quat = norm2unit(x_quat) #, nr_cate=self.nr_cate)

        Prob = edict(quat=x_quat)
        return Prob

    def compute_loss(self, Prob, GT):
        Loss = self.loss_handler.compute_loss(self.targets, Prob, GT)
        return Loss

    @staticmethod
    def compute_pred(Prob):
        x_quat = Prob['quat']
        batchsize = x_quat.size(0)
        # Get cpu data.
        batch_data = x_quat.data.cpu().numpy().copy()
        assert batch_data.shape==(batchsize, 4), batch_data.shape
        Pred = edict(quat=batch_data)
        return Pred
#---------------------------------------------------------------------









#---------------------------------------------------------------------[reg_Sexp]
class reg_Sexp_Net(_BaseReg_Net):
    """ Spherical exponential activation + Sign classification, with cosine proximity loss """

    """BVLC alexnet architecture (Note: slightly different from pytorch implementation.)"""
    def __init__(self, nr_cate=12, net_arch='alexnet', init_weights=True):
        _BaseReg_Net.__init__(self, nr_cate=nr_cate, net_arch=net_arch, init_weights=init_weights)

        self.nr_cate = nr_cate
        self.reg_n_D = 4
        # Note: for a quaternion q=(a,b,c,d), we always ensure a>0, that this cos(theta/2)>0 --> theta in [0,pi]
        # Thus only b,c,d need sign prediction.
        dim_need_sign = 3
        _signs = list( product(*( [(-1,1)]*dim_need_sign )) ) # [(-1, -1, -1), (-1, -1, 1), ..., (1, 1, 1)], with len=8
        self.signs = [(1,)+x for x in _signs]   # [(1, -1, -1, -1), (1, -1, -1, 1), ..., (1, 1, 1, 1)], with len=8
        self.signs2label = odict(zip(self.signs, range(len(self.signs))))
        self.label2signs = Variable( torch.FloatTensor(self.signs) ).cuda()  # make it as a Variable

        #-- Head architecture
        # Note: for quaternion, there's only one regression head (instead of 3 (for a,e,t)).
        #       Thus, nr_fc8=996  (see design.py)
        self.head_sqrdprob_quat = self.head_seq(self.top_size,   self.reg_n_D, nr_cate=nr_cate, nr_fc8=996, init_weights=init_weights)
        # each of 3 quaternion complex component can be + or -, that totally 2**3 possible sign categories.
        self.head_signcate_quat = self.head_seq(self.top_size, len(self.signs), nr_cate=nr_cate, nr_fc8=996, init_weights=init_weights)

        # for abs branch
        self.maskout = Maskout(nr_cate=nr_cate)
        self.softmax = nn.Softmax(dim=1).cuda()
        # for sgc branch
        self.maskout_sgc = Maskout(nr_cate=nr_cate)  # make a new layer to maskout sign classification only.

        # loss module
        self.loss_handler_abs_quat = Cos_Proximity_Loss_Handler()  # Neg_Dot_Loss_Handler()  # Cos_Proximity_Loss_Handler() #
        self.loss_handler_sgc_quat = Cross_Entropy_Loss_Handler()

        self.targets    = ['abs_quat','sgc_quat']
        self.gt_targets = ['quat']


    def forward(self, x, label):
        """label shape (batchsize, )  """
        x = self.trunk(x)           # Forward Conv and Fc6,Fc7
        #
        batchsize = x.size(0)

        # Note: squared probability
        x_sqr_quat = self.maskout(self.head_sqrdprob_quat(x).view(batchsize, self.nr_cate, self.reg_n_D   ), label)  # ========>>>>> Maskout output  (B,4) hook gradient.
        #-- Exp and Normalize coordinate to a unit
        x_sqr_quat = self.softmax(x_sqr_quat) #, nr_cate=self.nr_cate)

        # sign category head (totally 2^4=16 category)
        x_sgc_quat = self.maskout_sgc(self.head_signcate_quat(x).view(batchsize, self.nr_cate, len(self.signs)), label)

        Prob = edict(abs_quat=torch.sqrt(x_sqr_quat), sgc_quat=x_sgc_quat)
        return Prob

    def compute_loss(self, Prob, GT):
        # First get sign label from GT
        #== Formulate absolute value of quaternion
        GT_abs_quat = torch.abs(GT.quat)
        #== Formulate signs label of quaternion
        GT_sign_quat = torch.sign(GT.quat)
        GT_sign_quat[GT_sign_quat==0] = 1 # make sign of '0' as 1
        signs_tuples = [tuple(x) for x in GT_sign_quat.data.cpu().numpy().astype(np.int32).tolist()]
        for signs_tuple in signs_tuples:  # q and -q gives the same rotation.
            assert signs_tuple[0]>0, "Need GT to be all positive on first dim of quaternion: %s" % GT  # assert all quaternion first dim is positive.
        # signs label
        GT_sgc_quat = Variable( torch.LongTensor([self.signs2label[signs_tuple] for signs_tuple in signs_tuples]) )
        if GT.quat.is_cuda:
            GT_sgc_quat = GT_sgc_quat.cuda()

        # here just because compute_loss need a same key from Prob and GT,
        # so we just give a fake name to GT.sqr_quat as '_GT.logsqr_quat'.
        _GT = edict(abs_quat=GT_abs_quat, sgc_quat=GT_sgc_quat)
        Loss_abs_quat = self.loss_handler_abs_quat.compute_loss(['abs_quat'], Prob, _GT)
        Loss_sgc_quat = self.loss_handler_sgc_quat.compute_loss(['sgc_quat'], Prob, _GT)

        # To add loss weights here.
        Loss = edict( abs_quat=Loss_abs_quat['abs_quat']*10,  #  / 5.
                      sgc_quat=Loss_sgc_quat['sgc_quat'],)
        return Loss

    def compute_pred(self, Prob):
        x_abs_quat = Prob['abs_quat']  # torch.sqrt(torch.exp(Prob['logsqr_quat']))
        x_sgc_quat = Prob['sgc_quat']
        batchsize = x_abs_quat.size(0)
        #
        sign_ind  = cls_pred(x_sgc_quat, topk=(1,)).data.view(-1,)
        item_inds = torch.from_numpy(np.arange(batchsize)).cuda()
        _label_shape = self.label2signs.size()
        x_sign_quat = self.label2signs.expand(batchsize,*_label_shape)[item_inds, sign_ind]

        x_quat = x_abs_quat * x_sign_quat

        # Get cpu data.
        batch_quat = x_quat.data.cpu().numpy().copy()
        batchsize = x_quat.size(0)
        assert batch_quat.shape==(batchsize, 4), batch_quat.shape
        #
        Pred = edict(quat=batch_quat)
        return Pred
#---------------------------------------------------------------------




#----------------------------------------------------------------------------------

if __name__ == '__main__':
    model = reg2D_Net().copy_weights()

    # import numpy as np
    dummy_batch_data  = np.zeros((2,3,227,227),dtype=np.float32)
    dummy_batch_label = np.zeros((2,1),        dtype=np.int64)
    dummy_batch_data  = torch.autograd.Variable( torch.from_numpy(dummy_batch_data ) )
    dummy_batch_label = torch.autograd.Variable( torch.from_numpy(dummy_batch_label) )

    Pred = model(dummy_batch_data, dummy_batch_label)
    # print (Prob.a)

