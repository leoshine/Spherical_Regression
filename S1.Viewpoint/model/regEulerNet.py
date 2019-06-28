# coding: utf8
"""
 @Author  : Shuai Liao
"""

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import torch
from basic.common import rdict
import numpy as np
from easydict import EasyDict as edict
from collections import OrderedDict as odict
from itertools import product

from pytorch_util.netutil.common_v2.trunk_alexnet_bvlc import AlexNet_Trunk
from pytorch_util.netutil.common_v2.trunk_vgg          import VGG16_Trunk
from pytorch_util.netutil.common_v2.trunk_resnet       import ResNet101_Trunk, ResNet50_Trunk, ResNet152_Trunk

net_arch2Trunk = dict(
    alexnet  = AlexNet_Trunk,
    vgg16    = VGG16_Trunk,
    resnet50 = ResNet50_Trunk,
    resnet101= ResNet101_Trunk,
    resnet152= ResNet152_Trunk,
)

from pytorch_util.libtrain import copy_weights, init_weights_by_filling
from pytorch_util.torch_v4_feature import LocalResponseNorm # *
from pytorch_util.torch_3rd_layers import Maskout
from pytorch_util.torch_3rd_funcs  import norm2unit, exp_Normalization


def reg2d_pred2tgt(pr_sin, pr_cos):
    theta = torch.atan2(pr_sin, pr_cos)
    return theta

def reg3d_pred2tgt(cos_1, cos_2, cos_3):
    Y = cos_1*np.cos(np.pi/3) + cos_2*np.cos(0) + cos_3*np.cos(-np.pi/3)
    X = cos_1*np.sin(np.pi/3) + cos_2*np.sin(0) + cos_3*np.sin(-np.pi/3)
    theta = torch.atan2(X, Y) #(Y, X)  Why?   X,Y  not Y,X
    return theta

def cls_pred(output, topk=(1,)):
    maxk = max(topk)
    batch_size = output.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    return pred


loss_balance = 4. # 1.

class _naiReg_Net(nn.Module): # (_View_Net_Base): #

    @staticmethod
    def head_seq(in_size, reg_n_D, nr_cate=12, nr_fc8=334):      # in_size=4096
        seq = nn.Sequential(
                nn.Linear(in_size, nr_fc8),                             # Fc8_a
                nn.ReLU(inplace=True),
                #nn.Dropout(),
                nn.Linear(nr_fc8, nr_cate*reg_n_D),                     # Prob_a
             )
        init_weights_by_filling(seq, gaussian_std=0.005)  # fill weight with gaussian filler
        return seq

    """BVLC alexnet architecture (Note: slightly different from pytorch implementation.)"""
    def __init__(self, nr_cate=12, net_arch='alexnet', pretrained=True):  # AlexNet_Trunk
        super(_naiReg_Net, self).__init__()
        _Trunk = net_arch2Trunk[net_arch]
        self.trunk = _Trunk(pretrained=pretrained)

        self.nr_cate = nr_cate
        self.top_size = 4096 if not self.trunk.net_arch.startswith('resnet') else 2048

    def forword(self, x, label):
        raise NotImplementedError



#---------------------------------------------------------------------[reg2D]
class reg_Euler2D_flat_Net(_naiReg_Net):

    """BVLC alexnet architecture (Note: slightly different from pytorch implementation.)"""
    def __init__(self, nr_cate=12, net_arch='alexnet', pretrained=True):
        _naiReg_Net.__init__(self, nr_cate=nr_cate, net_arch=net_arch, pretrained=pretrained)
        # super(reg2D_Net, self).__init__()

        self.nr_cate = nr_cate
        self.reg_n_D = 2

        #-- Head architecture
        self.head_a  = self.head_seq(self.top_size, self.reg_n_D, nr_cate=nr_cate)
        self.head_e  = self.head_seq(self.top_size, self.reg_n_D, nr_cate=nr_cate)
        self.head_t  = self.head_seq(self.top_size, self.reg_n_D, nr_cate=nr_cate)

        # for maskout a,e,t
        self.maskout = Maskout(nr_cate=nr_cate)

        # loss module
        self.loss_handler = Smooth_L1_Loss_Handler()
        self.targets= ['cos_a','sin_a',  'cos_e','sin_e',  'cos_t','sin_t']

    def forward(self, x, label):
        """label shape (batchsize, )  """
        x = self.trunk(x)       # Forward Conv and Fc6,Fc7
        #
        batchsize = x.size(0)   # .split(1, dim=1)

        # maskout prediction for this category
        x_a = self.maskout(self.head_a(x).view(batchsize, self.nr_cate, self.reg_n_D) , label)
        x_e = self.maskout(self.head_e(x).view(batchsize, self.nr_cate, self.reg_n_D) , label)
        x_t = self.maskout(self.head_t(x).view(batchsize, self.nr_cate, self.reg_n_D) , label)
        #-- split out [cosθ, sinθ] prediction
        x_cos_a, x_sin_a = x_a[:,0], x_a[:,1]
        x_cos_e, x_sin_e = x_e[:,0], x_e[:,1]
        x_cos_t, x_sin_t = x_t[:,0], x_t[:,1]


        Prob = edict(cos_a=x_cos_a, sin_a=x_sin_a,
                     cos_e=x_cos_e, sin_e=x_sin_e,
                     cos_t=x_cos_t, sin_t=x_sin_t)
        return Prob

    def compute_loss(self, Prob, GT):
        Loss = self.loss_handler.compute_loss(self.targets, Prob, GT)
        return Loss

    def compute_pred(self, Prob):
        Pred = edict( a=reg2d_pred2tgt(Prob.sin_a, Prob.cos_a),
                      e=reg2d_pred2tgt(Prob.sin_e, Prob.cos_e),
                      t=reg2d_pred2tgt(Prob.sin_t, Prob.cos_t), )
        return Pred
#---------------------------------------------------------------------





#---------------------------------------------------------------------
class reg_Euler2D_Sexp_Net(_naiReg_Net):

    @staticmethod
    def head_seq(in_size, nr_cate=12, nr_fc8=1024):
        seq_fc8 = nn.Sequential(
                nn.Linear(in_size, nr_fc8),                        # Fc8_a
                nn.ReLU(inplace=True),
                nn.Dropout(),
             )
        seq_ccss= nn.Linear(nr_fc8, nr_cate*2)                     # Prob_a
        seq_sgnc= nn.Linear(nr_fc8, nr_cate*4)                     # Prob_a
        #
        init_weights_by_filling(seq_fc8 , gaussian_std=0.005)  # fill weight with gaussian filler
        init_weights_by_filling(seq_ccss, gaussian_std=0.005)  # fill weight with gaussian filler
        init_weights_by_filling(seq_sgnc, gaussian_std=0.005)  # fill weight with gaussian filler
        return seq_fc8, seq_ccss, seq_sgnc


    """BVLC alexnet architecture (Note: slightly different from pytorch implementation.)"""
    def __init__(self, nr_cate=12, net_arch='alexnet', pretrained=True, nr_fc8=1024):
        _naiReg_Net.__init__(self, nr_cate=nr_cate, net_arch=net_arch, pretrained=pretrained)
        # super(regS1xy_Net, self).__init__()

        self.nr_cate = nr_cate

        #-- Head architecture
        # 2 categories for each {a,e,t}:    (log(x^2), log(y^2))
        self.head_fc8_a, self.head_xx_yy_a, self.head_sign_x_y_a = self.head_seq(self.top_size, nr_cate=nr_cate, nr_fc8=nr_fc8)
        self.head_fc8_e, self.head_xx_yy_e, self.head_sign_x_y_e = self.head_seq(self.top_size, nr_cate=nr_cate, nr_fc8=nr_fc8)
        self.head_fc8_t, self.head_xx_yy_t, self.head_sign_x_y_t = self.head_seq(self.top_size, nr_cate=nr_cate, nr_fc8=nr_fc8)

        # 4 category for each {a,e,t}
        # Given (x->cosθ, y->sinθ)
        #    { quadrant_label : (sign(x),sign(y)) } = { 0:++, 1:-+, 2:--, 3:+- }
        #
        # 1      |     0
        #   -,+  | +,+
        # -------|-------
        #   -,-  | +,-
        # 2      |     3

        # for maskout a,e,t
        self.maskout = Maskout(nr_cate=nr_cate)
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

        # loss module
        self.loss_handler_ccss = Neg_Dot_Loss_Handler()
        self.loss_handler_sign = Cross_Entropy_Loss_Handler()

        self.targets = ['ccss_a','ccss_e','ccss_t',  'sign_a','sign_e','sign_t'] # ['a','e','t']


    def forward(self, x, label):
        """label shape (batchsize, )  """
        x = self.trunk(x)       # Forward Conv and Fc6,Fc7
        #
        batchsize = x.size(0)   # .split(1, dim=1)

        # shared fc8 for each a,e,t
        x_a = self.head_fc8_a(x)
        x_e = self.head_fc8_e(x)
        x_t = self.head_fc8_t(x)

        log_xx_yy_a = self.maskout(self.head_xx_yy_a(x_a).view(batchsize, self.nr_cate, 2), label)
        log_xx_yy_e = self.maskout(self.head_xx_yy_e(x_e).view(batchsize, self.nr_cate, 2), label)
        log_xx_yy_t = self.maskout(self.head_xx_yy_t(x_t).view(batchsize, self.nr_cate, 2), label)
        #
        logprob_xx_yy_a = self.logsoftmax(log_xx_yy_a) #
        logprob_xx_yy_e = self.logsoftmax(log_xx_yy_e) #
        logprob_xx_yy_t = self.logsoftmax(log_xx_yy_t) #

        sign_x_y_a = self.maskout(self.head_sign_x_y_a(x_a).view(batchsize, self.nr_cate, 4), label)
        sign_x_y_e = self.maskout(self.head_sign_x_y_e(x_e).view(batchsize, self.nr_cate, 4), label)
        sign_x_y_t = self.maskout(self.head_sign_x_y_t(x_t).view(batchsize, self.nr_cate, 4), label)

        Prob = edict( # log probability of xx, yy   (xx+yy=1 or x^2+y^2=1)
                      logprob_ccss=edict(a=logprob_xx_yy_a,
                                         e=logprob_xx_yy_e,
                                         t=logprob_xx_yy_t,),
                      # sign of x, y.
                      sign_cate_cs=edict(a=sign_x_y_a,
                                         e=sign_x_y_e,
                                         t=sign_x_y_t,)
                    )
        return Prob

    def compute_loss(self, Prob, GT):
        Loss_ccss = self.loss_handler_ccss.compute_loss(['a','e','t'], Prob.logprob_ccss, dict(a=GT.ccss_a,e=GT.ccss_e,t=GT.ccss_t))
        Loss_sign = self.loss_handler_sign.compute_loss(['a','e','t'], Prob.sign_cate_cs, dict(a=GT.sign_a,e=GT.sign_e,t=GT.sign_t))
        # To add loss weights here.
        Loss = edict( ccss_a=Loss_ccss.a * loss_balance,
                      ccss_e=Loss_ccss.e * loss_balance,
                      ccss_t=Loss_ccss.t * loss_balance,
                      #
                      sign_a=Loss_sign.a,
                      sign_e=Loss_sign.e,
                      sign_t=Loss_sign.t,)
        return Loss

    def compute_pred(self, Prob):
        label2signs = torch.FloatTensor([[ 1, 1],
                                         [-1, 1],
                                         [-1,-1],
                                         [ 1,-1]])
        label2signs = Variable(label2signs).cuda()
        batchsize = Prob.logprob_ccss['a'].size(0)

        Pred = edict()
        for tgt in Prob.logprob_ccss.keys():
            log_xx_yy = Prob.logprob_ccss[tgt]
            abs_cos_sin  = torch.sqrt(torch.exp(log_xx_yy))
            sign_ind = cls_pred(Prob.sign_cate_cs[tgt], topk=(1,)).data.view(-1,)
            item_inds = torch.from_numpy(np.arange(batchsize)).cuda()
            sign_cos_sin = label2signs.expand(batchsize,4,2)[item_inds, sign_ind]  # label2signs
            cos_sin = abs_cos_sin*sign_cos_sin
            Pred[tgt] = torch.atan2(cos_sin[:,1], cos_sin[:,0]) #
        return Pred
#---------------------------------------------------------------------





class Cross_Entropy_Loss_Handler:
    def __init__(self):
        self.cross_entropy_loss = nn.CrossEntropyLoss().cuda()

    # interface function
    def compute_loss(self, tgts, Pred, GT):
        """ tgts: list of target names  e.g. tgts=['a', 'e', 't']
            GT  : dict of ground truth for each target
            Pred: dict of prediction   for each target
        """
        Loss = edict()
        for tgt in tgts:
            Loss[tgt] = self.cross_entropy_loss(Pred[tgt], GT[tgt])
        return Loss


class Neg_Dot_Loss_Handler:
    def __init_(self):
        pass

    def compute_loss(self, tgts, Pred, GT):
        Loss = edict()
        for tgt in tgts:
            Loss[tgt] = torch.mean( -torch.sum(GT[tgt]*Pred[tgt], dim=1))
        return Loss


class Cos_Proximity_Loss_Handler:
    def __init__(self):
        self.cos_sim = nn.CosineSimilarity().cuda()

    # interface function
    def compute_loss(self, tgts, Pred, GT):
        """ tgts: list of target names. In this case   has to be tgts=['quat']
            GT  : dict of ground truth for each target
            Pred: dict of prediction   for each target
        """
        # assert tgts==['quat'], tgts
        Loss = edict()
        for tgt in tgts:
            Loss[tgt] = torch.mean( 1 - self.cos_sim(Pred[tgt], GT[tgt]) )  # use 1-cos(theta) to make loss as positive.
        return Loss



class Smooth_L1_Loss_Handler:
    def __init__(self):
        self.smooth_l1_loss = nn.SmoothL1Loss().cuda()

    def compute_loss(self, tgts, Pred, GT):
        """ tgts: list of target names  e.g. tgts=['a', 'e', 't']
            GT  : dict of ground truth for each target
            Pred: dict of prediction   for each target
        """
        Loss = edict()
        for tgt in tgts:
            Loss[tgt] = self.smooth_l1_loss(Pred[tgt], GT[tgt])  # [warning] pred first, gt second
        return Loss



#----------------------------------------------------------------------------------

if __name__ == '__main__':
    model = reg_Euler2D_Sexp_Net()

    # import numpy as np
    dummy_batch_data  = np.zeros((2,3,227,227),dtype=np.float32)
    dummy_batch_label = np.zeros((2,1),        dtype=np.int64)
    dummy_batch_data  = torch.autograd.Variable( torch.from_numpy(dummy_batch_data ) )
    dummy_batch_label = torch.autograd.Variable( torch.from_numpy(dummy_batch_label) )

    Pred = model(dummy_batch_data, dummy_batch_label)

