# coding: utf8
"""
 @Author  : Shuai Liao
"""

import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from   torch.autograd import Variable
import torch
from   basic.common import rdict
import numpy as np
from   easydict import EasyDict as edict
from   collections import OrderedDict as odict
from   itertools import product

from basic.common import add_path, env
this_dir = os.path.dirname(os.path.abspath(__file__))
add_path(this_dir+'/../lib/')
from helper import *
from model   import VGG16_Trunk
from modelSE import VGG16_Trunk as VGG16SE_Trunk

# net_arch2Trunk = dict(
#     vgg16    = VGG16_Trunk,
#     vgg16se  = VGG16SE_Trunk,
# )
net_arch2Trunk = dict(
    vgg16=dict(
        Sflat  = VGG16_Trunk,
        Sexp   = VGG16SE_Trunk,
    ),
)

from pytorch_util.libtrain import copy_weights, init_weights_by_filling
from pytorch_util.torch_v4_feature import LocalResponseNorm # *
from pytorch_util.torch_3rd_layers import Maskout
from pytorch_util.torch_3rd_funcs  import norm2unit, exp_Normalization



def cls_pred(output, topk=(1,), dim=1):
    maxk = max(topk)
    batch_size = output.size(0)

    _, pred = output.topk(maxk, dim=dim, largest=True, sorted=True)
    return pred


class _regNormalNet(nn.Module):

    def __init__(self, method, net_arch='vgg16', init_weights=True):
        super(_regNormalNet, self).__init__()
        _Trunk = net_arch2Trunk[net_arch][method]
        self.trunk = _Trunk(init_weights=init_weights)

    def forword(self, x, label):
        raise NotImplementedError



#---------------------------------------------------------------------[regQuat]
class reg_Sflat_Net(_regNormalNet):

    def __init__(self, net_arch='vgg16', init_weights=True):
        _regNormalNet.__init__(self, 'Sflat', net_arch=net_arch, init_weights=init_weights)

        # loss module
        self.loss_handler  = Cos_Proximity_Loss_Handler()
        self.targets = ['norm']

    def forward(self, x):
        """label shape (batchsize, )  """
        x = self.trunk(x)           # Forward Conv and Fc6,Fc7
        #
        batchsize = x.size(0)       # x of shape (40, 3, 240, 320)

        #-- Normalize coordinate to a unit
        x_norm = norm2unit(x, dim=1)

        Prob = edict(norm=x_norm.permute(0,2,3,1).double()) # transpose prediction from BxCxHxW to BxHxWxC order.
        return Prob

    def compute_loss(self, Prob, GT):
        Loss, Errs = self.loss_handler.compute_loss(self.targets, Prob, GT)
        _metric_ = edict(norm=Errs['norm'])
        return Loss, _metric_

    def compute_pred(self, Prob, encode_bit=8):
        x_norm = Prob['norm']
        # Get cpu data.
        norm = x_norm.data.cpu().numpy().copy()  # B,H,W,C
        assert encode_bit in [8,16]
        if encode_bit==8:
            normImgs = ((norm+1)*(2**7)).astype(np.uint8)   # map [-1,1]  to [0,256)
        else:
            normImgs = ((norm+1)*(2**15)).astype(np.uint16) # map [-1,1]  to [0,65535)
        Pred = edict(norm=normImgs)
        return Pred



#---------------------------------------------------------------------[regQuat]
class reg_Sexp_Net(_regNormalNet):  # Spherical exponential Problem + sign classification

    def __init__(self, net_arch='vgg16', init_weights=True):
        _regNormalNet.__init__(self,  'Sexp', net_arch=net_arch, init_weights=init_weights)

        self.reg_n_D = 3
        # Note: for a surface normal (x,z,y)   (Watch out the order)
        #       z should always satisfy z<=0   (Surface normal should from visible surfaces)
        #       Thus only x,y need sign prediction.
        dim_need_sign = 2
        _signs = list( product(*( [(-1,1)]*dim_need_sign )) ) # [(-1, -1), (-1, 1), (1, -1), (1, 1)], with len=4
        self.signs = [(x[0],-1,x[1]) for x in _signs]         # y-z-x order: [(-1, -1, -1), (-1, -1, 1), (1, -1, -1), (1, -1, 1)], with len=4; z always -1
        self.signs2label = odict(zip(self.signs, range(len(self.signs))))
        self.label2signs = Variable( torch.DoubleTensor(self.signs) ).cuda()  # make it as a Variable

        self.softmax = nn.Softmax(dim=1).cuda()

        # loss module
        self.loss_handler_abs_norm = Cos_Proximity_Loss_Handler()
        self.loss_handler_sgc_norm = Cross_Entropy_Loss_Handler()

        self.targets    = ['sgc_norm','abs_norm']
        self.gt_targets = ['norm']

        self.cost, self.sint = torch.tensor(np.cos(np.pi/4)).double().cuda(), torch.tensor(np.sin(np.pi/4)).double().cuda()


    def forward(self, x):
        """label shape (batchsize, )  """
        x_abs, x_sgc = self.trunk(x)       # Forward Conv and Fc6,Fc7
        #
        batchsize = x_abs.size(0)

        #-- Exp and Normalize coordinate to a unit
        x_sqr_norm = self.softmax(x_abs) #, nr_cate=self.nr_cate)

        # sign category head (totally 4 category)
        x_sgc_norm = x_sgc

        Prob = edict(abs_norm=torch.sqrt(x_sqr_norm).permute(0,2,3,1).double(), # B,H,W,3
                     sgc_norm=x_sgc_norm.permute(0,2,3,1)   )                   # B,H,W,4
        return Prob

    def compute_loss(self, Prob, GT):
        B,H,W,_3_ = GT.norm.size()
        assert _3_==3, "Wrong dim: %s,%s,%s,%s" % (B,H,W,_3_)
        # First get sign label from GT
        #== Formulate squared value of quaternion
        GT_abs_norm = torch.abs(GT.norm)      # B,H,W,3
        #== Formulate signs label of quaternion
        GT_sign_norm = torch.sign(GT.norm)    # B,H,W,3
        #-------------------------------------
        # hard coded:   sign to label
        #-------------------------------------
        #   y   x       label
        # [-1  -1]  -->  0
        # [-1   1]  -->  1
        # [ 1  -1]  -->  2
        # [ 1   1]  -->  3
        # GT_sign_norm  (B,H,W,3)  in y-z-x order
        GT_sign_norm[GT_sign_norm==0] = -1 # make sign of '0' as -1 (use -1 instead of 1 just because z<=0)
        y_sign, x_sign = GT_sign_norm[:,:,:,0], GT_sign_norm[:,:,:,2]
        y_sign += 1 # [y_sign==-1]
        x_sign[x_sign==-1] = 0
        GT_sgc_norm = (y_sign+x_sign).long()  # data with shape with (B,H,W) index of [0,1,2,3]

        # here just because compute_loss need a same key from Prob and GT,
        # so we just give a fake name to GT.sqr_quat as '_GT.logsqr_norm'.
        _GT = edict(abs_norm=GT_abs_norm, sgc_norm=GT_sgc_norm, mask=GT.mask)  # abs_norm: (B,H,W,3)  sgc_norm: (B,H,W)
        Loss_abs_norm, abs_Errs = self.loss_handler_abs_norm.compute_loss(['abs_norm'], Prob, _GT)
        Loss_sgc_norm = self.loss_handler_sgc_norm.compute_loss(['sgc_norm'], Prob, _GT)

        # ----------------------------------------
        # Compute the metric.
        sign_ind  = cls_pred(Prob['sgc_norm'], topk=(1,), dim=3).data.squeeze(dim=3)  # B,H,W
        pr_sign_norm = self.label2signs[sign_ind]      # magic here: Indexing label2signs (4x3) by sign_ind (B,H,W) becomes (B,H,W,3)  (10, 240, 320, 3)
        pr_abs_norm  = Prob['abs_norm']
        _Prob = edict(norm=pr_abs_norm * pr_sign_norm) # current predicted final norm (applied sign prediction)
        _Loss_norm, out_Errs = self.loss_handler_abs_norm.compute_loss(['norm'], _Prob, GT) # just borrow loss_handler_abs_norm, nothing more.

        # Compute acc of classification:  sign_ind  vs    GT_sgc_norm
        mask = GT['mask']
        acc = eval_cls(sign_ind[mask], GT_sgc_norm[mask])
        _metric_ = edict(abs_norm     = abs_Errs['abs_norm'],
                         norm         = out_Errs['norm']    ,
                         sgc_norm_acc = acc                 ,)

        # To add loss weights here.
        Loss = edict( abs_norm=Loss_abs_norm['abs_norm']*10,  #  / 5.
                      sgc_norm=Loss_sgc_norm['sgc_norm'], )
        return Loss, _metric_ # .update(abs_Errs)

    def compute_pred(self, Prob, encode_bit=8):
        x_abs_norm = Prob['abs_norm']  # B,H,W,3
        x_sgc_norm = Prob['sgc_norm']  # B,H,W,4
        batchsize  = x_abs_norm.size(0)
        #
        sign_ind    = cls_pred(x_sgc_norm, topk=(1,), dim=3).data.squeeze(dim=3) # .view(-1,)  # B,H,W
        x_sign_norm = self.label2signs[sign_ind] # magic here: Indexing label2signs (4x3) by sign_ind (B,H,W) becomes (B,H,W,3)
        #
        x_norm = x_abs_norm * x_sign_norm  # B,H,W,3
        # --------------Recover rot45 trick --------------
        # Note: since we applied rot45 trick, here we recover it back
        _x_norm = x_norm.detach().clone() # return a copy of x_norm without grad
        _y,_z,_x = _x_norm[:,:,:,0],_x_norm[:,:,:,1],_x_norm[:,:,:,2]
        y, z, x = x_norm[:,:,:,0],x_norm[:,:,:,1],x_norm[:,:,:,2]
        x[:] = self.cost*_x - self.sint*_y
        y[:] = self.sint*_x + self.cost*_y
        # ------------------------------------------------
        # Get cpu data.
        norm = x_norm.data.cpu().numpy().copy()  # B,H,W,C
        assert encode_bit in [8,16]
        if encode_bit==8:
            normImgs = ((norm+1)*(2**7)).astype(np.uint8)   # map [-1,1]  to [0,256)
        else:
            normImgs = ((norm+1)*(2**15)).astype(np.uint16) # map [-1,1]  to [0,65535)
        Pred = edict(norm=normImgs)
        return Pred
