## How to make your first S_exp regression?

```python
from itertools import product


# Assume you have:
# [1] Ground truth spherical targets:                 GT       (batch_size x target_dim)
# [2] Network output for absolute value problem:      prob_abs (batch_size x target_dim)
# [3] Network output for sign classification problem: prob_sgc (batch_size x 2^target_dim)

#-------------------------------------------------------------------------------#
#---- Build up the signs combination to the classification label mapping. ------#
#-------------------------------------------------------------------------------#
# Here we show the example for quaternion q = (a,b,c,d). 
# Since we force 'a' to be positive, it results in only b,c,d need sign prediction.
dim_need_sign = 3
# Sign combination for (b,c,d)
_signs = list( product(*( [(-1,1)]*dim_need_sign )) ) # [(-1, -1, -1), (-1, -1, 1), ..., (1, 1, 1)], with len=8
# Sign combination for (a,b,c,d)
signs = [(1,)+x for x in _signs]                      # [(1, -1, -1, -1), (1, -1, -1, 1), ..., (1, 1, 1, 1)], with len=8
signs2label = odict(zip(self.signs, range(len(self.signs))))
label2signs = torch.FloatTensor(self.signs)  # make it as a Variable



#-------------------------------------------------------------------------------#
#------------------ Formulate absolute value of quaternion ---------------------#
#-------------------------------------------------------------------------------#
GT_abs = torch.abs(GT)

#-------------------------------------------------------------------------------#
#------------------- Formulate signs label of quaternion -----------------------#
#-------------------------------------------------------------------------------#
GT_sign = torch.sign(GT)
GT_sign[GT_sign==0] = 1  # make sign of '0' as 1
signs_tuples = [tuple(x) for x in GT_sign.data.cpu().numpy().astype(np.int32).tolist()]
# signs label for classification
GT_sgc  = torch.LongTensor([signs2label[signs_tuple] for signs_tuple in signs_tuples]) 


#-------------------------------------------------------------------------------#
#------------------------- Compute losses here     -----------------------------#
#-------------------------------------------------------------------------------#
loss_abs = cross_entropy_loss(prob_abs, GT_abs)
loss_sgc = neg_dot_prod_loss (prob_sgc, GT_sgc)  


#-------------------------------------------------------------------------------#
#-------------------- Make prediction at inference -----------------------------#
#-------------------------------------------------------------------------------#
x_abs     = prob_abs
batchsize = x_abs.size(0)
#
_, sign_ind  = prob_sgc.topk(maxk, 1, True, True)    # take argmax
item_inds    = torch.range(batchsize)
_label_shape = self.label2signs.size()
x_sign = self.label2signs.expand(batchsize,*_label_shape)[item_inds, sign_ind]

# combine absolute values with the signs to make final prediction.
pred_quat = x_abs * x_sign

```


<br/>
<br/>
<br/>

## Pytorch network module design

We make every network module to be easily extentable:


### For single target (or a.k.a class-agnostic)

```python
# coding: utf8
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch.autograd import Variable
from basic.common import rdict, add_path
import numpy as np
from easydict import EasyDict as edict

from pytorch_util.libtrain import copy_weights, init_weights_by_filling


net_arch2Trunk = dict(
    # alexnet    = AlexNet_Trunk,
    # vgg16      = VGG16_Trunk,
    # resnet18   = (ResNet18_Trunk, 512),
    # resnet101= ResNet101_Trunk,
    # resnet50 = ResNet50_Trunk,
)


class Base_Net(nn.Module):
    
    @staticmethod
    def head_seq(in_dim, out_dim, init_weights=True):
        seq = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                # nn.ReLU(inplace=True),
                # #nn.Dropout(),
                # nn.Linear(nr_fc8, out_dim),
             )
        if init_weights:
            init_weights_by_filling(seq, gaussian_std=0.005, kaiming_normal=True)  # fill weight with gaussian filler
        return seq

    def __init__(self, target_dim, net_arch='alexnet', init_weights=True):
        super(Base_Net, self).__init__()
        self.target_dim = target_dim

        # Get definition of trunk class/function, and it's expected output dim.
        _Trunk, _top_dim = net_arch2Trunk[net_arch]

        # define trunk module
        self.trunk = _Trunk(init_weights=init_weights)

        # define head module
        self.head = self.head_seq(_top_dim, self.target_dim, init_weights=init_weights)

        # define loss handler
        # self.loss_handler = _MAKE_LOSS_HANDLER()_

        # The rest part of your code ...
        # ....

    def forward(self, x):
        """label shape (batchsize, )  """
        # Forward trunk sequence
        x = self.trunk(x)
        batchsize = x.size(0)

        # Forward head sequence and compute problem representation
        # (Note: prob can be different from the expected final prediction).
        prob = self.head(x).view(batchsize, self.target_dim)
         
        return prob

    def compute_loss(self, prob, gt):
        """both Prob and GT are easy_dict."""
        # Loss = self.loss_handler.compute_loss(Prob, GT)
        loss = None
        raise NotImplementedError

        return loss

    def compute_pred(self, prob):

        # Implement your own mapping: prob --> pred
        pred = prob.cpu().numpy().copy()  # return numpy data.
        raise NotImplementedError

        return pred
```



### For multi-class and multi-target network

```python
class Multi_Class_Reg_Net(nn.Module):
    
    @staticmethod
    def head_seq(in_size, reg_n_D, nr_cate=12, nr_fc8=334, init_weights=True):  # in_size=4096
        seq = nn.Sequential(
                nn.Linear(in_size, nr_fc8),                             # Fc8_a
                nn.ReLU(inplace=True),
                #nn.Dropout(),
                nn.Linear(nr_fc8, nr_cate*reg_n_D),                     # Prob_a
             )
        if init_weights:
            init_weights_by_filling(seq, gaussian_std=0.005, kaiming_normal=True)  # fill weight with gaussian filler
        return seq

    def __init__(self, nr_cate, nr_target, 
                 net_arch='alexnet', init_weights=True):  
        super(Multi_Class_Reg_Net, self).__init__()
        self.nr_cate   = nr_cate
        self.nr_target = nr_target
        
        # Get definition of trunk class/function, and it's expected output dim.
        _Trunk, top_dim = net_arch2Trunk[net_arch]
        
        # define trunk module
        self.trunk = _Trunk(init_weights=init_weights)
        
        # define head module
        self.head = self.head_seq(top_dim, self.reg_n_D, nr_cate=nr_cate, nr_fc8=996, init_weights=init_weights)

        # define maskout layer for making prediction for specific class
        self.maskout = Maskout(nr_cate=nr_cate)
                
        # define loss handler
        self.loss_handler = _MAKE_LOSS_HANDLER()_
        
        # list out the names of output targets 
        # (keys of dict return by forward)
        self.targets = ['out']
        self.gt_targets = ['age']
        
        # The rest part of your code ...
        # ....


    def forward(self, x, cls_inds):
        """label shape (batchsize, )  """
        # Forward trunk sequence
        x = self.trunk(x)           
        batchsize = x.size(0)      

        # Forward head sequence
        x = self.head(x).view(batchsize, self.nr_cate, self.reg_n_D)
        
        # Forward maskout but class indice of each sample
        x = self.maskout(x, cls_inds)
        
        Prob = edict(out=x)
        return Prob

    def compute_loss(self, Prob, GT):
        """both Prob and GT are easy_dict."""
        Loss = self.loss_handler.compute_loss(self.targets, Prob, GT)
        return Loss

    def compute_pred(self, Prob):
        raw_out = Prob['out']
        
        # Get cpu data.
        raw_out = raw_out.cpu().numpy().copy()
        
        Pred = edict(out= raw_out)
        return Pred
```