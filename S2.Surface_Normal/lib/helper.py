# coding: utf8
"""
 @Author  : Shuai Liao
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
from easydict import EasyDict as edict
from collections import OrderedDict as odict
from itertools import product


def eval_cls(Preds, GTs):
    acc = torch.mean((Preds==GTs).float())
    return acc.item()

class Cross_Entropy_Loss_Handler:
    def __init__(self):
        self.cross_entropy_loss = nn.CrossEntropyLoss().cuda()

    # interface function
    def compute_loss(self, tgts, Pred, GT):
        """ tgts: list of target names
            GT  : dict of ground truth for each target  BxHxW
            Pred: dict of prediction   for each target  BxHxWx4
        """
        mask  = GT['mask']
        Loss = edict()
        for tgt in tgts:
            gt = GT[tgt][mask].view(-1)               # as (BxK,)
            pr = Pred[tgt][mask].view(gt.size(0),-1)  # Pred[tgt][mask]  (BxK, 4)
            Loss[tgt] = self.cross_entropy_loss(pr, gt).double()
        return Loss


class Neg_Dot_Loss_Handler:
    def __init_(self):
        pass

    def compute_loss(self, tgts, Pred, GT):
        Loss = edict()
        for tgt in tgts:
            """ Bug fixed on 22 Aug 2018
                torch.dot can only be applied to 1-dim tensor
                Don't know why there's no error.  """
            # Loss[tgt] = torch.mean( -torch.dot(GT[tgt],Pred[tgt]) ) # In fact here only does -GT[tgt]*Pred[tgt]
            Loss[tgt] = torch.mean( -torch.sum(GT[tgt]*Pred[tgt], dim=1))
        return Loss


class Cos_Proximity_Loss_Handler:
    def __init__(self):
        self.cos_sim = nn.CosineSimilarity(dim=1).cuda()

    def compute_loss(self, tgts, Pred, GT):
        """ tgts: list of target names. In this case   has to be tgts=['norm']
            GT  : dict of ground truth for each target   BxHxWx3
            Pred: dict of prediction   for each target   BxHxWx3
        """
        mask  = GT['mask']
        Loss = edict()
        Errs = edict()
        for tgt in tgts:
            cos_sim = self.cos_sim(Pred[tgt][mask], GT[tgt][mask])
            Loss[tgt] = torch.mean( 1 - cos_sim )  # use 1-cos(theta) to make loss as positive.
            Errs[tgt] = torch.acos(cos_sim.clamp(-1,1))*180./np.pi # .clip(-1,1)
        return Loss, Errs



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
