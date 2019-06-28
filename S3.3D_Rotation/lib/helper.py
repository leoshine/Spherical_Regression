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
#
from collections import OrderedDict


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
            """ Bug fixed on 22 Aug 2018
                torch.dot can only be applied to 1-dim tensor
                Don't know why there's no error.  """
            # Loss[tgt] = torch.mean( -torch.dot(GT[tgt],Pred[tgt]) ) # In fact here only does -GT[tgt]*Pred[tgt]
            Loss[tgt] = torch.mean( -torch.sum(GT[tgt]*Pred[tgt], dim=1))
        return Loss


class Cos_Proximity_Loss_Handler:
    def __init__(self):
        # self.period_l1_loss = PeriodL1Loss(period=period).cuda()
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


