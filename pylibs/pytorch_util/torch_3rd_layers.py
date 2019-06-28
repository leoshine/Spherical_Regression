import torch
import torch.nn as nn
import numpy as np
from basic.common import is_py3
if is_py3:
    from functools import reduce


'''
Define Maskout layer
'''
class Maskout(nn.Module):
    def __init__(self, nr_cate=3):  # ks: kernel_size
        super(Maskout, self).__init__()
        self.nr_cate = nr_cate

    def forward(self, x, label):
        '''[input]
             x:    of shape (batchsize, nr_cate, nr_feat)
                or of shape (batchsize, nr_cate)   where nr_feat is treat as 1.
             label:
           [output]
             tensor of shape (batchsize, nr_cate)
        '''
        batchsize, _nr_cate = x.size(0), x.size(1) # x can be (batchsize, nr_cate) or (batchsize, nr_cate, nr_feat)
        assert _nr_cate==self.nr_cate, "2nd dim of x should be self.nr_cate=%s" % self.nr_cate
        assert batchsize==label.size(0)            # first dim equals to batchsize
        assert batchsize==reduce(lambda a1,a2:a1*a2, label.size()) # total size equals to batchsize

        item_inds = torch.from_numpy(np.arange(batchsize))
        if x.is_cuda:
            item_inds = item_inds.cuda()
        cate_ind  = label.view(batchsize)
        assert cate_ind.lt(self.nr_cate).all(), '[Exception] All index in cate_ind should be smaller than nr_cate.'
        masked_shape = (batchsize,) + x.size()[2:]
        return x[item_inds, cate_ind].view(*masked_shape)  # (*label.size()) #(batchsize,1)



"""
x.shape:
    batch_size * nr_cate * nr_bins
"""
