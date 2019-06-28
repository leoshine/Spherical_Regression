import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

np.set_printoptions(precision=5, suppress=True)


def norm2unit(vecs, dim=1, p=2):
    """ vecs is of 2D:  (batchsize x nr_feat) or (batchsize x nr_feat x restDims)
        We normalize it to a unit vector here.
        For example, since mu is the coordinate on the circle (x,y) or sphere (x,y,z),
        we want to make it a unit norm.
    """
    vsize = vecs.size()
    batchsize, nr_feat = vsize[0], vsize[1]

    ##  check bad input
    if hasattr(vecs, 'data'): # vecs is Variable
        check_inf = ( torch.abs(vecs.data)==float('inf') ) # It's weird that Variable cannot compare with float('inf')
    else:                     # vecs is Tensor
        check_inf = ( torch.abs(vecs)     ==float('inf') )      # infinite can be '-inf' and 'inf'
    check_nan = ( vecs!=vecs )  # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/4
    if check_inf.any() or check_nan.any():
        print (vecs)
        print ('[Exception] some input values are either "nan" or "inf" !   (from norm2unit)')
        exit()

    # Trick for numeric stability (norm won't becomes 0)
    # vecs = vecs + torch.sign(vecs)*1e-4   #  Warning: sign(0) --> 0
    signs = torch.sign(vecs)
    signs[signs==0] = 1
    vecs = vecs + signs*1e-4

    # print mu
    # Just for debugging in case
    vecs_in = vecs.clone()

    # Compute norm.
    # Note the detach(), that is essential for the gradients to work correctly.
    # We want the norm to be treated as a constant while dividing the Tensor with it.
    # norm = torch.norm(vecs, p, dim, keepdim=True).detach()  # here p=2 just means l2 norm  Warning: L1 norm is the sum of absolute values.
    norm = ((vecs**p).sum(dim=dim, keepdim=True)**(1./p)).detach()  # Warning:  if p=1, this line doesn't use absolute values as L1 norm.
                                                                    # Warning:  In pytorch doc of "torch.norm()", the formula is not correct.
                                                                    #           It should be sqrt[p]{|x_1|^p + |x_2|^p + ... + |x_N|^p}
                                                                    #           (https://pytorch.org/docs/stable/torch.html#torch.norm)
    # print vecs[:5,:]
    # print norm[:5,:]
    # import torch.nn.functional as F #  normalize # .normalize(input, p=2, dim=1, eps=1e-12, out=None)
    # print F.normalize(vecs[:5,:], p=p, dim=1)
    # exit()
    # norm = norm.view(batchsize,1,*vsize[2:])

    # Check if any norm are close to 0
    # Should happen anymore since we have done "vecs = vecs + torch.sign(vecs)*1e-4"
    check_bad_norm = torch.abs(norm).lt(1e-4)
    if check_bad_norm.any():
        print (check_bad_norm)
        print ('[Exception] some norm close to 0.   (from norm2unit)')
        exit()

    # do normalize by division.
    vecs  = vecs.div(norm)
    # vecs  = vecs.div(norm.expand_as(vecs))

    recomputed_norm = torch.sum((vecs**p), dim=dim)
    check_mu = torch.abs(recomputed_norm-1.0).lt(1e-6)
    if not check_mu.all():
        np.set_printoptions(threshold=np.inf)
        # print mu.data.cpu().numpy()
        # print (mu**2).sum().eq(1).data.cpu().numpy()
        print (norm.data.cpu().numpy())
        print (torch.cat([vecs_in, vecs, norm, recomputed_norm.view(batchsize, 1)], dim=dim)[~check_mu,:].data.cpu().numpy())
        print ('[Exception] normalization has problem.   (from norm2unit)')
        exit()
    # assert (mu**2).sum().eq(1).all(), '[Exception] normalization has problem.'

    return vecs



def _check_inf_nan(vecs):
    # check if vecs contains inf
    if hasattr(vecs, 'data'): # vecs is Variable
        check_inf = ( torch.abs(vecs.data)==float('inf') ) # It's weird that Variable cannot compare with float('inf')
    else:                     # vecs is Tensor
        check_inf = ( torch.abs(vecs)     ==float('inf') ) # infinite can be '-inf' and 'inf'
    # check if vecs contains nan
    check_nan = ( vecs!=vecs )  # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/4

    if check_inf.any() or check_nan.any():
        print (vecs)
        print ('[Exception] some input values are either "nan" or "inf" !   (from norm2unit)')
        exit()
    # return check_inf.any() or check_nan.any()

def _check_bad_norm(norm_vecs):
    # Check if any norm are close to 0
    # Should happen anymore since we have done "vecs = vecs + torch.sign(vecs)*1e-4"
    check_bad_norm = torch.abs(norm_vecs).lt(1e-4)
    if check_bad_norm.any():
        print (check_bad_norm)
        print ('[Exception] some norm close to 0.   (from norm2unit)')
        exit()



def exp_Normalization(vecs, l_n=2, debug=True):  # when l1, same as softmax
    """ vecs is of 2D:  (batchsize x nr_feat)
        We first apply exponential, and then normalize it to a unit vector here.
        Compute:
            x_i / ( sum( x_j**l_n ) ) ** (1/l_n)
         e.g.:  L1:   x_i / sum( x_j )
                L2:   x_i / sqrt( sum( x_j**2 ) )

        Note: signs of input vecs is lost. (since this is exponential always return positive number.)
    """
    batchsize, nr_feat = vecs.size()

    # check if input are all valid number.
    _check_inf_nan(vecs)

    # apply exponential
    vecs = torch.exp(vecs)
    # check again
    _check_inf_nan(vecs)

    # Trick for numeric stability (norm won't becomes 0)
    vecs = vecs + torch.sign(vecs)*1e-4

    # Just for debugging in case
    if debug:
        _vecs_in_ = vecs.clone()

    # Compute norm.
    norm = torch.norm(vecs, p=l_n, dim=1).view(batchsize,1)  # here p=2 just means l2 norm
    # Note the detach(), that is essential for the gradients to work correctly.
    # We want the norm to be treated as a constant while dividing the Tensor with it.
    if hasattr(vecs, 'data'): # vecs is Variable
        norm = norm.detach()

    if debug: _check_bad_norm(norm)

    # do normalize by division.
    vecs = vecs.div(norm.expand_as(vecs))

    if debug:
        check_unit = torch.abs(torch.sum((vecs**l_n), dim=1)-1.0).lt(1e-6)
        if not check_unit.all():
            print (torch.cat([_vecs_in_, vecs, norm], dim=1).data.cpu().numpy())
            print ('[Exception] normalization has problem.   (from exp_Normalization)')
            exit()

    return vecs



def as_numpy(x):
    x = x.data if isinstance(x, Variable) else x
    if x.is_cuda:
        return x.cpu().numpy()
    else:
        return x.numpy()

if __name__ == '__main__':

    # ---- Generate input vecs -----
    batchsize = 3
    # vecs = torch.Tensor(batchsize, 3).uniform_(0, 1)
    vecs = torch.randn(batchsize, 10)
    vecs = torch.autograd.Variable(vecs)

    #--------------
    print ("[Input]", vecs)
    expNorm = exp_Normalization(vecs.clone(), l_n=1, debug=True)
    # print "[ExpL1]", expNorm
    print ("[LogExpL1]", torch.log(expNorm))

    #-------------- Calling pytorch build function.
    import torch.nn.functional as F
    # print "[F.softmax]", F.softmax(vecs.clone(), dim=1, _stacklevel=3)
    print ("[F.log_softmax]", F.log_softmax(vecs.clone(), dim=1)) # , _stacklevel=5



