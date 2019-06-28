"""
 @Author  : Shuai Liao
"""

import numpy as np
import scipy.io as spio
from collections import OrderedDict as odict
# http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dictlike):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dictlike:
        if isinstance(dictlike[key], spio.matlab.mio5_params.mat_struct):
            dictlike[key] = _todict(dictlike[key])
    return dictlike

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    od = odict() # {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            od[strg] = _todict(elem)
        elif isinstance(elem,np.ndarray):
            od[strg] = _tolist(elem)
        else:
            od[strg] = elem
    return od

def _tolist(ndarray):
    # return ndarray
    '''
    A recursive function which constructs lists from cellarrays
    (which are loaded as numpy ndarrays), recursing into the elements
    if they contain matobjects.
    '''
    elem_list = []
    for sub_elem in ndarray:
        if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
            elem_list.append(_todict(sub_elem))
        elif isinstance(sub_elem,np.ndarray):
            elem_list.append(_tolist(sub_elem))
        else:
            elem_list.append(sub_elem)
    return elem_list



