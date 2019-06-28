"""
 @Author  : Shuai Liao
"""

from easydict import EasyDict as edict
from collections import OrderedDict as odict
import sys
is_py3 = (sys.version_info > (3, 0))
try:
    from thread import get_ident as _get_ident
except ImportError:
    if is_py3: from _dummy_thread import get_ident as _get_ident
    else     : from dummy_thread  import get_ident as _get_ident


# In python 2, all of your classes should inherit from object.
# If you don't, you end up with "old style classes", which are always of type classobj,
# and whose instances are always of type instance.
class Ordered_EasyDict(object):
    def __init__(self, iterable_KeysValues=None):
        """ [param] iterable_KeysValues:
            * it can be `list', `OrderedDict'.
            * if it is  `dict', there actually no order (in this case the order is same as it is in dict obj)
            * can not be EasyDict, as it doesn't support use list to initialize,
        """
        self.__dict__['_odict'] = odict()   # avoid to call __setattr__ here
        if iterable_KeysValues is not None:
            self.update(iterable_KeysValues)

    def update(self, iterable_KeysValues): # Note: here update means "overwrite"!
        # print 'updating ...'
        # if isinstance(iterable_KeysValues, Ordered_EasyDict):
        o_list = odict(iterable_KeysValues)
        for k, v in o_list.items():
            self[k] = v # by __setattr__ (now we have already initialized _edict and _ordered_keys)


    def __len__(self):
        return len(self.__dict__['_odict'])

    def __getattr__(self, key): # make obj indexable
        """ for obj.xxx operation """
        return self._odict[key]
        # if there's no __len__ method, there will be a issue of no key "__len__" when call list(this_obj)

    def __setattr__(self, key, value):
        """ for obj.xxx = yyy  operation """
        if type(value) in [dict, edict, odict]:
            value = Ordered_EasyDict(value)  # recurrently make value as oedict if possible.
        self._odict[key] = value
        # self._ordered_keys.append(key)

    def __setitem__(self, key, value): # make obj indexable
        """ for obj['xxx'] operation """
        if type(value) in [dict, edict, odict]:
            value = Ordered_EasyDict(value)  # recurrently make value as oedict if possible.
        return self.__setattr__(key, value)

    def __getitem__(self, key): # make obj indexable
        """ for obj['xxx'] = yyy operation """
        return self._odict[key] # self.__getattr__(key)

    def __iter__(self):
        # makes an object iterable
        return self._odict.__iter__()

    def __contains__(self, key):
        return key in self._odict

    def keys(self):
        '''oedict.keys() -> list of values in oedict'''
        return self._odict.keys()
        # return self._ordered_keys

    def values(self):
        '''oedict.values() -> list of values in oedict'''
        return [self[key] for key in self._odict]

    def items(self):
        '''oedict.items() -> list of (key, value) pairs in oedict'''
        return self._odict.items()

    def __repr__(self, _repr_running={}): # [Adapted from]  https://github.com/python/cpython/blob/2.7/Lib/collections.py
        'oedict.__repr__() <==> repr(oedict)'
        # return '@oedict@'
        call_key = id(self), _get_ident()
        if call_key in _repr_running:
            return '...'
        _repr_running[call_key] = 1
        try:
            if not self._odict:
                return '%s()' % (self.__class__.__name__,)
            return '%s(%r)' % (self.__class__.__name__, self.items())
        finally:
            del _repr_running[call_key]

    def __str__(self):
        return "{" +", ".join([str((k,v)) for k,v in self._odict.items()]) +"}"


def test_case1():
    inblob2shape = Ordered_EasyDict()
    inblob2shape.label = [1,1]
    inblob2shape.e1 = [1,1]
    inblob2shape.e2 = [1,1]
    inblob2shape.e3 = [1,1]
    inblob2shape.e1coarse = [1,1]
    inblob2shape.e2coarse = [1,1]
    inblob2shape.e3coarse = [1,1]
    inblob2shape.data = [1,3,227,227]
    print (list(inblob2shape))
    print (inblob2shape)

def test_case2():
    tpls = [('label', [1, 1]), ('e1', [1, 1]), ('e2', [1, 1]), ('e3', [1, 1]), ('e1coarse', [1, 1]), ('e2coarse', [1, 1]), ('e3coarse', [1, 1]), ('data', [1, 3, 227, 227])]
    # tpls = odict(tpls)
    # tpls = dict(tpls) # dangerous
    inblob2shape = Ordered_EasyDict(tpls)
    print (list(inblob2shape))
    print (inblob2shape)


def test_case3():
    print ('-----------------------')
    tpls1 = [('label', [1, 1]), ('e1', [1, 1]), ('e2', [1, 1]), ('e3', [1, 1]), ]
    tpls2 = [('e3coarse', [1, 1]), ('e1coarse', [1, 1]), ('e2coarse', [1, 1]), ('data', [1, 3, 227, 227])]

    od1 = Ordered_EasyDict(tpls1)
    od2 = Ordered_EasyDict(tpls2)
    # TODO  to implement update
    od1.update(od2)
    print ('keys:')
    print (od2.keys())
    print ('items:')
    print ([k for k,v in od1.items()])
    print ('iter:')
    for k in od1: #.items():
        print (k)



if __name__ == '__main__':
    test_case1()
    test_case2()
    test_case3()


