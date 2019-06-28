import numpy as np
import torch


'''
Class hierarchy:
                            net_reducer
                         //      ||     \\
                       //        ||       \\
                     //          ||         \\
                   //            ||           \\
          reducer_group(a)  reducer_group(e)  reducer_group(t)
                           /       |       \
                          /        |        \
                reducer(gt) reducer(pred)   reducer(loss)
'''

class reducer:
    """Bind one blob to a reducer for collection of batch data."""
    def __init__(self, post_func=None, as_numpy=True, **func_kwargs):
        self.batch_data_list = []  # for collecting batch data.
        self.nr_batch = 0  # number of batch collected.
        self.blob_handler = (post_func, func_kwargs)
        self.as_numpy     = as_numpy

    def squeeze(self, arr, dim=None):
        if isinstance(arr, np.ndarray):
            return np.squeeze(arr, axis=dim)
        elif isinstance(arr, torch.Tensor):
            # Although dim is optional args in pytorch, it's not allowed to be None when it's explicitly specified.
            return torch.squeeze(arr, dim=dim)  if (dim is not None)  else torch.squeeze(arr)
        else:
            print ("Unknown array type: ", type(arr))
            raise NotImplementedError

    def cat(self, list_arr, dim=0):
        assert isinstance(list_arr, list) or isinstance(list_arr, tuple), type(list_arr)
        _arr = list_arr[0]
        if   isinstance(_arr, np.ndarray):
            return np.concatenate(list_arr, axis=dim)
        elif isinstance(_arr, torch.Tensor):
            return torch.cat(list_arr, dim=dim )
        else:
            print ("Unknown array type: ", type(_arr))
            raise NotImplementedError

    def reset(self):
        self.batch_data_list = []  # for collecting batch data.
        self.nr_batch = 0  # number of batch collected.

    def resume(self, pre_batch_data):
        self.batch_data_list = [pre_batch_data]

    def collect(self, batch_data, squeeze=True):
        post_func, func_kwargs = self.blob_handler

        if isinstance(batch_data, torch.Tensor):
            batch_data = batch_data.data  # .cpu().numpy().copy()
        #if not isinstance(batch_data, np.ndarray): # batch_data is still pytorch tensor.
        #    batch_data = batch_data.data.cpu().numpy().copy()
        if squeeze:
            batch_data = self.squeeze(batch_data) #.reshape((-1,))

        if post_func is not None:
            batch_data = post_func( batch_data, **func_kwargs )
            if squeeze:
                batch_data = self.squeeze(batch_data)

        if batch_data.shape==():
            # TODO:  what if reduce array is not of shape (batch_size, 1), but (batch_size, c, h, w)?
            batch_data = batch_data.reshape((-1,)) # hack for preventing squeeze single value array.

        self.batch_data_list.append(batch_data)
        self.nr_batch += 1

        # just return a copy of batch_data in case needed
        return batch_data

    def reduce(self, reset=False): #, blobs=None):
        assert len(self.batch_data_list)>0, "[Exception] No data to reduce."
        concated_data = self.cat(self.batch_data_list, dim=0)
        if reset:
            self.reset()
        if isinstance(concated_data, torch.Tensor) and self.as_numpy:
            if concated_data.is_cuda:
                concated_data = concated_data.data.cpu()
            return concated_data.numpy().copy()
        else:
            return concated_data

class reducer_group:
    def __init__(self, target_names, post_func=None, as_numpy=True, **func_kwargs):
        self.names = target_names  # name is gt, pred, scr, loss
        self.name2reducer  = {}
        for name in self.names:
            self.name2reducer[name] = reducer(post_func=post_func, as_numpy=as_numpy, **func_kwargs)

    def reset(self):
        for name in self.names:
            self.name2reducer[name].reset()

    def resume(self, name2pre_batch_data):
        for name in self.names:
            self.name2reducer[name].resume( name2pre_batch_data[name] )

    def collect(self, tgts_dict, squeeze=True):
        """collect add new batch data to list."""
        name2batch_data = {}
        for name, var in tgts_dict.items():
            batch_data = self.name2reducer[name].collect(var, squeeze=squeeze)
            name2batch_data[name] = batch_data
        # just return a copy of batch_data in case needed
        return name2batch_data

    def reduce(self, reset=False):
        """reduce only return data, will change anything."""
        name2data = {} # edict()
        for name, reducer in self.name2reducer.items():
            name2data[name] = reducer.reduce()
        if reset:
            self.reset()
        return name2data

'''
class reducer_group:
    def __init__(self, **name2varName):
        self.name2varName = name2varName  # name is gt, pred, scr, loss
        self.name2reducer  = {}
        for name, varName in self.name2varName.items():
            if isinstance(varName, tuple):
                if len(varName)==2:
                    varName, post_func = varName
                    func_kwargs = dict()  # No func kwargs.
                elif len(varName)==3:
                    varName, post_func, func_kwargs = varName
                else:
                    print "Don't don't how to unpack: blobname, post_func, func_kwargs: %s " % (str(varName))
                    raise NotImplementedError
                self.name2reducer[name] = reducer(post_func, **func_kwargs)

            else: # string, not ppost_func
                self.name2reducer[name] = reducer()

    def collect(self, tgts_dict):
        """collect add new batch data to list."""
        name2batch_data = {}
        for name, var in tgts_dict.items():
            batch_data = self.name2reducer[name].collect(var)
            name2batch_data[name] = batch_data
        # just return a copy of batch_data in case needed
        return name2batch_data

    def reduce(self):
        """reduce only return data, will change anything."""
        name2data = {} # edict()
        for name, reducer in self.name2reducer.items():
            name2data[name] = reducer.reduce()
        return name2data
'''


# class net_reducer:
#     """A netreducer handel all groups of reducer."""
#     def __init__(self, net):
#         self.net    = net
#         self.groups = {}
#
#     def add_group(self, gname, **name2blobname):
#         self.groups[gname] = reducer_group(self.net, **name2blobname)
#
#     def collect(self):
#         g_name2g_batch_data = {}
#         for g_name, group in self.groups.items():
#             g_batch_data = group.collect()  # a dictionary
#             g_name2g_batch_data[g_name] = g_batch_data
#
#         # just return a copy of batch_data in case needed
#         return g_name2g_batch_data
#
#     def reduce(self):
#         group2data = {}  # group name to data
#         for g_name, group in self.groups.items():
#             group2data[g_name] = group.reduce()
#         return group2data
#
#
# if __name__=="__main__":
#     net = None
#
#     rd = reducer(net)
#     rd.add('e3')
#
#     # g_redu = reducer_group(net)
#     # g_redu.add_group("a", gt='e3', pred='prob_e3', scr='prob_e3', loss='loss_e3', acc='acc_e3')
#     # g_redu.add_group("e", gt='e2', pred='prob_e2', scr='prob_e2', loss='loss_e2', acc='acc_e2')
#     # g_redu.add_group("t", gt='e1', pred='prob_e1', scr='prob_e1', loss='loss_e1', acc='acc_e1')
#
#     g_redu = net_reducer(net)
#     g_redu.add_group("a", gt='e3', scr='prob_e3', loss='loss_e3', acc='acc_e3')
#     g_redu.add_group("e", gt='e2', scr='prob_e2', loss='loss_e2', acc='acc_e2')
#     g_redu.add_group("t", gt='e1', scr='prob_e1', loss='loss_e1', acc='acc_e1')
#
#
#
#
#
#
#
# '''
# class gt(reduce):
#     def __init__(self, net, blobname):
#         Layer.__init__(self, type='scalar')
#
# gt = reduce("e3")
#
#     # # for classification
#     # name2gt    = {}  # gt   class label
#     # name2pred  = {}  # pred class label
#     # name2score = {}  # pred scores for all classes
#
#     # ------ Prefix -------
#     # Accuary:  acc
#     # Softmax:  prob
#     # Loss:     loss
#     #
#     # group_type2blobname
#     group("a", gt='e3', pred='prob_e3', scr='prob_e3', loss='loss_e3', acc='acc_e3')
#     reduce_dict = { #                       Softmax output           Softmax output  SoftmaxLoss    Accurary
#                 "a": oedict(gt='e3', pred='prob_e3', scr='prob_e3', loss='loss_e3', acc='acc_e3'),  # np.argmax, axis=-1
#                 "e": oedict(gt='e2', pred='prob_e2', scr='prob_e2', loss='loss_e2', acc='acc_e2'),
#                 "t": oedict(gt='e1', pred='prob_e1', scr='prob_e1', loss='loss_e1', acc='acc_e1'),
#     }
#
#     # reduce_group = { #                       Softmax output           Softmax output  SoftmaxLoss    Accurary
#     #             "a": oedict(gt='e3', pred=(np.argmax,'prob_e3'), scr='prob_e3', loss='loss_e3', acc='acc_e3'),  # np.argmax, axis=-1
#     #             "e": oedict(gt='e2', pred=(np.argmax,'prob_e2'), scr='prob_e2', loss='loss_e2', acc='acc_e2'),
#     #             "t": oedict(gt='e1', pred=(np.argmax,'prob_e1'), scr='prob_e1', loss='loss_e1', acc='acc_e1'),
#     # }
#
# from basic.common import RefObj as rdict
#
# class extractor:
#     """Extract data from blob"""
#     if __init__(self, blobname, )
#
#
# class batch_reducer:
#     """Collect and reduce the results from each batch and concatenate them into one.
#         Reduce type:
#             - scalar
#             - vector (softmax  or  simply feature.)
#             - vector with argmax.
#         Type name:
#             gt, pred, scr, loss, acc
#         It results in:
#             name2gt, name2pred, name2scr, name2loss, name2acc  (if exists)
#     """
#     scalar=set(['gt', 'pred', 'loss','acc'])
#     vector=set(['scr'])
#
#     def __init__(self, netblobs, **kwargs):
#             # self.reduce_dict =
#             self.name2gt = {}
#
#     def collect(self, post_func=None, **func_kwargs):
#         for key, tgt in kwargs:
#             name2tgt = 'name2%s' % key
#             if key in ['gt','pred','scr','loss','acc']:
#                 if not hasattr(self, name2tgt ):
#                     self.__setattr__(name2tgt) = {}  # create dict.
#
#                 if   key in scalar:
#                     assert netblobs[tgt].data.shape==1
#                 elif key in vector:
#                     assert netblobs[tgt].data.shape>1
#                 else: raise NotImplementedError
#
#                 if post_func is None:
#                     batch_data = netblobs[tgt].data.copy()
#                 else:
#                     batch_data = post_func( netblobs[tgt].data.copy(), func_kwargs)
#                 self.__getattr__(name2tgt).setdefault(tgt, []).append( batch_data )
#             else:
#                 print "[Error] Don't know how to reduce type: %s " % key
#                 raise NotImplementedError
#
#     def reduce(self)
#         for key, tgt in kwargs:
#             name2tgt = 'name2%s' % key
#             if key in ['gt','pred','scr','loss','acc']:
#                 # if not hasattr(self, name2tgt ):
#                 #     self.__setattr__(name2tgt) = {}  # create dict.
#                 concated = np.concatenate( self.__getattr__(name2tgt), axis=0)
#                 self.__setattr__(name2tgt, concated )
#
#             if   key=='gt':
#                 self.name2gt.setdefault(tgt, []).append( netblobs[key].data.copy() )
#             elif key=='pred': # reduce from Softmax by applying argmax
#                 self.name2gt.setdefault(tgt, []).append( netblobs[key].data.copy() )
#             elif key=='scr':  # reduce from Softmax
#                 self.name2gt.setdefault(tgt, []).append( netblobs[key].data.copy() )
#             elif key=='loss': # reduce from Loss layer output.
#                 self.name2gt.setdefault(tgt, []).append( netblobs[key].data.copy() )
#             elif key=='acc':  # reduce from Accuray layer.
#                 self.name2gt.setdefault(tgt, []).append( netblobs[key].data.copy() )
#             else:
#                 print "Don't know how to reduce type: %s " % key
#                 raise NotImplementedError
#     def
# '''
#
#
