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
    def __init__(self, post_func=None, **func_kwargs):
        self.batch_data_list = []  # for collecting batch data.
        self.nr_batch = 0  # number of batch collected.
        self.blob_handler = (post_func, func_kwargs)

    def reset(self):
        self.batch_data_list = []  # for collecting batch data.
        self.nr_batch = 0  # number of batch collected.

    def resume(self, pre_batch_data):
        self.batch_data_list = [pre_batch_data]

    def collect(self, batch_data, squeeze=True):
        post_func, func_kwargs = self.blob_handler

        if not isinstance(batch_data, np.ndarray): # batch_data is still pytorch tensor.
            batch_data = batch_data.data.cpu().numpy().copy()
        # batch_data = self.net.blobs[self.blobname].data.copy()
        # batch_data = batch_data.numpy().copy()
        if squeeze:
            batch_data = np.squeeze(batch_data) #.reshape((-1,))

        if post_func is not None:
            batch_data = post_func( batch_data, **func_kwargs )
            if squeeze:
                batch_data = np.squeeze(batch_data)

        if batch_data.shape==():
            # TODO:  what if reduce array is not of shape (batch_size, 1), but (batch_size, c, h, w)?
            batch_data = batch_data.reshape((-1,)) # hack for preventing squeeze single value array.

        self.batch_data_list.append(batch_data)
        self.nr_batch += 1

        # just return a copy of batch_data in case needed
        return batch_data

    def reduce(self, reset=False): #, blobs=None):
        assert len(self.batch_data_list)>0, "[Exception] No data to reduce. blobname: %s" % self.blobname
        batchdata_shape = self.batch_data_list[0].shape
        if   len(batchdata_shape)==0: # scalar for each batch:   loss accurary blobs
            concated_data = np.vstack( self.batch_data_list).reshape((-1,))  # result in (nr_batch, ) array
        else: # if len(batchdata_shape)==1:
            concated_data = np.concatenate( self.batch_data_list, axis=0)
        if reset:
            self.reset()
        return concated_data


class reducer_group:
    def __init__(self, *names):
        self.names = names  # name is gt, pred, scr, loss
        self.name2reducer  = {}
        for name in self.names:
            if isinstance(name, tuple):
                if len(name)==2:
                    name, post_func = name
                    func_kwargs = dict()  # No func kwargs.
                elif len(name)==3:
                    name, post_func, func_kwargs = name
                else:
                    print( "Don't don't how to unpack: blobname, post_func, func_kwargs: %s " % (str(name)))
                    raise NotImplementedError
                self.name2reducer[name] = reducer(post_func, **func_kwargs)

            else: # string, not ppost_func
                self.name2reducer[name] = reducer()

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
