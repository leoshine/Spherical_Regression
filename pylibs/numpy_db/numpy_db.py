"""
 @Author  : Shuai Liao
"""

import os
import numpy as np
# import cPickle as pickle
# python2/3 compatible
import six
from six.moves import cPickle as pickle

import h5py, gzip # , re
from basic.common import RefObj, Open, env, is_py3 # , parse_yaml
from collections import OrderedDict as odict


class npy_table:
    @staticmethod
    def merge(table_list):
        for tbl in table_list:
            assert isinstance(tbl, npy_table)
        primary_key = table_list[0].primary_key
        for tbl in table_list:
            assert tbl.primary_key==primary_key
        recs_list = [tbl.recs for tbl in table_list]
        return npy_table(np.concatenate(recs_list, axis=0))

    def __init__(self, recarr, primary_key=None):  # , yaml_conf=None
        self.dtype = recarr.dtype
        self.shape = recarr.shape
        self.primary_key = primary_key
        if self.primary_key is None:
            self.primary_key = self.dtype.names[0]  # default primary_key for an np.array is the first field of the dtype.
        # ---------------------------------------
        # Notice: The main part of npy_table is
        #    - npy_table.recs
        #    - npy_table.keys
        self.recs  = recarr  # recarr is numpy.array  or  numpy.recarray
        self.keys  = self.recs[self.primary_key]
        # check primary key unique.
        assert len(self.keys)==len(set(self.keys)), '[Error] keys of primary_key are not unique: %s != %s ' % (len(self.keys), len(set(self.keys)))
        #
        self.key2ind = dict( zip(self.keys, range(len(self.keys))) )

    @property
    def data(self):
        return self.keys, self.recs.view(np.recarray)

    # def rc(self, key):
    #     """Get record by key."""
    #     ind = self.key2ind[key]
    #     return self.recs[ind]

    def __getitem__(self, key):
        """Get record by key."""
        ind = self.key2ind[key]
        return self.recs[ind].view(np.recarray)  # view as recarray   self.recs[ind]


def reorder_dtype(old_dtype, primary_key):
    """Put the primary key at the first field of dtype."""
    old_field_names = list(old_dtype.names)
    _ind = old_field_names.index(primary_key)

    new_field_names = [primary_key] + old_field_names[:_ind] + old_field_names[_ind+1:]
    new_dtype = [(fname, old_dtype.fields[fname][0]) for fname in new_field_names]  # dtype.fields return (dtype, offset[, title])

    # Great that astype can actually have more fields.
    return np.dtype(new_dtype)


class npy_db:
    def __init__(self):
        self.name2table = odict() # {}
        self.infile = None

    def __str__(self):
        lines = []
        lines.append('====================================')
        lines.append('[file] %s' % self.infile)
        lines.append('---------- npy_db summary ----------')
        for name, tb in self.name2table.items():
            lines.append('   [Table] %-15s %-15s %s' % ('"%s"'%name, 'shape=%s'%str(tb.recs.shape), str(tb.dtype)[:50]+'  ...' ))
        return '\n'.join(lines)+'\n'

    def add_table(self, obj, name=None ):
        """ obj can be an npy_table  or  a numpy array"""
        if   isinstance(obj, npy_table):
            self.name2table[name] = obj
        elif isinstance(obj, np.ndarray) or isinstance(obj, np.recarray):
            self.name2table[name] = npy_table(obj)
        else:
            raise NotImplementedError

    def __getitem__(self, tb_name):
        return self.name2table[tb_name]

    def dump(self, outfile): #  pkl  or hdf5  # method='pickle'  or hdf5 h5py
        """Dump to pickle"""
        # For this moment, we treat primary_key for an np.array is the first field of the dtype.
        # So we don't dump primary_key here.
        name2recarr = {}
        for name, table in self.name2table.items():
            assert isinstance(table.recs, np.ndarray) or isinstance(table.recs, np.recarray), '[Exception] %s' % (type(table.recs))
            #print len(table.dtype.tolist())
            old_dtype = table.recs.dtype
            new_dtype = reorder_dtype(old_dtype, table.primary_key)
            table.recs = table.recs.astype(dtype=new_dtype)

            assert table.primary_key==table.recs.dtype.names[0], '[Exception] table "%s" has non-first field primary_key "%s"' % (name, table.primary_key)
            name2recarr[name] = table.recs

        try:  os.makedirs(os.path.dirname(outfile))
        except: pass

        if   outfile.endswith('pkl'):
            pickle.dump(name2recarr, open(outfile,'wb'),-1)
        elif outfile.endswith('pkl.gz'):
            pickle.dump(name2recarr, gzip.open(outfile,'wb'),-1)
        elif outfile.endswith('hdf5') or outfile.endswith('hdf5.gz'):  # h5py
            to_compress = outfile.endswith('hdf5.gz')
            hdb = h5py.File(outfile, 'w')
            for name, recarr in name2recarr.items():
                if to_compress:
                    hdb.create_dataset(name, data=recarr, compression='gzip')
                else:
                    hdb.create_dataset(name, data=recarr)
            hdb.close()
        else:
            print ('[Exception] unknown type of dumped file: "%s". \nNow only support "pkl", "pkl.gz", "hdf5", "hdf5.gz" ' % outfile)
            raise NotImplementedError

        print ('[npy_db] dumped to: %s' % outfile)

    def load(self, infile):
        """Load from pickle"""
        # print '[npy_db] loading from: %s' % infile
        self.infile = infile
        # backend = os.path.splitext(infile)[-1]
        if   infile.endswith('pkl'):
            name2recarr = pickle.load(open(infile,'rb'))
            for name, recarr in name2recarr.items():
                self.name2table[name] = npy_table(recarr)
        elif infile.endswith('pkl.gz'):
            name2recarr = pickle.load(gzip.open(infile,'rb'))
            for name, recarr in name2recarr.items():
                self.name2table[name] = npy_table(recarr)
        elif infile.endswith('hdf5') \
          or infile.endswith('hdf5.gz'):
            hdb = h5py.File(infile, 'r')
            for name, hdf_dataset in hdb.items():
                self.name2table[name] = npy_table(hdf_dataset[:])  # hdf_dataset[:] will trigger read IO from file.
            hdb.close()
        else:
            print ('[Exception] unknown type of dumped file: "%s". \nNow only support "pkl", "pkl.gz", "hdf5", "hdf5.gz" ' % infile)
            raise NotImplementedError
        # print '         tables:  %s' % self.name2table.keys()


def dtype_summary(dtype, indent=0):

    if indent==0:
        print ('======================= [Ttl itemsize=%s] =======================' % (dtype.itemsize))

    for na in dtype.names:
        dt, offset = dtype.fields[na]
        if dt.names is not None:
            print ("[%s]------------- [itemsize=%s]" % (na, dt.itemsize))
            dtype_summary(dt, indent+1)
        else:
            # print "%-20s" % na,  "%-30s  %-30s" % (dt,dt.shape), dt.descr
            if dt.subdtype is None:
                # print "\t"*indent, "%-20s" % na,  "%-30s  %-20s %-20s" % (dt,dt.shape, dt.itemsize)
                print ("%-40s" % ("    "*indent+na),  "%-20s  %-10s %-10s" % (dt,dt.shape, dt.itemsize))
            else:
                item_dtype, shape = dt.subdtype
                # print "\t"*indent, "%-20s" % na,  "%-30s  %-20s %-20s" % (item_dtype, shape, dt.itemsize)
                print ("%-40s" % ("    "*indent+na),  "%-20s  %-10s %-10s" % (item_dtype, shape, dt.itemsize))






#---------------------------------------------------------------[Test case]
def test_table_db1():
    collection, cad_type = "Val", 1
    data_dir = env.HOME+'/working/cvpr17pose/code/mycode/Subcnn.surgery/Voxelproj_RC_net/data'
    anno_file = os.path.join(data_dir, 'cache/binnedJoint_%s_no_augment-with_proj_param.flat_obj_list.aeroplane-%02d.pkl' % (collection, cad_type))
    obj_ids, obj_recs  = pickle.load(open(anno_file))
    # test table
    npt = npy_table(obj_recs, primary_key='object_id')
    # test db
    db = npy_db()
    db.add_table(npt, name="obj_recs")
    db.dump('test_db.pkl')

def test_table_db2():
    # Check: https://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.html
    db = npy_db()
    db.load('test_db.pkl')
    # for na in db.name2table['obj_recs'].dtype.names:
    #     dt, offset = db.name2table['obj_recs'].dtype.fields[na]
    #     if dt.subdtype is None:
    #         print "%-20s" %na, (dt,dt.shape)
    #     else:
    #         print "%-20s" %na, dt.subdtype
    dtype_summary(db.name2table['obj_recs'].dtype)


if __name__=="__main__":
    #test_table_db1()
    test_table_db2()
    exit()


    collection, cad_type = "Val", 1
    rootpath= env.HOME + '/working/cvpr17pose/code/mycode/Subcnn.surgery/Voxelproj_RC_net/prototxts/alexnet_triplet_rank/cache/PNCC_img.aeroplane01/'
    anno_file = rootpath + '%s_anno.pkl' % collection
    print ('Load from pickled cache file: ', anno_file)
    rendered_ids, rendered_annos = pickle.load(open(anno_file))
    assert len(rendered_ids)==len(rendered_annos)
    rendered_id2Ind = dict(zip(rendered_ids, range(len(rendered_ids))))
    dtype_summary(rendered_annos.dtype)


    dd = rendered_annos['proj_rcd'].dtype
    dtype_summary(dd)
    new_dd = reorder_dtype(dd, 'anchor_visib')
    dtype_summary(new_dd)
    exit()


    collection, cad_type = "Val", 1
    data_dir = env.HOME+'/working/cvpr17pose/code/mycode/Subcnn.surgery/Voxelproj_RC_net/data'
    anno_file = os.path.join(data_dir, 'cache/binnedJoint_%s_no_augment-with_proj_param.aeroplane-%02d.pkl' % (collection, cad_type))
    ll = pickle.load(open(anno_file))
    dtype_summary(ll[0]['objects'].dtype)


