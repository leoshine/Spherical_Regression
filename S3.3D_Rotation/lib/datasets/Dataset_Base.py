"""
 @Author  : Shuai Liao
"""
import os, sys
import numpy as np
from math import ceil, floor, pi

import torch
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict as odict

import cv2
from basic.common import add_path, env, rdict, cv2_wait, cv2_putText, is_py3
if is_py3:
    import pickle
else:
    import cPickle as pickle
from lmdb_util import ImageData_lmdb
from numpy_db import npy_table, npy_db, dtype_summary, reorder_dtype

this_dir = os.path.dirname(os.path.abspath(__file__))

base_dir = this_dir +'/../../../dataset'  # where the dataset directory is.
assert os.path.exists(base_dir)

cate10 = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']

cate40 = [ 'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl',
           'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser',
           'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop',
           'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant',
           'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool',
           'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']


## Net configurations that are independent of task
netcfg = rdict( # configuration for alexnet
                alexnet=rdict(  TRAIN=rdict(BATCH_SIZE=200),
                                TEST =rdict(BATCH_SIZE=200),
                                INPUT_SHAPE=(227, 227),  # resize_shape
                                PIXEL_MEANS=np.array([[[102.9801, 115.9465, 122.7717]]]),
                                RNG_SEED=3, ),  # ignore_label=-1,
                # configuration for vgg
                vgg16  =rdict(  TRAIN=rdict(BATCH_SIZE=40), #64 20
                                TEST =rdict(BATCH_SIZE=20),
                                INPUT_SHAPE=(224, 224),
                                PIXEL_MEANS=np.array([[[102.9801, 115.9465, 122.7717]]]),
                                RNG_SEED=3, ),
                # configuration for resnet50
                resnet50 =rdict(TRAIN=rdict(BATCH_SIZE=100), # 128
                                TEST =rdict(BATCH_SIZE=64),
                                INPUT_SHAPE=(224, 224),
                                PIXEL_MEANS=np.array([[[102.9801, 115.9465, 122.7717]]]),
                                RNG_SEED=3, ),
                # configuration for resnet101
                resnet101=rdict(TRAIN=rdict(BATCH_SIZE=64),
                                TEST =rdict(BATCH_SIZE=20),
                                INPUT_SHAPE=(224, 224),
                                PIXEL_MEANS=np.array([[[102.9801, 115.9465, 122.7717]]]),
                                RNG_SEED=3, ),
                # configuration for resnet152
                resnet152=rdict(TRAIN=rdict(BATCH_SIZE=32),
                                TEST =rdict(BATCH_SIZE=10),
                                INPUT_SHAPE=(224, 224),
                                PIXEL_MEANS=np.array([[[102.9801, 115.9465, 122.7717]]]),
                                RNG_SEED=3, ),
               )



def get_anno(db_path):  # target=''
    # TO Move to generation of data db.
    viewID2quat  = pickle.load(open(os.path.join(db_path, 'viewID2quat.pkl'), 'rb'), encoding='latin1')
    viewID2euler = pickle.load(open(os.path.join(db_path, 'viewID2euler.pkl'),'rb'), encoding='latin1')
    keys = np.array(list(viewID2quat.keys()))
    add_path(this_dir)
    from db_type import img_view_anno
    rcs = np.zeros( (len(viewID2quat.keys()),), dtype=img_view_anno ).view(np.recarray)
    for i, (key, quat) in enumerate(viewID2quat.items()):
        rc = rcs[i]
        rc.img_id   = key  # bathtub_0107.v001
        cad_id, viewId = key.split('.')
        category = cad_id[:cad_id.rfind('_')]
        rc.category = category
        rc.cad_id   = cad_id
        rc.so3.quaternion = quat if quat[0]>0 else -quat  # q and -q give the same rotation matrix.
                                                          # Make sure all q[0]>0, that is rotation angle in [0,pi]
        rc.so3.euler = viewID2euler[key]

    return keys, rcs


class Dataset_Base(Dataset):
    collection2dbname = \
        dict(train='train_100V.Rawjpg.lmdb',  # 'train_20V.Rawjpg.lmdb'
             test ='test_20V.Rawjpg.lmdb',
        )

    def __init__(self, collection='train', net_arch='alexnet', sampling=1.0):
        self.net_arch   = net_arch
        self.cfg        = netcfg[net_arch]
        self.collection = collection
        self.cates      = cate10
        #
        self.cate2ind = odict(zip(self.cates, range(len(self.cates))))
        # get im_db
        self.db_path = os.path.join(base_dir, 'ModelNet10-SO3', self.collection2dbname[collection])
        assert self.db_path is not None, '%s  is not exist.' % (self.db_path)
        self.datadb = ImageData_lmdb(self.db_path)
        # Get anno
        self.keys, self.recs = get_anno(self.db_path)
        assert sampling>0 and sampling<=1.0, sampling
        if sampling<1.0:
            print('Sampling dataset: %s' % sampling)
            _inds = np.arange(len(self.keys))
            sample_inds = np.random.choice(_inds, size=int(len(_inds)*sampling), replace=False)
            sample_inds.sort()
            self.keys, self.recs = [self.keys[x] for x in sample_inds], self.recs[sample_inds]
        self.key2ind = dict( zip(self.keys, range(len(self.keys))) )
        # self.resize_shape = rsz_shape
        self.mean_pxl = np.array([102.9801, 115.9465, 122.7717], np.float32)
        #

    def _image2data(self, img, data_normal_type='caffe'):
        if self.net_arch=='alexnet':
            img = np.pad(img, [(0,3),(0,3), (0,0)], mode='edge')  # (0,0,3,3)
        # caffe-style
        if   data_normal_type=='caffe':
            # Subtract mean pixel
            data = (img - self.mean_pxl).astype(np.float32)
            # Transpose
            data = data.transpose((2,0,1)) # H,W,C -> C,H,W
        elif data_normal_type=='pytorch':
            #-# img = cv2.cvtColor( img, cv2.COLOR_GRAY2RGB )
            #-# if self.transform is not None:
            #-#     img = self.transform(img) # return (3,224,224)
            raise NotImplementedError
        else:
            raise NotImplementedError
        return data

    def _get_image(self, rc):
        img_id = rc.img_id
        img = self._image2data(self.datadb[img_id])
        return img # if not flip else cv2.flip( img, 1 )

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, idx):
        rcobj    = self.recs[idx]
        cate     = rcobj.category
        obj_id   = rcobj.obj_id
        image_id = rcobj.src_img.image_id
        """ To implement construction of sample dictionary.
            To get image data: call 'self.roiloader(rcobj)'
        """
        print('This is an interface method, and you need to implement it in inherited class.')
        raise NotImplementedError

    def get_recs(self, query_keys):
        inds = [self.key2ind[k] for k in query_keys]
        return self.recs[inds]






class Dataset_Example(Dataset_Base):
    def __getitem__(self, idx):
        rc     = self.recs[idx]
        cate   = rc.category
        # img_id = rc.img_id
        quat = rc.so3.quaternion
        #
        sample = dict( idx   = idx,
                       label = self.cate2ind[cate],
                       quat  = quat,
                       data  = self._get_image(rc) )
        return sample


    # interface method
    def _vis_minibatch(self, sample_batched):
        """Visualize a mini-batch for debugging."""
        for i, (idx, label, quat, data) in enumerate( zip(sample_batched['idx'],  # note: these are tensors
                                                           sample_batched['label'],
                                                           sample_batched['quat'],
                                                           sample_batched['data']) ):
            rc = self.recs[idx]
            # print idx
            im = data.numpy().transpose((1, 2, 0)).copy()
            im += self.cfg.PIXEL_MEANS
            im = im.astype(np.uint8) # xmin, ymax
            a,b,c,d = quat

            cv2_putText(im, (0,20), rc.category, bgcolor=(255,255,255))
            text = '%.1f %.1f %.1f %.1f' % (a,b,c,d)
            cv2_putText(im, (0,40), text, bgcolor=(255,255,255))
            cv2.imshow('im',im)
            cv2_wait()
            # pass


if __name__ == '__main__':
    def test_dataloader():
        dataset = Dataset_Example(collection='test', sampling=0.2)
        #
        dataloader = DataLoader(dataset, batch_size=4,
                                shuffle=False, num_workers=1)

        for i_batch, sample_batched in enumerate(dataloader):
            dataset._vis_minibatch(sample_batched)


    test_dataloader()

