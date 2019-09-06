"""
 @Author  : Shuai Liao
"""

import os, sys
import numpy as np
from math import ceil, floor, pi

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from collections import OrderedDict as odict

import cv2
from basic.common import add_path, env, rdict, cv2_wait, cv2_putText
from lmdb_util import ImageData_lmdb
from numpy_db import npy_table, npy_db, dtype_summary, reorder_dtype

this_dir = os.path.dirname(os.path.abspath(__file__))
Home_Dir = env.Home

base_dir = os.path.abspath(this_dir +'/../../../dataset')  # where the dataset directory is.
assert os.path.exists(base_dir), base_dir

## Net configurations that are independent of task
netcfg = rdict( # configuration for vgg
                vgg16  =rdict(  TRAIN=rdict(BATCH_SIZE=10), #64 20
                                TEST =rdict(BATCH_SIZE=5),  #10
                                INPUT_SHAPE=(224, 224),
                                PIXEL_MEANS=np.array([[[102.9801, 115.9465, 122.7717]]]),
                                RNG_SEED=3, ),
                vgg16se=rdict(  TRAIN=rdict(BATCH_SIZE=10), #64 20
                                TEST =rdict(BATCH_SIZE=5 ),
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


caffe_mean_pxl=[102.9801, 115.9465, 122.7717]


def sample_keys(keys, rate):
    _inds = np.arange(len(keys))
    sample_inds = np.random.choice(_inds, size=int(len(_inds)*rate), replace=False)
    sample_inds.sort()
    return [keys[x] for x in sample_inds]

class Dataset_Base(Dataset):
    def __init__(self, collection='train', net_arch='vgg16', style='pytorch', with_flip=False, sampling=dict(nyu=1.0, syn=0.0), Ladicky_normal=False): #
        self.net_arch   = net_arch
        self.cfg        = netcfg[net_arch]
        self.collection = collection
        self.Ladicky_normal = Ladicky_normal
        #
        if collection=='test':
            assert not with_flip, 'Test collection should without flip.'
            assert sampling['syn']==0
        self.with_flip = with_flip
        #
        db_path = base_dir+'/SurfaceNormal/nyu_v2/{}.lmdb'
        id_path = base_dir+'/SurfaceNormal/nyu_v2/{}Ndxs.txt'
        self.nyu_dbImage = ImageData_lmdb(db_path.format('ImageData.Rawpng') , always_load_color=False) # cv2.IMREAD_UNCHANGED
        if Ladicky_normal:
            print ('Using GT normal of NYU v2 from Ladicky et al.  and  ALL Valid  mask')
            self.nyu_dbNorm  = ImageData_lmdb(db_path.format('NormCamera_Ladicky.Rawpng'), always_load_color=False) # cv2.IMREAD_UNCHANGED
            self.nyu_dbValid = ImageData_lmdb(db_path.format('Valid_ALL.Rawpng')     , always_load_color=False) # cv2.IMREAD_UNCHANGED
        else:
            self.nyu_dbNorm  = ImageData_lmdb(db_path.format('NormCamera.Rawpng'), always_load_color=False) # cv2.IMREAD_UNCHANGED
            self.nyu_dbValid = ImageData_lmdb(db_path.format('Valid.Rawpng')     , always_load_color=False) # cv2.IMREAD_UNCHANGED
        if sampling['syn']>0:
            syn_db_path = base_dir+'/SurfaceNormal/pbrs/{}.lmdb'
            syn_id_path = base_dir+'/SurfaceNormal/pbrs/data_goodlist_v2.txt'
            self.syn_dbImage = ImageData_lmdb(syn_db_path.format('ImageData.Rawjpg') , always_load_color=False) # cv2.IMREAD_UNCHANGED
            self.syn_dbNorm  = ImageData_lmdb(syn_db_path.format('NormCamera.Rawpng'), always_load_color=False) # cv2.IMREAD_UNCHANGED
            self.syn_dbValid = ImageData_lmdb(syn_db_path.format('Valid.Rawpng')     , always_load_color=False) # cv2.IMREAD_UNCHANGED
            syn_keys = list(map(str.strip, open(syn_id_path).readlines()))
            if sampling['syn']<1:
                syn_keys = sample_keys(syn_keys, sampling['syn'])
        else:
            syn_keys = []
        #
        nyu_keys = list(map(lambda x:x.strip().split('/')[1], open(id_path.format(collection)).readlines()))  # testNdxs.txt   trainNdxs.txt
        if sampling['nyu']<1.0:
            print ('Sampling dataset: %s' % sampling)
            nyu_keys = sample_keys(nyu_keys, sampling['nyu'])

        #
        self.keys = nyu_keys + syn_keys # Note: NYU keys always goes first
        self.key2ind = dict( zip(self.keys, range(len(self.keys))) )
        self._nyu_idx_range = len(nyu_keys)
        #
        self.style = style # 'pytorch', 'caffe'
        self.transform = transforms.Compose([
                      transforms.ToPILImage(),
                      transforms.ToTensor(),  # Converts np_arr (H x W x C) in the range [0, 255]
                                              # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
                      transforms.Normalize(mean=[0.86067367, 0.86067367, 0.86067367],
                                           std =[0.22848366, 0.22848366, 0.22848366],)
                    ])
        #
        print ('---------------- Dataset_Base -------------------'   )
        print ('         collection : %s' % collection               )
        print ('         len(keys)  : %s' % len(self.keys)           )
        print ('           net_arch : %s' % net_arch                 )
        print ('          with_flip : %s' % with_flip                )
        print ('       sampling NYU : %5s%%' % (sampling['nyu']*100) )
        print ('       sampling Syn : %5s%%' % (sampling['syn']*100) )
        print ('--------------------------------------------------'  )


    def __len__(self):
        return len(self.keys)

    def _decide_flip(self):
        """[Warning] flip image only applies when doing azimuth estimation only."""
        _flip = self.with_flip # make a copy of self.flip
        if self.with_flip and self.collection in ['train']:
            # Only when with_flip is enable and it's on train set, do random flip.
            _flip = np.random.choice([True,False])
        return _flip

    def _is_nyu_data(self, idx):
        return idx<self._nyu_idx_range

    def __getitem__(self, idx):
        img_id = self.keys[idx]
        _flip = self._decide_flip()
        """ To implement construction of sample dictionary.
            To get image data: call 'self.roiloader(rcobj)'
        """
        print ('This is an interface method, and you need to implement it in inherited class.')
        raise NotImplementedError


class Dataset_reg_Sflat(Dataset_Base):
    def __getitem__(self, idx):
        img_id = self.keys[idx]
        _flip  = self._decide_flip()
        #=================================================================== Get Color Image
        if self._is_nyu_data(idx):
            img  = cv2.resize(self.nyu_dbImage[img_id], (320,240) )
        else:
            img  = self.syn_dbImage[img_id]            # already in shape of (320,240)
        img  = img if not _flip else cv2.flip(img, 1)  # vertically: 0 (x-axis), horizontally: 1 (y-axis), both: -1
        data = self.transform(img[:,:,::-1])           # first swap to RGB channel
        #=================================================================== Get Normal Image
        if self._is_nyu_data(idx):
            normImg = cv2.resize(self.nyu_dbNorm[img_id], (320,240), interpolation=cv2.INTER_NEAREST)
        else:
            normImg  = self.syn_dbNorm[img_id]         # already in shape of (320,240)
        normImg = normImg if not _flip else cv2.flip(normImg, 1)
        assert normImg.dtype in [np.uint8, np.uint16], normImg.dtype
        if normImg.dtype==np.uint8:
            norm = normImg.astype(np.float64)/(2**7) -1 # norm.astype(np.float64)/(2**8)*2-1  # map to [-1,1]
        else:
            norm = normImg.astype(np.float64)/(2**15)-1 # norm.astype(np.float64)/(2**16)*2-1  # map to [-1,1]
        norm[:,:,1][norm[:,:,1]>0] = 0                  # Note: x-z-y order;  make z>0 pixel as 0
        norm /= np.sqrt(np.power(norm, 2).sum(axis=2, keepdims=True))  # normalization to unit vec
        norm = norm if not _flip else norm*[1,1,-1]     # y-z-x  x should becomes -x
        norm = torch.from_numpy(norm)
        #=================================================================== Get Mask Image
        if self._is_nyu_data(idx):
            mask = cv2.resize(self.nyu_dbValid[img_id], (320,240), interpolation=cv2.INTER_NEAREST)
        else:
            mask = self.syn_dbValid[img_id]            # already in shape of (320,240)
        assert mask.dtype==np.uint8, mask.dtype
        mask = mask if not _flip else cv2.flip(mask, 1)
        mask = torch.from_numpy(mask)
        #
        return dict( idx   = idx,
                     data  = data,  # float32
                     norm  = norm,  # float64
                     mask  = mask,  # uint8
                     flip  = np.uint8(_flip),
                    )

    # interface method
    def _vis_minibatch(self, sample_batched):
        """Visualize a mini-batch for debugging."""
        for i, (idx, data, norm, mask, flip) in enumerate( zip( sample_batched['idx' ],  # note: these are tensors
                                                          sample_batched['data'],
                                                          sample_batched['norm'],
                                                          sample_batched['mask'],
                                                          sample_batched['flip']) ):
            self._vis_one(idx, data, norm, mask, flip)

    def _vis_one(self, idx, data, norm, mask, flip):
        data = data.numpy() if isinstance(data, torch.Tensor) else data
        norm = norm.numpy() if isinstance(norm, torch.Tensor) else norm
        mask = mask.numpy() if isinstance(mask, torch.Tensor) else mask
        flip = flip.numpy() if isinstance(flip, torch.Tensor) else flip
        #
        _norm = norm.reshape(-1,3)
        _mask = mask.astype(np.bool).reshape(-1)
        _norm = _norm[_mask]
        #
        if True:
            sampling = 0.01
            _inds = np.arange(len(_norm))
            sample_inds = np.random.choice(_inds, size=int(len(_inds)*sampling), replace=False)
            _nodes = _norm[sample_inds]
            nodes = np.zeros_like(_nodes)
            nodes[:,0],nodes[:,1],nodes[:,2] = _nodes[:,0],_nodes[:,2],_nodes[:,1]
            visualize3d(nodes)
        #
        im = data.transpose((1, 2, 0)).copy()
        im += caffe_mean_pxl # mean_pxl
        im = im.astype(np.uint8) # xmin, ymax
        #
        norm = ((norm+1)*(2**7)).astype(np.uint8)  # map from [-1,1] to [0,256]
        #
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        #
        showim = np.concatenate([im, norm, mask], axis=1)
        cv2_putText(showim, (0,20), "with_flip: %s " % flip.astype(np.bool))
        cv2.imshow('im',showim)
        cv2_wait()
        # pass


class Dataset_reg_Sexp(Dataset_reg_Sflat):
    def __getitem__(self, idx):
        img_id = self.keys[idx]
        _flip  = self._decide_flip()
        #=================================================================== Get Color Image
        if self._is_nyu_data(idx):
            img  = cv2.resize(self.nyu_dbImage[img_id], (320,240) )
        else:
            img  = self.syn_dbImage[img_id] # already in shape of (320,240)
        img  = img if not _flip else cv2.flip(img, 1)  # vertically: 0 (x-axis), horizontally: 1 (y-axis), both: -1
        data = self.transform(img[:,:,::-1])   # first swap to RGB channel
        #
        #=================================================================== Get Normal Image
        if self._is_nyu_data(idx):
            normImg = cv2.resize(self.nyu_dbNorm[img_id], (320,240), interpolation=cv2.INTER_NEAREST)
        else:
            normImg  = self.syn_dbNorm[img_id]          # already in shape of (320,240)
        normImg = normImg if not _flip else cv2.flip(normImg, 1)
        assert normImg.dtype in [np.uint8, np.uint16], normImg.dtype
        if normImg.dtype==np.uint8:
            norm = normImg.astype(np.float64)/(2**7) -1 # norm.astype(np.float64)/(2**8)*2-1  # map to [-1,1]
        else:
            norm = normImg.astype(np.float64)/(2**15)-1 # norm.astype(np.float64)/(2**16)*2-1  # map to [-1,1]
        norm[:,:,1][norm[:,:,1]>0] = 0                  # Note: x-z-y order;  make z>0 pixel as 0
        norm /= np.sqrt(np.power(norm, 2).sum(axis=2, keepdims=True))  # normalization to unit vec
        norm = norm if not _flip else norm*[1,1,-1]  # y-z-x  x should becomes -x
        #-------- rotate 45 trick (right hand rule) -------
        # x =  cost*x + sint*y
        # y = -sint*x + cost*y
        cost,sint = np.cos(np.pi/4), np.sin(np.pi/4)
        _norm = norm.copy()
        _y,_z,_x = _norm[:,:,0],_norm[:,:,1],_norm[:,:,2]
        y,z,x = norm[:,:,0],norm[:,:,1],norm[:,:,2]
        x[:] = cost*_x + sint*_y
        y[:] = -sint*_x + cost*_y
        #--------------------------------------------------
        norm = torch.from_numpy(norm)
        #=================================================================== Get Mask Image
        if self._is_nyu_data(idx):
            mask = cv2.resize(self.nyu_dbValid[img_id], (320,240), interpolation=cv2.INTER_NEAREST)
        else:
            mask = self.syn_dbValid[img_id] # already in shape of (320,240)
        assert mask.dtype==np.uint8, mask.dtype
        mask = mask if not _flip else cv2.flip(mask, 1)
        mask = torch.from_numpy(mask)
        #--------------------------------------------------------------------

        return dict( idx   = idx,
                     data  = data,  # float32
                     norm  = norm,  # float64
                     mask  = mask,  # uint8
                     flip  = np.uint8(_flip),
                    )


def visualize3d(node, mark_i=-1):
    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    import matplotlib.pyplot as plt

    # https://stackoverflow.com/questions/11140163/python-matplotlib-plotting-a-3d-cube-a-sphere-and-a-vector
    # draw a vector
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d
    #
    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)

    x,y,z = node[:,0], node[:,1], node[:,2]

    # Plot the surface
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    # draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]  # 0:np.pi/2:5j] #
    _x = np.cos(u)*np.sin(v) #  * 0.99
    _y = np.sin(u)*np.sin(v) #  * 0.99
    _z = np.cos(v)           #  * 0.99
    ax.plot_wireframe(_x, _y, _z, color="r",linewidth=0.2)
    # ax.plot_surface(_x,_y,_z, alpha=1.0) # ,  rstride=1, cstride=1, color='c', alpha=0.6, linewidth=0)
    #
    L = 1.4
    axis_x = Arrow3D([0, L], [0, 0], [0, 0], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
    axis_y = Arrow3D([0, 0], [0, L], [0, 0], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
    axis_z = Arrow3D([0, 0], [0, 0], [0, L], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
    ax.text(L, 0, 0, "x", color='red')
    ax.text(0, L, 0, "y", color='red')
    ax.text(0, 0, L, "z", color='red')
    ax.add_artist(axis_x)
    ax.add_artist(axis_y)
    ax.add_artist(axis_z)
    #
    # ax.scatter(x,y,z, c='b', marker='o', depthshade=True)
    #
    assert mark_i<len(node)
    if mark_i>=0:
        inds = np.arange(len(node))
        ax.scatter(x[inds==mark_i],y[inds==mark_i],z[inds==mark_i], c='r', marker='^', depthshade=False, linewidth=2  )
        ax.scatter(x[inds!=mark_i],y[inds!=mark_i],z[inds!=mark_i], c='b', marker='o', depthshade=True , linewidth=0.5)
    else:
        ax.scatter(x,y,z, c='b', marker='o', depthshade=True, linewidth=0.5)
    # Hide grid lines
    ax.grid(False)
    plt.axis('off')

    # # the histogram of the data
    # fig, ax = plt.subplots()
    # n, bins, patches = ax.hist(e, 10, density=1)
    fig.tight_layout()
    # if mark_i>=0:
    #     fn = 'node%s/%s.jpg' % (len(node), mark_i)
    #     mkdir4file(fn)
    #     plt.savefig(fn )
    plt.show()


if __name__ == '__main__':
    def test_dataloader():
        dataset = Dataset_reg_Sexp(collection='train', style='caffe', with_flip=False, sampling=dict(nyu=1.0, syn=0.0), Ladicky_normal=True)

        for idx in range(len(dataset)):
            one = dataset[idx]
            idx, data, norm, mask, flip = one['idx'], one['data'], one['norm'], one['mask'], one['flip']
            dataset._vis_one(idx, data, norm, mask, flip)

    test_dataloader()

