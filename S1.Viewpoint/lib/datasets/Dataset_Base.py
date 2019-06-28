"""
 @Author  : Shuai Liao
"""

import os, sys
import cv2
import pickle
import numpy as np
from math import ceil, floor, pi

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
#
from basic.common import add_path, env, rdict, cv2_wait, cv2_putText
from lmdb_util import ImageData_lmdb
from numpy_db  import npy_table, npy_db, dtype_summary, reorder_dtype


print('[Dataset_Base]:  [TODO] get rid of the relative paths ../../')

this_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.realpath(this_dir+'/../../..')
add_path(root_dir+'/dataset')
from Pascal3D import categories, get_anno_dbs_tbl, get_anno_db_tbl


__all__ = ['Dataset_Base', 'netcfg']

#
syn_data_path = root_dir+'/dataset/Pascal3D/SynImages_r4cnn.cache'
syn_imdb_path = os.path.join(syn_data_path, 'ImageData_gtbox_crop.Rawpng.lmdb')
syn_anno_path = os.path.join(syn_data_path, 'obj_anno_db/{cate}.pkl')

pascal3d_imdb_path   = root_dir+'/dataset/Pascal3D/ImageData.Rawjpg.lmdb'
pascal3d_gt_box_path = root_dir+'/dataset/Pascal3D/anno_db_v2/data.cache/objId2gtbox/cate12_{collection}.all.pkl'

collection2filter=rdict(train='all', val='easy')

## Net configurations that are independent of task
netcfg = rdict( # configuration for alexnet
                alexnet=rdict(  TRAIN=rdict(BATCH_SIZE=200),
                                TEST =rdict(BATCH_SIZE=200),
                                INPUT_SHAPE=(227, 227),
                                ),  # ignore_label=-1,
                # configuration for vgg
                vgg16  =rdict(  TRAIN=rdict(BATCH_SIZE=50), #64 20
                                TEST =rdict(BATCH_SIZE=20),
                                INPUT_SHAPE=(224, 224),
                                ),
                # configuration for resnet50
                resnet50 =rdict(TRAIN=rdict(BATCH_SIZE=100), # 128
                                TEST =rdict(BATCH_SIZE=64),
                                INPUT_SHAPE=(224, 224),
                                ),
                # configuration for resnet101
                resnet101=rdict(TRAIN=rdict(BATCH_SIZE=64),
                                TEST =rdict(BATCH_SIZE=20),
                                INPUT_SHAPE=(224, 224),
                                ),
                # configuration for resnet152
                resnet152=rdict(TRAIN=rdict(BATCH_SIZE=32),
                                TEST =rdict(BATCH_SIZE=10),
                                INPUT_SHAPE=(224, 224),
                               ),
               )



def get_keys_recs_v1(cates, collection='train', filter='all', sampling=dict(pascalvoc=1.0, imagenet=0.5, synthetic=0.2), img_scale='Org'):
    ''' Note: this function provided strategy of sampling the amount training data.
    In general, there're 3 party of data to use:  1) pascalvoc  2) imagenet and 3) synthetic data.
    The first 2 parts are from pascal3d+ and synthetic data is created by render4cnn.
    The order to use the these data in principle is pascalvoc <- imagenet <- synthetic data.
    '''
    np.random.seed(0)

    # Load Pascal3D annotation.
    obj_tb_pascal3d = get_anno_dbs_tbl(cates, collection=collection, filter=filter, img_scale=img_scale, withCoarseVp=True)
    if collection=='train':
        inds = np.arange(len(obj_tb_pascal3d.keys))
        # Perform sampling if needed.
        inds_pascalvoc, inds_imagenet = [], []
        for i in inds:
            if obj_tb_pascal3d.keys[i].startswith('n'):
                inds_imagenet.append(i)
            else:
                inds_pascalvoc.append(i)
        # sample pascalvoc if needed.
        if sampling['pascalvoc']<1.0:
            nr_sample = int(len(inds_pascalvoc)*sampling['pascalvoc'])
            inds_pascalvoc = np.random.choice(inds_pascalvoc, nr_sample, replace=False)
        # sample imagenet if needed.
        if sampling['imagenet']<1.0:
            nr_sample = int(len(inds_imagenet)*sampling['imagenet'])
            inds_imagenet = np.random.choice(inds_imagenet, nr_sample, replace=False)
        #
        recs_pascalvoc = obj_tb_pascal3d.recs[inds_pascalvoc]
        recs_imagenet  = obj_tb_pascal3d.recs[inds_imagenet ]
        recs_Pascal3D  = np.concatenate([recs_pascalvoc, recs_imagenet], axis=0)
        obj_tb_pascal3d = npy_table(recs_Pascal3D)

        # Load rend4cnn.SynImages annoation.
        list_obj_tb_synimage = []
        if sampling['synthetic']>0: # with_SynImage:
            # we keep collection here just to keep same format as it's in 'Pascal3D_Base.py'
            assert collection=='train', 'Only train collection suppoted "with_SynImage".'
            for cate in cates:
                anno_file = syn_anno_path.format(cate=cate)
                # load npy_db
                db = npy_db()
                db.load(anno_file)
                obj_tb_synimage = db['obj_tb']
                inds_syn = np.arange(len(obj_tb_synimage.keys))
                nr_sample = int(len(inds_syn)*sampling['synthetic'])
                inds_syn = np.random.choice(inds_syn, nr_sample, replace=False)
                obj_tb_synimage = npy_table(obj_tb_synimage.recs[inds_syn])
                list_obj_tb_synimage.append(obj_tb_synimage)

        # Merge Pascal3D and rend4cnn.SynImages
        obj_tbs = []
        obj_tb = npy_table.merge( [obj_tb_pascal3d]+list_obj_tb_synimage )
        keys_recs = obj_tb.keys, obj_tb.recs.view(np.recarray)
    else:
        keys_recs = obj_tb_pascal3d.keys, obj_tb_pascal3d.recs

    return keys_recs


class _RoiLoader_Base(object):
    def __init__(self, collection, rsz_shape, mode='torchmodel'):  # with_flip=False,
        """ The loader assume ROI is already cropped.

            # with_flip: only affect train,dev collection.
        """
        self.rsz_shape = rsz_shape
        self.mode      = mode
        #
        if self.mode=='torchmodel':
            self.pxl_mean, self.pxl_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        elif self.mode=='caffemodel':
            self.pxl_mean, self.pxl_std = [102.9801, 115.9465, 122.7717], [1.0, 1.0, 1.0] # only subtract mean
        else:
            print("Unknown mode: %s  (should be torchmodel or caffemodel)" % self.mode)
            raise NotImplementedError

        # self.with_flip = with_flip if collection in ['train','dev'] else False
        #
        transforms_Identity = transforms.Lambda(lambda x: x)
        transforms_x255     = transforms.Lambda(lambda x: x*255.)
        # if   collection in ['train','dev']:
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),            # Converts a Tensor of shape C x H x W   or   a numpy ndarray of shape H x W x C to a PIL Image
            transforms.Resize(self.rsz_shape),
            transforms.ToTensor(),  #  always return [0,1] ranged pixel data.
            transforms_x255 if self.mode=='caffemodel' else transforms_Identity,  # for training only
            transforms.Normalize(mean=self.pxl_mean, std=self.pxl_std)
        ])


class _RoiLoader_OnTheFly_SynImage(_RoiLoader_Base):
    """ Do cropping on the fly"""
    def __init__(self, collection, cates, rsz_shape, mode='torchmodel'):
        super().__init__(collection, rsz_shape, mode=mode) # python3 only
        # LMDB for original image in jpg format. (rescaled to max side 500).
        db_path = syn_imdb_path # _get_local_db_path(syn_db_path)
        assert db_path is not None, '%s  is not exist.' % (syn_imdb_path)

        self.datadb   = ImageData_lmdb(db_path, 'r')

    def __call__(self, rcobj, flip=False):
        obj_id   = rcobj.obj_id
        image_id = rcobj.src_img.image_id

        real_img_id = (obj_id+image_id)[:-4]  # [:-4] to remove '.Syn'
        roi_img = self.datadb[real_img_id]
        if flip: # Horizontally flip image.
            roi_img =  cv2.flip(roi_img, 1) # vertically: 0 (x-axis), horizontally: 1 (y-axis), both: -1

        if self.mode=='torchmodel':
            roi_img = roi_img[:,:,::-1]  # BGR --> RGB
        return self.transforms(self.datadb[real_img_id])



class _RoiLoader_OnTheFly_Pascal3D(_RoiLoader_Base):
    """ Do cropping on the fly"""
    def __init__(self, collection, cates, rsz_shape, with_aug=False, context_pad=16, img_scale='Org',mode='torchmodel'):
        super().__init__(collection, rsz_shape, mode=mode) # python3 only

        db_path = pascal3d_imdb_path # _get_local_db_path(_db_path)
        assert db_path is not None, '%s  is not exist.' % (db_path)
        self.datadb   = ImageData_lmdb(db_path)
        #
        self.with_aug = with_aug

        if self.with_aug:
            # Pre-computed augmentation box
            raise NotImplementedError  #@Shuai: (View Estimation on GT doesn't use box augmentation).
        else:  # use gt box
            # Pre-computed (verified) clamped gt box
            gtbox_path = pascal3d_gt_box_path.format(collection=collection)
            self.objId2gtbox = pickle.load( open(gtbox_path, 'rb') )

        self.context_scale = float(rsz_shape[0])/(rsz_shape[0] - 2*context_pad)

    def add_context(self, boxes): #, crop_size=reshape_size, context_pad=16):
        """boxes is np.ndarray"""
        if self.context_scale==1.0:
            return boxes
        _boxes = boxes.astype(np.float32).copy()
        x1,y1,x2,y2 = _boxes # _boxes[:,0], _boxes[:,1], _boxes[:,2], _boxes[:,3]
        # h, w = imgDims
        # context_scale = float(crop_size)/(crop_size - 2*context_pad)
        # compute the expanded region
        half_height= (y2-y1+1)/2.0
        half_width = (x2-x1+1)/2.0
        center_x   = (x1) + half_width
        center_y   = (y1) + half_height

        x1 = np.round(center_x - half_width *self.context_scale)
        x2 = np.round(center_x + half_width *self.context_scale)
        y1 = np.round(center_y - half_height*self.context_scale)
        y2 = np.round(center_y + half_height*self.context_scale)

        # Warning:
        # the expanded region may go outside of the image.
        # do clipping afterwards.
        return _boxes.astype(np.int32)

    def crop_roi(self, img, bbox, flip=False):
        # Clamp bbox
        h, w, _ = img.shape
        x1,y1,x2,y2 = self.add_context(bbox)
        x1 = max(x1, 0  )
        y1 = max(y1, 0  )
        x2 = min(x2, w-1)
        y2 = min(y2, h-1)
        if x1>=x2 and y1>=y2:
            print ('[bad box] ' + "h,w=%s,%s   %s  %s" % (h,w,  '(%s,%s,%s,%s)' % tuple(bbox), '(%s,%s,%s,%s)'%(x1,y1,x2,y2) ))
            return torch.Tensor((3,*self.rsz_shape),torch.float32)
        else:
            # Crop
            roi_img = img[y1:y2,x1:x2]
            if flip: # Horizontally flip image.
                roi_img =  cv2.flip(roi_img, 1) # vertically: 0 (x-axis), horizontally: 1 (y-axis), both: -1

            if self.mode=='torchmodel':
                roi_img = roi_img[:,:,::-1]  # BGR --> RGB
            return self.transforms(roi_img)

    def __call__(self, rcobj, flip=False):
        obj_id   = rcobj.obj_id
        image_id = rcobj.src_img.image_id

        if self.with_aug:
            aug_box_ind = np.random.randint(len(self.objId2augboxes[obj_id]))
            aug_box = self.objId2augboxes[obj_id][aug_box_ind]
            return self.crop_roi(self.datadb[image_id], aug_box, flip)
        else:
            gt_bbox = self.objId2gtbox[obj_id]
            return self.crop_roi(self.datadb[image_id], gt_bbox, flip)









class Dataset_Base(Dataset):
    def __init__(self, collection='train', cates=None, net_arch='alexnet', with_aug=False, with_flip=False, mode='torchmodel',
                 sampling=dict(pascalvoc=1.0, imagenet=1.0, synthetic=0.0) ):
        self.cfg        = netcfg[net_arch]
        self.collection = collection
        self.cates      = categories if cates is None else cates
        self.cate2ind   = dict(zip(self.cates, range(len(self.cates))))
        self.keys, self.recs = get_keys_recs_v1(self.cates, collection, filter=collection2filter[collection], sampling=sampling)
        self.mode       = mode

        self.roiloader_pascal3d = _RoiLoader_OnTheFly_Pascal3D(collection, self.cates, self.cfg.INPUT_SHAPE,
                                                               with_aug=with_aug, mode=mode)
        if sampling['synthetic']>0: #with_SynImage:
            self.roiloader_synimage = _RoiLoader_OnTheFly_SynImage(collection, self.cates, self.cfg.INPUT_SHAPE,
                                                                   mode=mode)

        #========================================
        # To turn on/off flip mode (for image and anno. az)
        self.set_flipmode(with_flip)
        #=========================================
        print ('---------------- Dataset_Base -------------------'         )
        print ('         collection : %s' % collection                     )
        print ('         len(cates) : %s' % len(self.cates)                )
        print ('           net_arch : %s' % net_arch                       )
        print ('           with_aug : %s' % with_aug                       )
        print ('          with_flip : %s' % with_flip                      )
        print (' sampling pascalvoc : %5s%%' % (sampling['pascalvoc']*100) )
        print (' sampling imagenet  : %5s%%' % (sampling['imagenet' ]*100) )
        print (' sampling synthetic : %5s%%' % (sampling['synthetic']*100) )
        print ('--------------------------------------------------'        )

    def set_flipmode(self, with_flip):
        self.with_flip = with_flip

    def _decide_flip(self):
        """[Warning] flip image should be applied only for azimuth estimation only."""
        _flip = self.with_flip # make a copy of self.flip
        if self.with_flip and self.collection in ['train', 'dev']:
            # Only when with_flip is enable and it's on train set, do random flip.
            _flip = np.random.choice([True,False]) # ,p=[0.7,0.3] probabilities associated with each entry.
        return _flip

    def _get_image(self, rcobj, flip=False):
        image_id = rcobj.src_img.image_id
        if image_id.endswith('.Syn'):
            img = self.roiloader_synimage(rcobj, flip)
        else:
            img = self.roiloader_pascal3d(rcobj, flip)
        return img # if not flip else cv2.flip( img, 1 )

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, idx):
        rcobj    = self.recs[idx]
        cate     = rcobj.category
        obj_id   = rcobj.obj_id
        image_id = rcobj.src_img.image_id

        # In degree
        a = rcobj.gt_view.a # * np.pi/180.
        e = rcobj.gt_view.e # * np.pi/180.
        t = rcobj.gt_view.t # * np.pi/180.
        #
        _flip = self._decide_flip()
        if _flip:
            a = 360-a
        data  = self._get_image(rcobj, flip=_flip)

        """ To implement construction of sample dictionary.
            To get image data: call 'self.roiloader(rcobj)'
        """
        print ('This is an interface method, and you need to implement it in inherited class.')
        raise NotImplementedError






#---------------------------------------------------
class Dataset_Example(Dataset_Base):
    def __getitem__(self, idx):
        rcobj    = self.recs[idx]
        cate     = rcobj.category
        obj_id   = rcobj.obj_id
        image_id = rcobj.src_img.image_id

        # In degree
        a = rcobj.gt_view.a # * np.pi/180.
        e = rcobj.gt_view.e # * np.pi/180.
        t = rcobj.gt_view.t # * np.pi/180.
        # _flip = self._decide_flip()
        # if _flip:
        #     a = 360-a
        # # Re-Mapping a: [0,180]->[180,360]->[0,180]    [180,360]->[0,180]->[-180,0]
        # a = (a+180.) % 360. - 180.
        # # To radius
        # a = (a * np.pi/180.).astype(np.float32)
        # e = (e * np.pi/180.).astype(np.float32)
        # t = (t * np.pi/180.).astype(np.float32)

        sample = dict( idx   = idx,
                       label = self.cate2ind[cate],
                       a     = a,
                       e     = e,
                       t     = t,
                       data  = self._get_image(rcobj) )  #sub_trs(self.datadb[obj_id]) )
        return sample


    # interface method
    def _vis_minibatch(self, sample_batched):
        """Visualize a mini-batch for debugging."""
        for i, (idx, label, a,e,t, data) in enumerate( zip(sample_batched['idx'],  # note: these are tensors
                                                           sample_batched['label'],
                                                           sample_batched['a'],
                                                           sample_batched['e'],
                                                           sample_batched['t'],
                                                           sample_batched['data']) ):

            rcobj = self.recs[idx]
            print (idx, rcobj.obj_id)
            im = data.numpy().transpose((1, 2, 0)).copy()
            im = (im*self.roiloader_pascal3d.pxl_std)+self.roiloader_pascal3d.pxl_mean
            if self.mode=='torchmodel':
                im = (im*255)[:,:,::-1].astype(np.uint8) # RBG->BGR
            else: # caffemodel type
                im = im.astype(np.uint8)
            text = '%s %.1f' %(rcobj.category,a)
            cv2_putText(im, (0,20), text, bgcolor=(255,255,255)) #
            text = ' a=%.1f,e=%.1f,t=%.1f' % (rcobj.gt_view.a,rcobj.gt_view.e,rcobj.gt_view.t)
            cv2_putText(im, (0,40), text, bgcolor=(255,255,255)) #
            cv2.imshow('im',im)
            cv2_wait()
            # pass


if __name__ == '__main__':
    def test_dataloader():
        dataset = Dataset_Example(collection='val', with_aug=False, with_flip=False, mode='caffemodel',
                                  sampling=dict(pascalvoc=1.0, imagenet=1.0, synthetic=0.0))
        dataloader = DataLoader(dataset, batch_size=4,
                                shuffle=False, num_workers=1)

        for i_batch, sample_batched in enumerate(dataloader):
            dataset._vis_minibatch(sample_batched)


    test_dataloader()

