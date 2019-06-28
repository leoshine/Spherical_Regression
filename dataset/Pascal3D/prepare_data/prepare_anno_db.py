import os, sys
import pickle
from easydict import EasyDict as edict
from collections import OrderedDict as odict
import cv2
import numpy as np

from basic.common import env, Open, add_path
from basic.util import load_yaml
from numpy_db import npy_table, npy_db,  dtype_summary, reorder_dtype

add_path('../anno_db_v2')
from db_type  import viewpoint, proj_info, image_info, object_anno  # , pose_hypo
from util_v2 import rescale_anno

conf = load_yaml('config.yml')  # odict
Pascal3D_root = os.path.expanduser(conf['Pascal3D_release_root'])
# Pascal3D_root  = env.Home+'/working/cvpr17pose/dataset/PASCAL3D+_release1.1'  # Image_sets/%s_imagenet_%s.txt
PascalVOC_root = Pascal3D_root + '/PASCAL'
assert os.path.exists(Pascal3D_root) , "Please replace with your path/to/PASCAL3D+_release1.1"
assert os.path.exists(PascalVOC_root), "Cannot find %s" % PascalVOC_root


protocol      = conf['pkl_protocol']  # pickle dump protocol. Change -1 to 2 for python2.x compatibility.



if True:
    print ('\n\n---------   viewpoint   --------')
    dtype_summary(np.dtype(viewpoint))
    print ('\n\n---------   proj_info   --------')
    dtype_summary(np.dtype(proj_info))
    print ('\n\n---------   image_info  --------')
    dtype_summary(np.dtype(image_info))
    print ('\n\n---------   object_anno --------')
    dtype_summary(np.dtype(object_anno))


# dataset = 'PASCAL3D'
basedir = '../' # env.Home + '/working/cvpr17pose/dataset/'+dataset
py_anno_dir = os.path.join('./working_dump.cache', 'Imgwise_Annotations.py')

categories = [x.strip() for x in open(os.path.join(basedir,'ImageSets','categories.txt')).readlines()]  # 'PASCAL3D.txt')).readlines()]
print (categories)


# MAX_IMAGE_SIDE = 256
# MAX_IMAGE_SIDE = 500

def read_anno_pkl(anno_filename, cate, source, image_id, img_set, opts=None, include_coarse_vp_anno=False, MAX_IMAGE_SIDE=None):
    anno = pickle.load(open(anno_filename, 'rb'))
    anno = edict(anno)
    objects = anno.record.objects

    selected_objs = []
    if not isinstance(objects,list):
        objects = [objects]  # convert single object anno as list

    for k, obj in enumerate(objects):
        # print obj.viewpoint # ~isempty(obj.viewpoint)
        if obj['class'] == cate:
            # write view annotation
            # print obj.viewpoint.azimuth, obj.viewpoint.elevation, obj.viewpoint.theta
            # Notice: a, e, t are already within [0,360], [-180,180], [-180,180] range.

            truncated = obj.truncated
            occluded  = obj.occluded
            difficult = obj.difficult

            # skip un-annotated image
            vp = obj.viewpoint

            """
            This part of code is just for checking:
            #-----------------------------------------------------------------------------------------------#
            # [Conclusion] vp.distance==0 is equivalent to this annotation is coarse viewpoint annotation.  #
            #-----------------------------------------------------------------------------------------------#

            # Check [1]: Whenever   vp.distance==0,
            #            this holds: (vp.azimuth==0 and vp.elevation==0 and vp.theta==0)
            is_coarse_vp_anno = (vp.distance==0)
            if is_coarse_vp_anno:
                assert (vp.azimuth==0 and vp.elevation==0 and vp.theta==0)

            # Check [2]: Whenever   (vp.azimuth==0 and vp.elevation==0 and vp.theta==0),
            #            this holds: (vp.distance==0)
            is_coarse_vp_anno = (vp.azimuth==0 and vp.elevation==0 and vp.theta==0)
            if is_coarse_vp_anno:
                assert (vp.distance==0)
            """

            is_coarse_vp_anno = (vp.distance==0)
            if is_coarse_vp_anno:
                assert (vp.azimuth==0 and vp.elevation==0 and vp.theta==0)  # just for checking.

            if is_coarse_vp_anno:
                # print 'skip %s...' % image_id
                # assert vp.azimuth_coarse!=0 or vp.elevation_coarse!=0, anno_filename
                # print '[only coarse anno. %s  ]   %s   %s' %(anno_filename.split('/')[-1][:-4], vp.azimuth_coarse, vp.elevation_coarse)
                if not include_coarse_vp_anno:
                    continue
                else:
                    # use coarse annotation of azimuth and elevation as replacement.
                    vp.azimuth   = vp.azimuth_coarse
                    vp.elevation = vp.elevation_coarse


            # skip unwanted image
            if opts is not None:
                if (difficult==1 and opts.difficult==0) or \
                   (truncated==1 and opts.truncated==0) or \
                   (occluded ==1 and opts.occluded ==0):
                    # print 'fliter skip %s...' % image_id
                    continue

            bboxStr = ",".join( map(str, map(int,obj['bbox']) ) )
            cadId = "%s%02d" %(cate, obj.cad_index)

            #======================================
            #--[new object record]
            rcobj_1arr = np.zeros( (1,), dtype=object_anno )
            rcobj = rcobj_1arr[0].view(np.recarray) # cast as recarray
            #
            rcobj.obj_id    = "{}-{}".format(image_id, bboxStr)
            rcobj.category  = obj['class'] # cate
            rcobj.cad_id    = "%s%02d" % (obj['class'], obj.cad_index)
            rcobj.bbox[:] = obj.bbox
            #-----[source image]
            rcobj.src_img.image_id = image_id
            rcobj.src_img.H        = anno.record.size.height  # anno.record['imgsize'][1] # Original image size
            rcobj.src_img.W        = anno.record.size.width   # anno.record['imgsize'][0]
            rcobj.src_img.C        = anno.record.size.depth   # anno.record['imgsize'][2]
            rcobj.src_img.h        = rcobj.src_img.H  # The size of image this annotation based on,
            rcobj.src_img.w        = rcobj.src_img.W  # it can be different from original image if rescaled by MAX_IMAGE_SIDE.
            #-----[viewpoint]
            rcobj.gt_view.a  = obj.viewpoint.azimuth
            rcobj.gt_view.e  = obj.viewpoint.elevation
            rcobj.gt_view.t  = obj.viewpoint.theta
            rcobj.gt_view.d  = obj.viewpoint.distance  # Note: if distance=0, this means a,e field is coarse vp annotation.
            rcobj.gt_view.px = obj.viewpoint.px
            rcobj.gt_view.py = obj.viewpoint.py
            rcobj.gt_view.f  = obj.viewpoint.focal
            rcobj.gt_view.mx = obj.viewpoint.viewport
            rcobj.gt_view.my = obj.viewpoint.viewport
            # rcobj.gt_view.is_coarse_anno = is_coarse_vp_anno
            ''' Note: is_coarse_vp_anno
                We notice that, whenever only coarse is available, distance=0
            '''
            #--other annotation.
            rcobj.difficult = obj.difficult
            rcobj.truncated = obj.truncated
            rcobj.occluded  = obj.occluded
            #======================================

            H, W = int(rcobj.src_img.H), int(rcobj.src_img.W)
            ## if max(H, W)>MAX_IMAGE_SIDE or max(H, W)<MAX_IMAGE_SIDE:
            # No rescaling here. (this code generate annotation from original image size.)
            if (MAX_IMAGE_SIDE is not None)  and (max(H, W) != MAX_IMAGE_SIDE):
                rescale_ratio = float(MAX_IMAGE_SIDE) / max(H, W)
                rescale_anno(rcobj, None, fx=rescale_ratio, fy=rescale_ratio )

            '''
            # record in matlab:
            'viewpoint': {'distance': 5.786937334541722, 'py': 144.33914314685566, 'elevation': -38.11983344583065, 'px': 280.2637711691764, 'elevation_coarse': -32.5, 'interval_azimuth': 0.03878493891316417, 'error': 150.99527810530716, 'num_anchor': 8, 'azimuth': 72.49999985390725, 'viewport': 3000, 'interval_elevation': 0.03348538986576915, 'azimuth_coarse': 50, 'theta': -8.663952905955943, 'focal': 1},
            '''

            selected_objs.append(rcobj)
            # if rcobj.proj.valid:
            #     selected_objs.append(rcobj)
            # else:
            #     # ... log to file ....
            #     pass

    return selected_objs




def filter_img_set(img_set, filter='easy', source='pascal', include_coarse_vp_anno=False, MAX_IMAGE_SIDE=None):
    print ('=================[%s_%s] %s' % (source, img_set, filter))
    # {'flip':1, 'aug_n':1,'jitter_IoU':1,'difficult':1,'truncated':1,'occluded':1}
    if   filter=='all':
        opts = None
    elif filter=='easy':                                          # Select option opts: Mark 1 as to be kept, 0: to be filtered.
        opts = edict({'difficult':0,'truncated':0,'occluded':0})  # filter out truncated, occluded but keep difficult
    elif filter=='nonOccl':
        opts = edict({'difficult':1,'truncated':0,'occluded':0})  # filter out truncated, occluded but keep difficult
    elif filter=='nonDiff':
        opts = edict({'difficult':0,'truncated':1,'occluded':1})  # filter out difficult but keep truncated, occluded
    else:
        raise NotImplementedError

    cnt_image = 0
    for cate in categories:
        obj_list = []

        if source=='pascal':
            filename = PascalVOC_root + '/VOCdevkit/VOC2012/ImageSets/Main/%s.txt' % img_set
            all_ids = list(map(str.strip, open(filename).readlines()))
        elif source=='imagenet':
            # /Users/shine/QuvaMnt/working/cvpr17pose/dataset/PASCAL3D+_release1.1/Image_sets/aeroplane_imagenet_val.txt
            filename = Pascal3D_root  + '/Image_sets/%s_imagenet_%s.txt' % (cate, img_set)
            all_ids = list(map(str.strip, open(filename).readlines()))
        else:
            raise NotImplementedError

        cnt_obj = 0
        for i,id in enumerate(all_ids):
            anno_filename = os.path.join(py_anno_dir, '%s_%s/%s.pkl' %(cate, source, id))  # aeroplane_pascal/xxx.pkl
            if not os.path.exists(anno_filename):
                # print 'File not exists: ', anno_filename
                continue

            selected_objs = read_anno_pkl(anno_filename, cate, source, id, img_set, opts, include_coarse_vp_anno, MAX_IMAGE_SIDE)
            if len(selected_objs)>0:
                cnt_obj += len(selected_objs)
                obj_list += selected_objs
            # else:
            #     print anno_filename

        # print "-------------------------------------   len(obj_list): ", len(obj_list)
        obj_rcs = np.vstack(obj_list).reshape((-1,))       # np.concatenate(obj_list, axis=0) # np.vstack(obj_list)
        obj_rcs = obj_rcs.view(np.recarray) # cast as np.recarray
        img_ids = set(obj_rcs.src_img.image_id.tolist())
        cnt_image += len(img_ids)
        print ('%-20s img %5s   obj  %5s' % (cate, len(img_ids), len(obj_rcs)))

        # print '%-20s img %5s  obj  %5s' % (cate, len(set(obj_rcs.src_img.image_id.tolist())), len(obj_rcs))
        filterStr = filter+'_withCoarseVp' if include_coarse_vp_anno else filter
        maxSideStr = 'Org' if  (MAX_IMAGE_SIDE is None)  else 'Max%s' % MAX_IMAGE_SIDE
        pickle.dump(obj_rcs, Open('working_dump.cache/Catewise_obj_anno/%s.%s/%s_%s.%s.pkl' % (filterStr, maxSideStr, source, img_set, cate) ,'wb'), protocol)

    print ('selected:  %d / %d  images' % (cnt_image, len(all_ids)))



def build_db(filter='easy', include_coarse_vp_anno=False, MAX_IMAGE_SIDE=None):
    _args = dict(filter=filter,
                 include_coarse_vp_anno=include_coarse_vp_anno,
                 MAX_IMAGE_SIDE=MAX_IMAGE_SIDE )
    filter_img_set('train', source='pascal', **_args)
    filter_img_set('val'  , source='pascal', **_args)
    #
    filter_img_set('train', source='imagenet', **_args)
    filter_img_set('val'  , source='imagenet', **_args)

    #filter_img_set('train', filter='hard',source='pascal')
    #filter_img_set('val',   filter='hard',source='pascal')
    #
    #filter_img_set('train', filter='hard', source='imagenet')
    #filter_img_set('val',   filter='hard', source='imagenet')

    # Merge easy
    # try:
    #     os.makedirs('pascal3d_easy.cache/train')
    #     os.makedirs('pascal3d_easy.cache/val')
    # except: pass



    # val collection.
    print ('\n\n\n>>>>=================[pascal3d_%s] val' % filter)
    for cate in categories:
        # pascal_val.{cate}.easy.pkl
        # e.g. pascal_val.bottle.easy.pkl
        filterStr = filter+'_withCoarseVp' if include_coarse_vp_anno else filter
        maxSideStr = 'Org' if  (MAX_IMAGE_SIDE is None)  else 'Max%s' % MAX_IMAGE_SIDE
        scr = 'working_dump.cache/Catewise_obj_anno/{}.{}/pascal_val.{}.pkl'.format(filterStr, maxSideStr, cate)
        des = '../anno_db_v2/data.cache/{}.{}/val/{}.pkl'.format(filterStr, maxSideStr, cate)

        obj_rcs = pickle.load(open(scr, 'rb'))
        img_ids = set(obj_rcs.src_img.image_id.tolist())
        print ('%-20s img %5s  obj  %5s' % (cate, len(img_ids), len(obj_rcs)))

        # create npy_table and npy_db, and dumpy it.
        obj_tb = npy_table(obj_rcs)
        db = npy_db()
        db.add_table(obj_tb, name="obj_tb")
        db.dump(des)


    # train collection.
    print ('\n\n\n>>>>=================[pascal3d_%s] train' % filter)
    for cate in categories:
        # pascal_val.{cate}.easy.pkl
        # e.g. pascal_val.bottle.easy.pkl
        filterStr = filter+'_withCoarseVp' if include_coarse_vp_anno else filter
        maxSideStr = 'Org' if  (MAX_IMAGE_SIDE is None)  else 'Max%s' % MAX_IMAGE_SIDE
        scrs = ['working_dump.cache/Catewise_obj_anno/{}.{}/pascal_train.{}.pkl'.format(filterStr, maxSideStr, cate),
                'working_dump.cache/Catewise_obj_anno/{}.{}/imagenet_train.{}.pkl'.format(filterStr, maxSideStr, cate),
                'working_dump.cache/Catewise_obj_anno/{}.{}/imagenet_val.{}.pkl'.format(filterStr, maxSideStr, cate)]
        des   = '../anno_db_v2/data.cache/{}.{}/train/{}.pkl'.format(filterStr, maxSideStr, cate)
        obj_list = []
        for scr in scrs:
            _obj_rcs = pickle.load(open(scr, 'rb'))
            obj_list.append(_obj_rcs)

        obj_rcs = np.concatenate(obj_list, axis=0) # np.vstack(obj_list)
        obj_rcs = obj_rcs.view(np.recarray) # cast as np.recarray

        img_ids = set(obj_rcs.src_img.image_id.tolist())
        print ('%-20s img %5s  obj  %5s' % (cate, len(img_ids), len(obj_rcs)))

        # create npy_table and npy_db, and dumpy it.
        obj_tb = npy_table(obj_rcs)
        db = npy_db()
        db.add_table(obj_tb, name="obj_tb")
        db.dump(des)

if __name__ == '__main__':

    build_db(filter='easy'   , include_coarse_vp_anno=True, MAX_IMAGE_SIDE=None)
    build_db(filter='all'    , include_coarse_vp_anno=True, MAX_IMAGE_SIDE=None)
    # build_db(filter='nonOccl', include_coarse_vp_anno=True, MAX_IMAGE_SIDE=None)
    # build_db(filter='nonDiff', include_coarse_vp_anno=True, MAX_IMAGE_SIDE=None)


    # build_db(filter='easy'   , include_coarse_vp_anno=True, MAX_IMAGE_SIDE=500)
    # build_db(filter='all'    , include_coarse_vp_anno=True, MAX_IMAGE_SIDE=500)
    # build_db(filter='nonOccl', include_coarse_vp_anno=True, MAX_IMAGE_SIDE=500)
    # build_db(filter='nonDiff', include_coarse_vp_anno=True, MAX_IMAGE_SIDE=500)


