"""
 @Author  : Shuai Liao
"""

from basic.common import env, add_path, cv2_wait, Open
add_path('../../')
from Pascal3D import get_imgIDs, get_anno, categories
import os, sys
import cv2
import pickle
from tqdm import tqdm

from lmdb_util import ImageData_lmdb

from basic.util import load_yaml
conf = load_yaml('config.yml')  # odict
Pascal3D_root = os.path.expanduser(conf['Pascal3D_release_root'])
src_img_dir   = os.path.expanduser(conf['Synthetic_Images_root'])

protocol      = conf['pkl_protocol']  # pickle dump protocol. Change -1 to 2 for python2.x compatibility.


def process(imgID, imgpath, datadb, visualize=False):
    datadb.put(imgID, imgpath)


cate_synsetID = [ ( 'aeroplane'   , '02691156' ) ,
                  ( 'bicycle'     , '02834778' ) ,
                  ( 'boat'        , '02858304' ) ,
                  ( 'bottle'      , '02876657' ) ,
                  ( 'bus'         , '02924116' ) ,
                  ( 'car'         , '02958343' ) ,
                  ( 'chair'       , '03001627' ) ,
                  ( 'diningtable' , '04379243' ) ,
                  ( 'motorbike'   , '03790512' ) ,
                  ( 'sofa'        , '04256520' ) ,
                  ( 'train'       , '04468005' ) ,
                  ( 'tvmonitor'   , '03211117' ) , ]

cate2synsetID = dict(cate_synsetID)
synsetIDs = [x[1] for x in cate_synsetID]


base_dir = '../SynImages_r4cnn.cache/'
db_path  = os.path.join(base_dir, 'ImageData_gtbox_crop.Rawpng.lmdb')

def build_db():

    try: os.makedirs(db_path)
    except: print('Make dirs skipped!')


    # NpyData_lmdb
    imdb = ImageData_lmdb(db_path, 'w')

    cate2imgIDs = {}
    # from multiprocessing import Pool
    for cate, synsetID in cate_synsetID:
        cate_dir = os.path.join(src_img_dir, synsetID)
        shapeIDs = [x for x in os.listdir(cate_dir) if os.path.isdir(os.path.join(cate_dir,x))]
        print(cate)
        for _k, shapeID in enumerate(tqdm(shapeIDs)):
            sys.stdout.flush()
            cate_shape_dir = os.path.join(cate_dir, shapeID)
            imgIDs = [x[:-4] for x in os.listdir(cate_shape_dir) if x.endswith('.png')]
            for imgID in imgIDs:
                imgpath = os.path.join(cate_shape_dir,imgID+'.png')
                cate2imgIDs.setdefault(cate, []).append(imgID)
                # write to lmdb.
                #-# img = cv2.imread(imgpath) # , cv2.IMREAD_UNCHANGED)
                #-# imdb[imgID] = img
                imdb.put(imgID, imgpath)

        Open(os.path.join(base_dir,'cate2imgIDs/%s.txt'%cate), 'w').write('\n'.join(cate2imgIDs[cate])+'\n')
        # exit()

    pickle.dump(cate2imgIDs, Open(os.path.join(base_dir,'cate2imgIDs.pkl'), 'wb'), protocol)
    return db_path


def read_image_size(db_path):

    imdb = ImageData_lmdb(db_path)
    print ("Nr. Images: ", len(imdb.keys), imdb.len)

    imgID2size = {}
    for img_id in imdb.keys:
        # print
        h, w, c = imdb[img_id].shape
        assert c==3, img_id
        imgID2size[img_id] = (h, w)
    pickle.dump(imgID2size, open(os.path.join(db_path,'imgID2size.pkl'), 'wb'), protocol)





def gen_anno_db(cate, MAX_STRING_LEN=64):
    import numpy as np
    from numpy_db import npy_table, npy_db, dtype_summary, reorder_dtype
    add_path('../../')
    from Pascal3D import image_info, object_anno

    cate2imgIDs = pickle.load( open(os.path.join(base_dir,'cate2imgIDs.pkl'), 'rb') )
    imgID2size = pickle.load( open(os.path.join(db_path,'imgID2size.pkl'),'rb') )

    # for cate in categories:
    if True:
        imgIDs = cate2imgIDs[cate]

        obj_rcs = np.zeros( (len(imgIDs),), dtype=object_anno )
        for _k_, imgID in enumerate(imgIDs):
            if _k_%1000==0:
                print ('\r%-20s    %6d  / %6d       ' % (cate, _k_, len(imgIDs)))
                sys.stdout.flush()
            synsetID, shapeID, a,e,t,d = imgID.split('_')  # e.g.  02691156_b089abdb33c39321afd477f714c68df9_a357_e034_t-03_d002
            assert a[0]=='a' and e[0]=='e' and t[0]=='t' and d[0]=='d', imgID
            a,e,t,d = float(a[1:]), float(e[1:]), float(t[1:]), float(d[1:])
            """ NOTE: The tilt angle should be flipped as to what is written in the filename
                to make it consistent with PASCAL3D annotation. (https://github.com/ShapeNet/RenderForCNN)"""
            t = -t

            #h, w, c = datadb[imgID].shape
            #assert c==3, imgID # datadb[imgID].shape
            h,w = imgID2size[imgID]

            # It's quite annoying that 232 imgIDs (from chair category) are of 66 length,
            # which exceed the MAX_STRING_LEN=64 in db_type
            # So here the solution is as following:
            obj_id   = imgID[:MAX_STRING_LEN]
            image_id = imgID[MAX_STRING_LEN:]+'.Syn'

            #======================================
            #--[new object record]
            # rcobj_1arr = np.zeros( (1,), dtype=object_anno )
            rcobj = obj_rcs[_k_].view(np.recarray) # cast as recarray
            #
            rcobj.obj_id    = obj_id    # "{}.{}".format(imgID, 'Syn')  # Any obj_id ends up with '.Syn' is from this dataset.
            rcobj.category  = cate      # obj['class'] # cate
            rcobj.cad_id    = shapeID   # "%s%02d" % (obj['class'], obj.cad_index)
            rcobj.bbox[:]   = [0,0,w,h] # just the image boarder (x1,y1,x2,y2).
            #-----[source image]
            rcobj.src_img.image_id = image_id  # imgID
            rcobj.src_img.H        = h  # Original image size
            rcobj.src_img.W        = w
            rcobj.src_img.C        = 3
            rcobj.src_img.h        = h  # The size of image this annotation based on,
            rcobj.src_img.w        = w  # it can be different from original image if rescaled by MAX_IMAGE_SIDE.
            #-----[viewpoint]
            rcobj.gt_view.a  = a
            rcobj.gt_view.e  = e
            rcobj.gt_view.t  = t
            rcobj.gt_view.d  = d  # Note: if distance=0, this means a,e field is coarse vp annotation.
            rcobj.gt_view.px = 0  # --- Not-Available! ---
            rcobj.gt_view.py = 0  # --- Not-Available! ---
            rcobj.gt_view.f  = 0  # --- Not-Available! ---
            rcobj.gt_view.mx = 0  # --- Not-Available! ---
            rcobj.gt_view.my = 0  # --- Not-Available! ---
            #--other annotation.
            rcobj.difficult  = 0  # --- Not-Available! ---
            rcobj.truncated  = 0  # --- Not-Available! ---
            rcobj.occluded   = 0  # --- Not-Available! ---
            #======================================
        #
        # in_file_path = os.path.join(out_dir, 'obj_anno_dump/%s.pkl' % cate)
        # obj_rcs = pickle.load(open(in_file_path))

        # create npy_table and npy_db, and dumpy it.
        out_file_path = os.path.join(base_dir, 'obj_anno_db/%s.pkl' % cate)
        obj_tb = npy_table(obj_rcs)
        db = npy_db()
        db.add_table(obj_tb, name="obj_tb")
        db.dump(out_file_path)
        print ('%-20s  %6d  [dump to] %s' % (cate, len(imgIDs), out_file_path))





def test(visualize=True):

    datadb = ImageData_lmdb(db_path)
    print (datadb.keys)
    for k in datadb.keys:
        img = datadb[k]

        if visualize: # :
            cv2.imshow('real_img', img)
            cv2_wait()



if __name__ == '__main__':

    # --- build up lmdb ---
    build_db()
    read_image_size(db_path)

    # --- build up annotation db ---
    # for cate in categories:
    #     gen_anno_db(cate, MAX_STRING_LEN=64)
    #
    from multiprocessing import Pool
    pool = Pool(12)
    pool.map(gen_anno_db, categories)

