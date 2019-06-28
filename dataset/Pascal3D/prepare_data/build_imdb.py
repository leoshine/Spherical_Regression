"""
 @Author  : Shuai Liao
"""

from basic.common import add_path, env, is_py3
from lmdb_util import ImageData_lmdb
from tqdm import tqdm
import os, sys, cv2
import numpy as np
import lmdb
import pickle

add_path('../../')
from Pascal3D import get_imgIDs, get_anno, categories

from basic.util import load_yaml

conf = load_yaml('config.yml')  # odict
Pascal3D_root = os.path.expanduser(conf['Pascal3D_release_root'])
protocol      = conf['pkl_protocol']  # pickle dump protocol. Change -1 to 2 for python2.x compatibility.

base_dir    = '../'
db_path = os.path.join(base_dir, 'ImageData.Rawjpg.lmdb')

def build_db(fn_pattern='.jpg'):

    src_img_dir = os.path.join(Pascal3D_root, 'Images')
    #
    try: os.makedirs(db_path)
    except: pass
    imdb = ImageData_lmdb(db_path, 'w')  # 'a+')  #

    allIDs = []
    for collection in ['train','val']:
        for label, cate in enumerate(categories):
            _, rcobjs = get_anno(cate, collection=collection, filter='all', img_scale='Org', withCoarseVp=True)
            imgIDs = get_imgIDs(rcobjs)

            print ('%15s  %s   %5d' % (cate, collection, len(imgIDs)))
            for i, imgID in enumerate(tqdm(imgIDs)):
                # if is_py3:
                #     imgID = imgID.decode('UTF-8')

                if imgID[0]=='n':
                    fo = '%s_imagenet' % cate
                else:
                    fo = '%s_pascal'   % cate
                image_file = os.path.join(src_img_dir, fo, '%s.%s' % (imgID, fn_pattern.strip('.')))
                assert os.path.exists(image_file), image_file

                img = cv2.imread(image_file) # , cv2.IMREAD_UNCHANGED)
                imdb[imgID] = img
                allIDs.append(imgID)

    imdb.close()
    print ('All Images: %d' % len(allIDs))
    print ('All Images: %d  (unique)' % len(set(allIDs)))

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





if __name__ == '__main__':

    db_path = os.path.join(base_dir, 'ImageData.Rawjpg.lmdb')
    if os.path.exists( db_path ):
        print('%s exists. skipped.' % db_path)
        pass
    else:
        build_db(fn_pattern='.jpg')  # (fn_pattern='.JPEG')
        read_image_size(db_path)

    #creat_db(src_img_dir=os.path.join(PASCAL3D_Dir, 'ImageData.Max500'), dst_img_fmt='Rawjpg')
    #read_image_size(db_type="Max500.Rawjpg")


