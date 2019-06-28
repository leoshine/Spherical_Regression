"""
 @Author  : Shuai Liao
"""

import os,sys
from basic.common import env, add_path, Open
add_path('../../')
from Pascal3D import get_imgIDs, get_anno, categories, category2nrmodel
import numpy as np
import pickle

from basic.util import load_yaml
from tqdm import tqdm

conf = load_yaml('config.yml')  # odict
Pascal3D_root = os.path.expanduser(conf['Pascal3D_release_root'])
protocol      = conf['pkl_protocol']  # pickle dump protocol. Change -1 to 2 for python2.x compatibility.


base_dir   = '../'
imgID2size = pickle.load(open(os.path.join(base_dir, 'ImageData.Rawjpg.lmdb/imgID2size.pkl'),'rb'))



#----------------------------------------------------------------------------------
# Note:
#   The purpose of the script is to generate objId2gtbox.pkl
#   Whereas rcobj has gt_bbox field, however, we still regenerate objId2gtbox.pkl
#   to make sure these gt_bbox is properly clamped by image boarding,
#   and has no problem in cropping, resizing (to avoid exception during training).
#----------------------------------------------------------------------------------

def process(rcobj):
    # read data from lmdb
    # img = imgdb[rcobj.src_img.image_id]
    # assert img is not None, rcobj.src_img.image_id
    # h,w,_  = img.shape

    imgID = rcobj.src_img.image_id
    h,w = imgID2size[imgID]

    x1,y1,x2,y2 = map(int, rcobj.bbox)
    # print x1,y1,x2,y2
    x1 =max(x1, 0  )
    y1 =max(y1, 0  )
    x2 =min(x2, w-1)
    y2 =min(y2, h-1)

    if x2<=x1 or y2<=y1:
        error_record = '%-40s  %-20s  %d,%d,%d,%d'%(rcobj.category, rcobj.obj_id, x1,y1,x2,y2)
        open('error-anno.txt','a+').write(error_record+'\n')
        print ("[error-anno]  ", error_record)
        exit()
        # return []

    gt_box = np.array([x1,y1,x2,y2], np.int32)

    return gt_box





def main(collection='train', filter='all',
         cates=categories,  # cates=['aeroplane','boat','car'],  #
        ):


    out_dir = '../anno_db_v2/data.cache/objId2gtbox'

    try: os.makedirs(out_dir)
    except: print('Make dirs skipped!')

    # from multiprocessing import Pool
    objId2gtbox = dict()
    nr_box = 0
    for cate in cates:
        print(' >>> %10s %5s  %20s    ' % (collection, filter, cate))
        objIDs, rcobjs = get_anno(cate, collection=collection, filter=filter)

        for _k, rcobj in enumerate(rcobjs): # tqdm()
            # print ("%s    %s / %s        " % (cate, _k, len(rcobjs)))
            gt_box = process(rcobj) # resize_shape cate, _k, len(rcobjs)
            objId2gtbox[rcobj.obj_id] = gt_box
            nr_box += 1

    outpath = os.path.join(out_dir, 'cate%s_%s.%s.pkl' % (len(cates), collection, filter))
    pickle.dump(objId2gtbox, Open(outpath, 'wb'), protocol)
    print ('[outpath]: ', outpath)
    print ('nr_box:  ', nr_box)



if __name__ == '__main__':
    main( 'train', 'easy', cates=categories )
    main( 'val',   'easy', cates=categories )

    main( 'train', 'all' , cates=categories )
    main( 'val',   'all' , cates=categories )


"""
matlab code:

function boxes =  overlappingBoxes(box,imgDims)
deltaX = (box(3)-box(1))/6;
deltaY = (box(4)-box(2))/6;
boxes = [];
for x1Shift = -1:1
    for y1Shift = -1:1
        for x2Shift = -1:1
            for y2Shift = -1:1
                boxes(end+1,:) = [box(1)+x1Shift*deltaX box(2)+y1Shift*deltaY box(3)+x2Shift*deltaX box(4)+y2Shift*deltaY];
            end
        end
    end
end
boxes = round(boxes);
%boxes(:,1) = max(boxes(:,1),1);
%boxes(:,2) = max(boxes(:,2),1);
%boxes(:,3) = min(boxes(:,3),imgDims(2));
%boxes(:,4) = min(boxes(:,4),imgDims(1));
"""
