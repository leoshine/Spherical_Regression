"""
 @Author  : Shuai Liao
"""

from basic.common import env, add_path
add_path('../../')
from Pascal3D import get_imgIDs, get_anno, categories

import os, sys, pickle

base_dir   = '../'
imgID2size = pickle.load(open(os.path.join(base_dir, 'ImageData.Rawjpg.lmdb/imgID2size.pkl'),'rb'))



def check_one(rcobj, visualize=False):
    # get image size
    imgID = rcobj.src_img.image_id
    h,w = imgID2size[imgID]

    x1,y1,x2,y2 = map(int, rcobj.bbox)
    # print  (x1,y1,x2,y2)

    x1 =max(x1, 0  )
    y1 =max(y1, 0  )
    x2 =min(x2, w-1)
    y2 =min(y2, h-1)

    if x2<=x1 or y2<=y1:
        error_record = '%-40s  %-20s  %d,%d,%d,%d'%(rcobj.category, rcobj.obj_id, x1,y1,x2,y2)
        open('error-anno.txt','a+').write(error_record+'\n')
        print ("[error-anno]  ", error_record)
        print ('img.shape: ', (h,w))
        print (rcobj.bbox)
        # print (rcobj.obj_id, img[y1:y2,x1:x2].shape)
        # if visualize:
        #     _img = img.copy()
        #     x1,y1,x2,y2 = 15,15, 460, 440
        #     cv2.rectangle(_img, (x1,y1), (x2,y2), (255,0,0))
        #     #cv2.imshow('img', _img)
        #     #cv2.imwrite('%s.jpg' % rcobj.obj_id, _img)
        #     add_path(env.Home+'/working/cvtools/cvRectangleTool')
        #     from cvRectTool import drawRect
        #     x=drawRect(_img, rcobj.obj_id)
        #     x.run()
        #     print (error_record)
        #     exit()

        # n02691156_39039-401,424,1999,1381  [  401.1399689    424.97433904  1999.82892691  1381.09097978] -> ( 92, 85,402,264)
        # n03335030_14098-548,558,4957,2673  548,558,499,332 (333, 500, 3)  [  548.71150855   558.91290824  4957.73172628  2673.99844479] --> ( 56, 50,499,255)




def check(cates=categories):
    nr_box = 0
    for cate in cates:
        print(cate)
        for collection in ['train', 'val']:
            objIDs, rcobjs = get_anno(cate, collection=collection)
            for _k, rcobj in enumerate(rcobjs):
                check_one(rcobj) # resize_shape cate, _k, len(rcobjs)




if __name__ == '__main__':
    check(cates=categories)

