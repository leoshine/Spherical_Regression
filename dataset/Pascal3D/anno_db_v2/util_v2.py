"""
 @Author  : Shuai Liao
"""

import os, sys
import numpy as np
import cv2
from basic.common import env, add_path, cv2_wait
from numpy_db import npy_table, npy_db, dtype_summary, reorder_dtype

this_dir = os.path.realpath(os.path.dirname(__file__))
add_path(this_dir)
from db_type  import viewpoint, proj_info, image_info, object_anno

PASCAL3D_Dir = os.path.realpath(this_dir+'/../')

def get_anno_db_tbl(cate, collection="val", filter='easy', img_scale='Org', withCoarseVp=True, quiet=True):
    # obtain obj annotation for dataset anno_db.
    assert img_scale in ['Max500', 'Org']
    if filter in ['easy', 'all', 'nonOccl', 'nonDiff']:
        filterStr = filter+'_withCoarseVp' if withCoarseVp else filter
        anno_folder = 'data.cache/{}.{}'.format(filterStr, img_scale)  #
    else:
        print ("Unidentified filter name=", filter)
        raise NotImplementedError

    anno_file = os.path.join(this_dir, anno_folder, '%s/%s.pkl' % (collection, cate) )
    # load npy_db
    db = npy_db()
    db.load(anno_file)
    if not quiet:
        print (db)  # print db summary.
    #
    obj_tb = db['obj_tb']
    return obj_tb

def get_anno_dbs_tbl(cates, collection="val", filter='easy', img_scale='Org', withCoarseVp=True, quiet=True):
    obj_tbs = []
    for cate in cates:
        obj_tbs.append(get_anno_db_tbl(cate, collection, filter, img_scale, withCoarseVp, quiet))
    return npy_table.merge(obj_tbs)

def get_anno(cate, collection="val", filter='easy', img_scale='Org', withCoarseVp=True, quiet=True):
    obj_tb = get_anno_db_tbl(cate, collection, filter, img_scale, withCoarseVp, quiet)
    return obj_tb.keys, obj_tb.recs.view(np.recarray) # obj_tb.data  #



#-------------------------------------------------------
def get_all_imgIDs(cate, collection="val"):
    '''Return all imgIDs regardless easy/difficult, truncated, occluded'''
    lines = open(PASCAL3D_Dir+'/ImageSets/{collection}/{cate}.txt'.format(collection=collection, cate=cate)).readlines()
    imgIDs = [x.strip() for x in lines if x.strip()!='']
    return imgIDs


def get_imgIDs(rcobjs):
    '''Get all unique image ids from rcojs records.'''
    imgIDs_easy = list(set(rcobjs.src_img.image_id))
    imgIDs_easy.sort()
    return imgIDs_easy



def get_imgID2size(img_scale='Org'):  # Max500
    import cPickle as pickle
    if   img_scale=='Org':
        return pickle.load(open(os.path.join(PASCAL3D_Dir, 'ImageData.Rawjpg.lmdb', 'imgID2size.pkl'), 'rb'))
    elif img_scale=='Max500':
        return pickle.load(open(os.path.join(PASCAL3D_Dir, 'ImageData.%s.Rawjpg.lmdb'%img_scale, 'imgID2size.pkl'), 'rb'))
    else:
        raise NotImplementedError
#--------------------------------------------------------




def mirror_anno(rcobj):
    raise NotImplementedError
    # Almost impossible (if 3D model is not symmetric.)
    # Do mirror the rendered image instead.
    # vp = rcobj.gt_view
    # im = rcobj.src_img
    # vp.a = -vp.a
    # vp.e = vp.e-180
    # vp.px = im.w - vp.px
    # return rcobj

def rescale_anno(rcobj, dsize, fx=None, fy=None):
    # This usage of this function is similar to cv2.resize()
    """
    # fx, fy:rescale factor about x-axis and y-axis

    Following field will be updated:
        - obj_anno.src_img.h
          obj_anno.src_img.w
        - obj_anno.bbox
        - obj_anno.gt_view.px
          obj_anno.gt_view.py
          obj_anno.gt_view.mx
          obj_anno.gt_view.my
    """
    # the old image size.
    im = rcobj.src_img
    vp = rcobj.gt_view
    _h, _w = im.h, im.w

    # Check max side no bigger than 500.
    H, W = im.H, im.W
    if dsize is not None: #----- Rescale by dsize, (fx,fy are ignored)
        new_h, new_w = dsize
        fx, fy = float(new_h)/im.h,  float(new_w)/im.w
        assert fx>0 and fy>0
    else:                 #----- Rescale by fx,fy
        # Warning: don't use int(), it will always round down and possibly produce MAX_IMAGE_SIDE=499 instead of 500
        new_h, new_w = int(round(fx*H)), int(round(fy*W))

    if fx!=1 or fy!=1: # necessary to do rescaling.
        #-- rescale h, w
        im.h, im.w = new_h, new_w
        #-- rescale bbox
        rcobj.bbox *= [fx,fy, fx,fy] # rescale x1, y1, x2, y2 separately.
        #-- rescale principle point(px,py)
        vp.px *= fx
        vp.py *= fy
        #-- rescale viewport(mx,my)
        vp.mx *= fx
        vp.my *= fy

    # obj_anno is already updated, return just in case needed.
    return rcobj


def rescale_viewpoint(vp, fx=1.0, fy=1.0):
    """
    Following field will be updated:
        - obj_anno.gt_view.px
          obj_anno.gt_view.py
          obj_anno.gt_view.mx
          obj_anno.gt_view.my
    """
    if fx!=1 or fy!=1: # necessary to do rescaling.
        # rescale principle point(px,py)  viewport(mx,my)
        vp.px *= fx
        vp.py *= fy
        vp.mx *= fx
        vp.my *= fy

    return vp




def is_anno_of_org(rcobj):
    '''Test if annotation is on original image size.'''
    anno_size = (rcobj.src_img.h, rcobj.src_img.w)
    org_size  = (rcobj.src_img.H, rcobj.src_img.W)
    return anno_size == org_size

def recale_anno_to_org(rcobj):
    # Warning[1]:  update_proj=False, with_visibility=False
    # Warning[2]: this function will change data field in rcobj. If you want keep it, pass in rcobj.copy()
    if not is_anno_of_org(rcobj):
        return rescale_anno(rcobj, (rcobj.src_img.H, rcobj.src_img.W))
    else:
        return rcobj




if __name__ == '__main__':

    add_path('../../')
    from Pascal3D import categories

    collection =  'train' # 'val' #
    Nr_easy = Nr_nonDiff = Nr_all = 0
    for cate in categories:
        anno_db  = get_anno_db_tbl(cate, collection=collection, filter='easy', withCoarseVp=True, quiet=True)
        nr_easy  = len(anno_db.keys)
        Nr_easy += nr_easy

        anno_db  = get_anno_db_tbl(cate, collection=collection, filter='nonDiff', withCoarseVp=True, quiet=True)
        nr_nonDiff = len(anno_db.keys)
        Nr_nonDiff += nr_nonDiff

        anno_db  = get_anno_db_tbl(cate, collection=collection, filter='all', withCoarseVp=True, quiet=True)
        nr_all   = len(anno_db.keys)
        Nr_all  += nr_all

        print ('%-30s   %5s  %5s  %5s' % (cate, nr_easy, nr_nonDiff, nr_all))
    print ('%-30s   %5s  %5s  %5s' % ('TOTAL', Nr_easy, Nr_nonDiff, Nr_all))


'''
                                      no CoarseVp             with CoarseVp
                                  ---------------------------------------------
        (pascal voc12 val)        easy nonDiff nr_all |     easy nonDiff nr_all
                                  ---------------------------------------------
aeroplane                          275    389    430  |      276    433    484
bicycle                            118    341    355  |      118    358    380
boat                               232    351    376  |      244    424    491
bottle                             251    588    645  |      252    630    733
bus                                154    271    281  |      154    301    320
car                                308    858    932  |      310   1004   1173
chair                              244   1085   1244  |      247   1176   1449
diningtable                         21    215    238  |       23    305    374
motorbike                          136    331    339  |      137    356    376
sofa                                39    228    267  |       40    285    387
train                              113    289    294  |      113    315    329
tvmonitor                          222    369    387  |      222    392    414
TOTAL                             2113   5315   5788  |     2136   5979   6910

                                      no CoarseVp             with CoarseVp
                                  ---------------------------------------------
        (pascal voc12 train)      easy nonDiff nr_all |     easy nonDiff nr_all
                                  ---------------------------------------------
aeroplane                         2074   2365   2391  |      2075   2407   2445
bicycle                            912   1681   1727  |       912   1695   1752
boat                              2516   2938   2967  |      2528   3023   3105
bottle                            1620   2112   2201  |      1621   2156   2276
bus                               1186   1344   1357  |      1187   1376   1401
car                               5667   6484   6565  |      5673   6645   6823
chair                             1204   2142   2318  |      1205   2231   2510
diningtable                        763   2541   2563  |       764   2617   2686
motorbike                          775   1584   1594  |       775   1614   1632
sofa                               675   1687   1720  |       677   1739   1857
train                             1136   1595   1601  |      1139   1621   1635
tvmonitor                         1398   1626   1643  |      1398   1644   1664
TOTAL                            19926  28099  28647  |     19954  28768  29786




'''
