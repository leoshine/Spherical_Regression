"""
 @Author  : Shuai Liao

This script evaluate the result with original gt surface normal images.
The difference to eval_sn.py should be rather minor.
"""

import os,sys
import numpy as np
import cv2
from basic.common import env, add_path, cv2_putText, cv2_wait


data_dir  = '/path/to/nyu_v2/'
gt_path   = data_dir+'torch_data/{img_id}_norm_camera.png'
mask_path = data_dir+'torch_data/{img_id}_valid.png'
#
# test_model = 'sync_physic_nyufinetune_nyu_sample_test'
# test_model = 'sync_physic_nyu_sample_test'
# test_model = 'sync_opengl_nyu_sample_test'
test_model = 'train_examplefinal_nyu_sample_test'
if len(sys.argv)>1:
    test_model = sys.argv[1]
rslt_path = '/path/to/result/surface/normal/images/%s/torch_data_{img_id}_normal_est.png' % test_model



def read_imgdata(img_path, flag=cv2.IMREAD_UNCHANGED): # IMREAD_ANYDEPTH):
    assert os.path.exists(img_path), "[File not found] %s" % img_path
    return cv2.imread(img_path, flag)


# Calculate average end point error
def eval_one(NGt, NPr, mask=None):
    assert NGt.shape[:2]==NPr.shape[:2]
    assert len(NGt.shape)<=3 and len(NPr.shape)<=3
    if len(NGt.shape)==3:
        NGt = NGt.reshape(-1,3)
    if len(NPr.shape)==3:
        NPr = NPr.reshape(-1,3)
    if mask is not None:
        if len(mask.shape)==2:
            mask = mask.reshape(-1)
        NGt = NGt[mask]
        NPr = NPr[mask]

    NGt = NGt / np.sqrt(np.power(NGt, 2).sum(axis=1, keepdims=True))
    NPr = NPr / np.sqrt(np.power(NPr, 2).sum(axis=1, keepdims=True))
    DP  = (NGt * NPr).sum(axis=1)
    T   = DP.clip(-1,1) # np.minimum(1, np.maximum(-1, DP)) #
    #
    E = np.rad2deg(np.arccos(T))
    return ( E, np.mean(E), np.median(E),
             np.sqrt(np.mean(np.power(E,2))),
             np.mean(E < 11.25) * 100,
             np.mean(E < 22.5 ) * 100,
             np.mean(E < 30   ) * 100, )


def process_one(test_img_id):
    """resize pred to (640,480)"""
    im    = read_imgdata(gt_path.format(img_id=test_img_id))
    gt    = read_imgdata(gt_path.format(img_id=test_img_id)).astype(np.float64)/(2**16)*2-1
    mask  = read_imgdata(mask_path.format(img_id=test_img_id)).astype(np.bool)
    # read prediction
    pr   = read_imgdata(rslt_path.format(img_id=test_img_id))
    pr   = cv2.resize(pr, (640,480), interpolation=cv2.INTER_NEAREST).astype(np.float64)/(2**8)*2-1
    E,mean,median, rmse, acc11,acc22,acc30 = eval_one(gt, pr, mask=mask)
    print ('\r %s  ' % (test_img_id), end='', flush=True)
    return E


def eval_all(test_ids=None, use_multiprocess=True):
    # from lmdb_util import ImageData_lmdb
    # imdb_gtNorm = ImageData_lmdb(data_dir+'NormCamera.Rawpng.lmdb', always_load_color=False)
    # imdb_gtMask = ImageData_lmdb(data_dir+'Valid.Rawpng.lmdb'     , always_load_color=False)
    #
    if test_ids is None:
        test_ids = [x.strip().split('/')[1] for x in open(data_dir+'testNdxs.txt').readlines()]

    # use multicores
    if use_multiprocess:
        from multiprocessing import Pool
        p = Pool(20)
        Es = p.map(process_one, test_ids)
    else:
        Es = []
        for i, test_img_id in enumerate(test_ids):
            E = process_one(test_img_id)
            Es.append(E)
            print ('\r %s / %s  ' % (i, len(test_ids)), end='', flush=True)
            #-----------------------------

    Es = np.concatenate(Es)
    #
    mean  = np.mean(Es)
    median= np.median(Es)
    rmse  = np.sqrt(np.mean(np.power(Es,2)))
    acc11 = np.mean(Es < 11.25) * 100
    acc22 = np.mean(Es < 22.5 ) * 100
    acc30 = np.mean(Es < 30   ) * 100
    return mean, median, rmse, acc11, acc22, acc30


if __name__ == '__main__':
    mean, median, rmse, acc11, acc22, acc30 = eval_all(use_multiprocess=True)
    print ('----------------------------------------')
    print ('mean  : %.3f' % mean                     )
    print ('median: %.3f' % median                   )
    print ('rmse  : %.3f' % rmse                     )
    print ('acc11 : %.3f' % acc11                    )
    print ('acc22 : %.3f' % acc22                    )
    print ('acc30 : %.3f' % acc30                    )

