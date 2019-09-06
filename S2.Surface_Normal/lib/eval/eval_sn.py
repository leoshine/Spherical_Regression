import os,sys
import numpy as np
import cv2
from basic.common import env, add_path, cv2_putText, cv2_wait

this_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(this_dir +'/../../../dataset')  # where the dataset directory is.
assert os.path.exists(base_dir), base_dir

data_dir = base_dir + '/SurfaceNormal/nyu_v2/'
# data_dir  = env.Home + '/working/cvpr19regpose/dataset/pbrs/nyu_data/'
# gt_path   = data_dir+'torch_data/{img_id}_norm_camera.png'
# mask_path = data_dir+'torch_data/{img_id}_valid.png'
#

def image2norm(normImg, trunc_back_facing=False):
    assert normImg.dtype in [np.uint8, np.uint16], normImg.dtype
    if normImg.dtype==np.uint8:
        norm = normImg.astype(np.float64)/(2**7) -1 # map to [-1,1]
    else:
        norm = normImg.astype(np.float64)/(2**15)-1 # map to [-1,1]
    if trunc_back_facing:
        norm[:,:,1][norm[:,:,1]>0] = 0              # Note: x-z-y order; this makes z>0 pixel as 0
    l2norm = np.sqrt(np.power(norm, 2).sum(axis=2, keepdims=True))
    l2norm[l2norm==0] = 1e-6 # prevent divided by 0
    norm /= l2norm           # normalization to unit vec
    return norm

def norm2image(norm, encode_bit=8):
    if encode_bit==8:
        normImg = ((norm+1)*(2**7)).astype(np.uint8)   # map [-1,1]  to [0,256)
    else:
        normImg = ((norm+1)*(2**15)).astype(np.uint16) # map [-1,1]  to [0,65535)
    return normImg


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


def process_one(args):
    test_img_id, imdb_gtNorm, imdb_gtMask, imdb_prNorm = args
    """resize pred to (640,480)"""
    gt    = image2norm(imdb_gtNorm[test_img_id], trunc_back_facing=True)
    mask  = imdb_gtMask[test_img_id].astype(np.bool)
    # read prediction
    pr   = imdb_prNorm[test_img_id]
    pr   = cv2.resize(pr, (640,480), interpolation=cv2.INTER_NEAREST)
    pr   = image2norm(pr, trunc_back_facing=True)
    #
    E,mean,median, rmse, acc11,acc22,acc30 = eval_one(gt, pr, mask=mask)
    print ('\r %s  ' % (test_img_id), end='', flush=True)
    # sys.stdout.flush()
    return E  # Es.append(E)


def eval_all(pred_dbpath, test_ids=None, use_multiprocess=False):
    from lmdb_util import NpyData_lmdb, ImageData_lmdb
    imdb_gtNorm = ImageData_lmdb(data_dir+'NormCamera.Rawpng.lmdb', always_load_color=False)
    imdb_gtMask = ImageData_lmdb(data_dir+'Valid.Rawpng.lmdb'     , always_load_color=False)
    imdb_prNorm = ImageData_lmdb(pred_dbpath, always_load_color=False)
    #
    if test_ids is None:
        test_ids = [x.strip().split('/')[1] for x in open(data_dir+'testNdxs.txt').readlines()]

    # use multicores
    if use_multiprocess:
        print ("Warning: TODO, haven't solve lmdb with multiprocessing issue.")
        raise NotImplementedError
        from multiprocessing import Pool, cpu_count
        from functools import partial
        from itertools import izip, repeat as Rp
        p = Pool(cpu_count())
        Es = p.map(process_one, izip(test_ids, Rp(imdb_gtNorm), Rp(imdb_gtMask), Rp(imdb_prNorm)))
    else:
        Es = []
        for i, test_img_id in enumerate(test_ids):
            E = process_one((test_img_id, imdb_gtNorm, imdb_gtMask, imdb_prNorm))
            Es.append(E)
            print ('\r %s / %s  ' % (i, len(test_ids)), end='', flush=True)

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
    pred_dbpath = '/path/to/naiReg/regNormSE/snapshots/vgg16se/PredNormal.Rawpng.lmdb'
    if len(sys.argv)>1:
        pred_dbpath = sys.argv[1]
    print (pred_dbpath)
    mean, median, rmse, acc11, acc22, acc30 = eval_all(pred_dbpath, use_multiprocess=False)
    print ('----------------------------------------')
    print ('mean  : %.3f' % mean                     )
    print ('median: %.3f' % median                   )
    print ('rmse  : %.3f' % rmse                     )
    print ('acc11 : %.3f' % acc11                    )
    print ('acc22 : %.3f' % acc22                    )
    print ('acc30 : %.3f' % acc30                    )


'''
mean  : 25.342
median: 18.984
rmse  : 33.031
acc11 : 30.300
acc22 : 57.040
acc30 : 68.948
'''






