"""
 @Author  : Shuai Liao
"""
from basic.common import add_path, env, Timer, Checkpoint, cv2_wait, is_py3
from basic.util import load_yaml, dump_yaml

import os, sys
import cv2
import numpy as np
import lmdb
from pprint import pprint
if is_py3:
    import pickle
else:
    import cPickle as pickle
import random

'''Prepare data to be store in lmdb.
    Return key, value  (serialized bytes data)
'''
class Data_Handler:
    def pack(self, imgID, imgpath):
        raise NotImplemented

    def unpack(self, key):
        raise NotImplemented


class Handle_Npyarr(Data_Handler):
    def pack(self, imgID, imgpath):
        img = cv2.imread(imgpath)
        h,w,c = img.shape
        #
        key   = imgID.encode('utf8') # ascii
        value = img.dumps()
        return key, value

    def pack_from_npyarr(self, imgID, npyarr):
        #
        key   = imgID.encode('utf8') # ascii
        value = npyarr.dumps()
        return key, value

    def unpack(self, data_bytes):
        if is_py3:
            return np.loads(data_bytes, encoding='latin1')
        else:
            return np.loads(data_bytes)

class Handle_Rawjpg(Data_Handler):
    def __init__(self, always_load_color=True):
        if always_load_color:
            self.cv2_IMREAD_FLAG = cv2.IMREAD_COLOR
        else:
            self.cv2_IMREAD_FLAG = cv2.IMREAD_UNCHANGED

    def pack(self, imgID, imgpath):
        rawbytes = open(imgpath, 'rb').read()
        #
        key   = imgID.encode('utf8') # ascii
        value = rawbytes
        return key, value

    def pack_from_npyarr(self, imgID, npyarr):
        retval, buf = cv2.imencode('.jpg', npyarr)
        assert retval==True
        rawbytes = buf.tostring()
        #
        key   = imgID.encode('utf8') # ascii
        value = rawbytes
        return key, value

    def unpack(self, data_bytes):
        data_npyarr = np.frombuffer(data_bytes, np.uint8)
        img = cv2.imdecode(data_npyarr, self.cv2_IMREAD_FLAG) # cv2.IMREAD_UNCHANGED) # IMREAD_COLOR) # CV_LOAD_IMAGE_COLOR)
        return img

class Handle_Rawebp(Data_Handler):
    """Webp format for storage.
        In comparison to the lossless compression of PNG,
        or lossy compression of JPEG, Webp has noticeably advantage.

        QUALITY from 0 to 100 (the higher is the better). Default value is 95.
    """
    def __init__(self, always_load_color=True, QUALITY=95):
        if always_load_color:
            self.cv2_IMREAD_FLAG = cv2.IMREAD_COLOR
        else:
            self.cv2_IMREAD_FLAG = cv2.IMREAD_UNCHANGED

        self.QUALITY = QUALITY

    def pack(self, imgID, imgpath):
        rawbytes = open(imgpath, 'rb').read()
        #
        key   = imgID.encode('utf8') # ascii
        value = rawbytes
        return key, value

    def pack_from_npyarr(self, imgID, npyarr):
        retval, buf = cv2.imencode('.webp', npyarr, [cv2.IMWRITE_WEBP_QUALITY, self.QUALITY])
        assert retval==True
        rawbytes = buf.tostring()
        #
        key   = imgID.encode('utf8') # ascii
        value = rawbytes
        return key, value

    def unpack(self, data_bytes):
        data_npyarr = np.frombuffer(data_bytes, np.uint8)
        img = cv2.imdecode(data_npyarr, self.cv2_IMREAD_FLAG) # cv2.IMREAD_UNCHANGED) # IMREAD_COLOR) # CV_LOAD_IMAGE_COLOR)
        return img



class Handle_Rawpng(Data_Handler):
    '''
    cv::IMWRITE_PNG_COMPRESSION
    For PNG, it can be the compression level from 0 to 9. A higher value means a smaller size and longer compression time.
    If specified, strategy is changed to IMWRITE_PNG_STRATEGY_DEFAULT (Z_DEFAULT_STRATEGY). Default value is 1 (best speed setting).

    config yaml:
    { png_dtype : xx, # uint8 | uint16
      npy_dtype : xx, # float32
      min   : xx,
      max   : xx,
      }
    '''
    # re_PXL_TYPE = r'CV_\d+(?:U|F|S)C\d+' # b(?:eq|ne|lt|gt)
    # png_PXL_MAX  = dict(uint8=255, uint16=65535)

    def __init__(self, always_load_color=True, remap=None):
        """e.g.
            remap=dict(png_dtype='uint16',npy_dtype='float32', min=[0,0], max=[1,1] )
        """
        if always_load_color:
            self.cv2_IMREAD_FLAG = cv2.IMREAD_COLOR
        else:
            self.cv2_IMREAD_FLAG = cv2.IMREAD_UNCHANGED
        if remap is not None:
            self.cv2_IMREAD_FLAG = cv2.IMREAD_UNCHANGED  # if remap is need, always load unchanged.
            #
            if hasattr(remap['min'], '__len__'):
                assert len(remap['min'])==len(remap['max']) # channel of pixels (Can be 1,2,3,4)
                self.c = len(remap['min'])
            else:
                self.c = 0
            assert remap['png_dtype'] in ['uint8', 'uint16'], remap['png_dtype']
            self.png_PXL_MAX = dict(uint8=255, uint16=65535)[ remap['png_dtype'] ]
            remap['npy_dtype'] = eval('np.'+remap['npy_dtype']) # make it as np.dtype
            remap['png_dtype'] = eval('np.'+remap['png_dtype']) # make it as np.dtype
            remap['min'] = np.array(remap['min'], dtype=remap['npy_dtype'])
            remap['max'] = np.array(remap['max'], dtype=remap['npy_dtype'])
        self.remap = remap

    # def remap(self, ):


    def pack(self, imgID, imgpath):
        rawbytes = open(imgpath, 'rb').read()
        if self.remap is not None:
            img = cv2.imdecode(np.frombuffer(rawbytes), self.cv2_IMREAD_FLAG)
            return self.pack_from_npyarr(imgID, img)
        #
        key   = imgID.encode('utf8') # ascii
        value = rawbytes
        return key, value

    def pack_from_npyarr(self, imgID, npyarr):
        if self.remap is not None:
            # convert to a pixel in (min,max) -> (0,1.0)
            assert npyarr.dtype==self.remap['npy_dtype']
            if self.c>0:
                assert len(npyarr.shape)==3 and npyarr.shape[2]==self.c
            npyarr = (npyarr-self.remap['min'])/(self.remap['max']-self.remap['min'])  # map to [0.0,1.0]
            npyarr = (npyarr * self.png_PXL_MAX).astype(self.remap['png_dtype'])       # map to [0,255] or [0,65535] # uint8 | uint16
            h,w = npyarr.shape[:2]
            # if len(npyarr.shape)==2:
            #     h,w = npyarr.shape
            # else:
            #     h,w,c = npyarr.shape
            if self.c==2: # pad 3rd channel with 0
                _npyarr = np.zeros((h,w,3), dtype=npyarr.dtype)
                _npyarr[:,:,:self.c] = npyarr
                npyarr = _npyarr
        retval, buf = cv2.imencode('.png', npyarr)
        assert retval==True
        rawbytes = buf.tostring()
        #
        key   = imgID.encode('utf8') # ascii
        value = rawbytes
        return key, value

    def unpack(self, data_bytes):
        data_npyarr = np.frombuffer(data_bytes, np.uint8)
        img = cv2.imdecode(data_npyarr, self.cv2_IMREAD_FLAG)  # cv2.IMREAD_UNCHANGED) # cv2.IMREAD_COLOR) # CV_LOAD_IMAGE_COLOR)
        if self.remap is not None:
            if self.c>0:
                img = (img[:,:,:self.c]/self.png_PXL_MAX).astype(self.remap['npy_dtype'])  # map to [0.0,1.0]
            else: # single channel image
                img = (img/self.png_PXL_MAX).astype(self.remap['npy_dtype'])               # map to [0.0,1.0]
            img = img*(self.remap['max']-self.remap['min'])+self.remap['min'] # map to [min,max]  # * 65535 # uint16
            assert img.dtype==self.remap['npy_dtype']  # [TODO] TO REMOVE
        return img



def _bytes(key):
    if   isinstance(key, np.string_) or isinstance(key, np.bytes_):
        return key.tobytes()
    elif isinstance(key, str):
        return key.encode('utf8')
    elif isinstance(key, bytes):
        return key
    else:
        print("Unknown type of key: ", type(key))
        raise NotImplementedError

# def resize2max(img, h=None, w=None):
#     assert not (h is None and w is None)

def pad_as_squared_img(img, Side=150):
    # resize image
    h,w = img.shape[:2]
    maxside = max(h,w)
    if Side is None or Side<=0:
        scale=1.0
    else:
        scale = Side/float(maxside)
    img = cv2.resize(img, None, fx=scale,fy=scale, interpolation=cv2.INTER_AREA) #INTER_LINEAR)
    # pad image
    if len(img.shape)==2:
        img = img[:,:,np.newaxis]
    h,w,c = img.shape
    # c = img.shape[2] if len(img.shape)==3 else 1
    maxside = max(h,w)
    squared_img = np.full((maxside, maxside, c), 255, dtype=img.dtype)
    if w<maxside:
        start = (maxside-w)//2
        squared_img[:,start:start+w, :] = img
    else:
        start = (maxside-h)//2
        squared_img[start:start+h,:, :] = img
    return squared_img



class ImageData_lmdb:
    def __init__(self, db_path, mode='r', map_N=30000, max_readers=256, always_load_color=True, silent=False, **kwargs): # map_N=250000
        """ [kwargs examples]:
              remap=dict(png_dtype='uint16',npy_dtype='float32', min=[0,0], max=[1,1] )   # for png handler of opt-flow (2-channel float32 img)
        """
        self.db_path  = os.path.abspath(db_path.rstrip("/")) # remove last '/' if there's any.
        # assert os.path.exists(self.db_path)
        self.mode    = mode
        if   self.mode=='w':
            os.system('rm -rf %s' % self.db_path) # overwrite if overwrite and os.path.exists(self.db_path):
        elif self.mode in ['r','a+']:
            assert os.path.exists(self.db_path), "[Path not exists] %s" %self.db_path
        else:
            raise NotImplementedError
        # self.lmdb_env = lmdb.open(self.db_path, map_size=self.map_size)
        if self.mode=='r':
            self.map_size = (map_N*256*256*3*4)
            self.lmdb_env = lmdb.open(self.db_path, map_size=self.map_size, max_readers=max_readers,readahead=True,readonly=True,lock=False) #
        else:
            self.map_size = (map_N*256*256*3*4) * 10
            self.lmdb_env = lmdb.open(self.db_path, map_size=self.map_size, max_readers=max_readers) # lock=True

        if   self.db_path.endswith('.Rawjpg.lmdb'):
            if not silent: print("[Using] Handle_Rawjpg")
            self.handle = Handle_Rawjpg(always_load_color=always_load_color) # bytes data handler (pack/unpack)
        elif self.db_path.endswith('.Rawpng.lmdb'):
            if not silent: print("[Using] Handle_Rawpng")
            yamlfile = os.path.join(db_path, 'remap.yml')
            if self.mode in ['r','a']:
                ''' e.g. {dtype: 32FC2, min : 0.0, max : 1.0} '''
                remap = load_yaml(yamlfile) if os.path.exists(yamlfile) else None
                print('---------> remap yaml: ', remap)
            else: # write mode
                remap = kwargs.get('remap', None)
                print("Write png with remap: %s" % remap)
                dump_yaml(remap, yamlfile)
            self.handle = Handle_Rawpng(always_load_color=always_load_color,remap=remap) # bytes data handler (pack/unpack)
        elif self.db_path.endswith('.Npyarr.lmdb'):
            if not silent: print("[Using] Handle_Npyarr")
            self.handle = Handle_Npyarr()
        elif self.db_path.endswith('.Rawebp.lmdb'):
            if not silent: print("[Using] Handle_Rawebp")
            if self.mode in ['w', 'a']:
                QUALITY = kwargs.get('QUALITY', 95)
                print("Compress QUALITY: ", QUALITY)
                self.handle = Handle_Rawebp(QUALITY=QUALITY)
            else:
                self.handle = Handle_Rawebp()
        else:
            print ('Unrecognized imagedata_lmdb extension:\n[db_path] %s' % self.db_path)
            raise NotImplementedError

        if not silent: print(self)
        # print(self.len)

        """ --- patch for rename keys ---
            In lmdb, "keys are always lexicographically sorted".
            This prevent us to shuffle the storage order of images, which is necessary when the training dataset size is really large, e.g ImageNet (45G~120G).
            Pre-shuffling image order favor image data loader in training code, as it can read sequentially along the physical storage.
            To do this, we re-name all image keys to '0000XXXXX' ('%09d' % image_id) format as it would be sorted by lmdb (same trick in caffe).
            So when imgId2dbId.pkl, we need to map the actual image_id to db_id for retrieve a data.
        """
        if os.path.exists(os.path.join(self.db_path, 'imgId2dbId.pkl')):
            assert self.mode not in ['w','a+'], 'Not implement renamed key '
            self.key_renamed = True
            self.imgId2dbId = pickle.load(open(os.path.join(self.db_path, 'imgId2dbId.pkl'), 'rb'))  # OrderedDict
        else:
            self.key_renamed = False

    def __str__(self):
        s = "[Path] %s \n" % self.db_path
        for k,v in self.lmdb_env.stat().items():
            s += '%20s  %-10s\n' % (k,v)
        return s

    @property
    def len(self):
        return self.lmdb_env.stat()['entries']

    @property
    def keys(self):
        if self.key_renamed:
            return self.imgId2dbId.keys()
        else:
            key_cache_file = os.path.join(self.db_path, 'keys.txt')
            if not os.path.exists(key_cache_file):
                print("Building up keys cache ...")
                with self.lmdb_env.begin() as txn:
                    keys = [ key.decode('utf-8')  for key, _ in txn.cursor() ] # very slow!!!
                    # Note: .decode('utf-8') is to decode the bytes object to produce a string. (needed for py3)
                with open(key_cache_file,'w') as f:
                    f.write('\n'.join(keys))
            else:
                return [x.strip() for x in open(key_cache_file).readlines()]
            return keys

    def __getitem__(self, key):
        ''' key is usually imgID, normally we need key as bytes.'''
        with self.lmdb_env.begin() as txn:
            if self.key_renamed:
                key = self.imgId2dbId[key]
            key = _bytes(key)
            raw_data = txn.get(key) #(key.tobytes()) #(b'00000000')
            assert raw_data is not None
            img = self.handle.unpack(raw_data)
        return img

    # new added.
    def __setitem__(self, key, npyarr):
        ''' key is usually imgID, normally we need key as bytes.'''
        assert self.mode in ['w','a+'], "Not open in write mode: %s " % self.mode
        key, value = self.handle.pack_from_npyarr(key, npyarr)
        with self.lmdb_env.begin(write=True) as txn:
            txn.put(key, value )

    def put(self, imgID, imgpath):
        key, value = self.handle.pack(imgID, imgpath)
        with self.lmdb_env.begin(write=True) as txn:
            txn.put(key, value)

    def get(self, key):
        """ return raw bytes buf."""
        with self.lmdb_env.begin() as txn:
            if self.key_renamed:
                key = self.imgId2dbId[key]
            key = _bytes(key)
            raw_data = txn.get(key)
            assert raw_data is not None
        return raw_data


    def vis(self, nr_row=5, nr_col=6, side=150, randshow=False, is_depth_img=False):  # nr_per_row=6,
        # Iter by keys
        keys = self.keys
        if randshow:
            random.shuffle(keys)
        itkeys = iter(keys)

        if True:
        # with self.lmdb_env.begin() as txn:
            row_images = []
            rows = []
            # for key, raw_data in txn.cursor():
            #     assert raw_data is not None
            #     pad_img = pad_as_squared_img(self.handle.unpack(raw_data), Side=side)
            # Iter by keys
            for key in itkeys:
                next_img = self[key]
                if is_depth_img:  # [TODO] TO REMOVE this case.
                    _max,_min = next_img.max(), next_img.min()
                    next_img = (next_img.astype(np.float32)-_min)/ (_max-_min)  # normalize [min,max] to [0,1]
                elif self.db_path.endswith('.Rawpng.lmdb') and self.handle.remap is not None:
                    _max,_min = self.handle.remap['max'], self.handle.remap['min']
                    next_img = (next_img-_min) / (_max-_min)
                pad_img = pad_as_squared_img(next_img, Side=side)
                if len(row_images)<nr_col:
                    row_images.append( pad_img )
                else:
                    cat_imgs = np.concatenate(row_images, axis=1)
                    row_images = [pad_img]  # reset  row_images

                    if len(rows)<nr_row:
                        rows.append(cat_imgs)
                    else:
                        cat_alls = np.concatenate(rows, axis=0)
                        cv2.imshow('images', cat_alls)
                        cv2_wait()
                        rows = [cat_imgs]    # reset rows


    def compute_mean_std(self, sample_k=None):
        import random
        random.seed(0)
        # Iter by keys
        keys = self.keys
        if sample_k is not None:
            keys = random.sample(keys, sample_k)

        C1 = 1e7
        cnt = 0
        sum_pxl = np.zeros((3,), np.float32)
        for i, key in enumerate(keys):
            data = self[key].astype(np.float32)/ C1 # 255.
            if len(data.shape)==3:
                h,w,c = data.shape
            else:
                (h,w),c = data.shape,1
            # print '--',np.sum(data.reshape(-1, c), axis=0)
            sum_pxl += np.sum(data.reshape(-1, c), axis=0)
            cnt += h*w
            if i %1000==0:
                print('\r [mean] %s / %s       ' % (i, len(keys))),
                sys.stdout.flush()
        pxl_mean = sum_pxl / cnt * C1  # [0,255]
        pxl_mean = np.array([2.19471788e-05, 2.19471788e-05, 2.19471788e-05], np.float32)*C1

        C2 = 1e8
        cnt = 0
        sum_var = np.zeros((3,), np.float32)
        for i, key in enumerate(keys):
            data = self[key].astype(np.float32) # / C # 255.
            if len(data.shape)==3:
                h,w,c = data.shape
            else:
                (h,w),c = data.shape,1
            sum_var += np.sum(((data.reshape(-1, c)-pxl_mean)**2)/C2, axis=0)
            cnt += h*w
            if i %1000==0:
                print ('\r [std]  %s / %s       ' % (i, len(keys))),
                sys.stdout.flush()
        pxl_std = np.sqrt(sum_var / (cnt-1) * C2)


        print ('pxl_mean: ', pxl_mean)  # [2.19471788e-05 2.19471788e-05 2.19471788e-05]
        print ('pxl_std:  ', pxl_std)
        return pxl_mean, pxl_std


def test_handle():
    # handle=Handle_Rawjpg()
    # handle = Handle_Npyarr()
    # test
    #-# handle=Handle_Rawjpg()
    #-# imgID   = 'n02690373_16'
    #-# imgpath = '/Users/shine/working/cvpr17pose/dataset/PASCAL3D/ImageData.Max500/aeroplane_imagenet/n02690373_16.jpg'

    handle=Handle_Rawpng()
    imgID   = '1507795919477055'
    imgpath = '/Users/shine/Pictures/1507795919477055.png'

    #
    k,v = handle.pack(imgID, imgpath)
    #
    cv2.imshow('image-pack', handle.unpack(v))
    cv2_wait()

    img_arr = cv2.imread(imgpath)
    k,v = handle.pack_from_npyarr(imgID, img_arr)
    #
    cv2.imshow('image-pack_from_npyarr', handle.unpack(v))
    cv2_wait()

    print (len(v))
    exit()

def test_lmdb():

    # lmdb_env = lmdb.open(os.path.join(PASCAL3D_Dir,'ImageData.Max500.Rawjpg.lmdb'), map_size=map_size)
    lmdb_env = lmdb.open(env.Home+'/working/cvpr17pose/dataset/PASCAL3D/ImageData.Max500.Rawjpg.lmdb', map_size=10000)
    handle   = Handle_Rawjpg()

    imgID = b'n03693474_19496'
    with lmdb_env.begin() as txn:
        raw_data = txn.get(imgID) #.tobytes()) #(b'00000000')

    cv2.imshow('image', handle.unpack(raw_data))
    cv2_wait()

    #print (len(v))
    #exit()



def test_img_lmdb():
    imgdb = ImageData_lmdb(env.Home+'/working/cvpr17pose/dataset/PASCAL3D/ImageData.Max500.Rawjpg.lmdb')
    #imgID = np.string_('n03693474_19496')            # np.string_
    #imgID = np.unicode_('n03693474_19496')           # np.string_
    #imgID = np.string_('n03693474_19496').tobytes()   # np.bytes_ ?
    imgID = 'n03693474_19496'                         # str
    #imgID = b'n03693474_19496'                       # bytes
    im = imgdb[imgID]
    cv2.imshow('image', im)
    cv2_wait()


def test_read_db1():
    imgdb = ImageData_lmdb(env.Home+'/working/cvpr17pose/dataset/PASCAL3D/ImageData.Max500.Rawjpg.lmdb')
    print(imgdb.keys[:10])
    for imgID in imgdb.keys:
        img = imgdb[imgID]
        cv2.imshow('img', img)
        cv2_wait()


def test_vis_lmdb(db_path):
    imgdb = ImageData_lmdb(db_path)
    imgdb.vis()


def test_compute():
    imgdb = ImageData_lmdb(env.Home+'/working/3DClassification/dataset/rendDB.cache/ModelNet10/MVCNN_12V.white/train.Rawjpg.lmdb', always_load_color=False)
    imgdb.compute_mean_std(sample_k=5000)
    # pxl_mean:  [219.47179 219.47179 219.47179]
    # pxl_std:   [58.2633316 58.2633316 58.2633316]


def show_lmdb():
    # optionally resume from a checkpoint
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('db_path'     , default='', type=str, metavar='PATH',)
    parser.add_argument('-f', '--format'    , default='5x6', type=str, help='rows and cols of display.')
    parser.add_argument('-s', '--side'      , default=150  , type=int, help='max side len of each image.')
    parser.add_argument('-r', '--random'    , action="store_true",  default=False,  help='random show entries in DB.')
    parser.add_argument('-d', '--depth'     , action="store_true",  default=False,  help='show depth image by normalize [min,max] to [0,1].')
    # --keep_last_only
    args = parser.parse_args()

    row, col = map(int, args.format.strip().split('x'))
    imgdb = ImageData_lmdb(args.db_path)
    if not args.depth and os.path.split(args.db_path.rstrip('/'))[1].lower().find('depth')>=0:
        if args.db_path.endswith('Rawpng.lmdb') and os.path.exists(os.path.join(args.db_path, 'remap.yml')):
            pass #
        else:
            use_depth_normalize = input('\nIs this depth image db? \nConsider visualize option "-d" or "--depth"?   Y/[N] ')
            if use_depth_normalize.upper()=='Y':
                args.depth = True
    imgdb.vis(row, col, args.side, args.random, args.depth)


if __name__ == '__main__':
    if len(sys.argv)>1:
        show_lmdb() # sys.argv[1]
    else:
        test_compute()
