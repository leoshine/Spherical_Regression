"""
 @Author  : Shuai Liao
"""
import numpy as np
from Dataset_Base import Dataset_Base, netcfg

import torch
from torch.utils.data import Dataset, DataLoader

import cv2
from basic.common import add_path, env, rdict, cv2_wait, cv2_putText

#===============================================================
def pred2angle(a, e, t):
    return (a*180./np.pi) % 360 ,  e*180./np.pi,  t*180./np.pi

def pred2angle_shift45(a, e, t):
    # shift 45 degree back
    return  (a*180./np.pi -45) % 360 ,  e*180./np.pi,  t*180./np.pi



class Dataset_regQuatNet(Dataset_Base):
    def __init__(self, *args, **kwargs):
        Dataset_Base.__init__(self, *args, **kwargs)

    def __getitem__(self, idx):
        rc     = self.recs[idx]
        cate   = rc.category
        quat   = rc.so3.quaternion
        sample = dict( idx   = idx,
                       label = self.cate2ind[cate],
                       quat  = quat,
                       data  = self._get_image(rc) )
        return sample

# build class alias
Dataset_reg_Direct=Dataset_regQuatNet
Dataset_reg_Sexp  =Dataset_regQuatNet
Dataset_reg_Sflat =Dataset_regQuatNet


if __name__ == '__main__':
    np.random.seed(3)

    def test_dataloader(collection='test', sampling=0.2):
        dataset = Dataset_reg_Sexp(collection=collection, sampling=sampling)
        print (len(dataset.keys))

        anno_path = dataset.db_path
        sampling_file =anno_path+'/%s_sampling%.2f.txt' % (collection, sampling)
        with open(sampling_file, 'w') as f:
            f.write('\n'.join(dataset.keys))
        print ('sampling_file:', sampling_file)
        exit()
        #
        dataloader = DataLoader(dataset, batch_size=50,
                                shuffle=False, num_workers=1, sampler=None)

        for i_batch, sample_batched in enumerate(dataloader):
            dataset._vis_minibatch(sample_batched)

