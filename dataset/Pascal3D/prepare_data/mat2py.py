import os, sys
from basic.common import Open
from load_nested_mat_struct import loadmat
import pickle

from basic.util import load_yaml

conf = load_yaml('config.yml')  # odict
Pascal3D_root = os.path.expanduser(conf['Pascal3D_release_root'])
protocol      = conf['pkl_protocol']  # pickle dump protocol. Change -1 to 2 for python2.x compatibility.

mat_anno_dir = os.path.join(Pascal3D_root,  'Annotations')
new_anno_dir  = os.path.join('./working_dump.cache', 'Imgwise_Annotations.py')
try:
    os.makedirs(new_anno_dir)
except:
    pass

all_fos = [x for x in os.listdir(mat_anno_dir) if os.path.isdir(os.path.join(mat_anno_dir,x))]
for i, fo in enumerate(all_fos):
    print('[%2d/%2d]  %s   ' %(i, len(all_fos), fo))
    for f in [x for x in os.listdir(os.path.join(mat_anno_dir,fo)) if x.endswith('.mat')]:
        matfile = os.path.join(mat_anno_dir, fo, f)
        pklfile = os.path.join(new_anno_dir, fo, f[:-4]+'.pkl')
        struct_dict = loadmat(matfile)
        pickle.dump(struct_dict, Open(pklfile,'wb'), protocol)

