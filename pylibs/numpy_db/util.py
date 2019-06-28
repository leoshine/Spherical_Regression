"""
 @Author  : Shuai Liao
"""

import numpy as np

'''
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
# ...
data = load(stream, Loader=Loader)
# ...
output = dump(data, Dumper=Dumper)
'''
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from easydict import EasyDict as edict
#
class yaml_conf:
    @staticmethod
    def parse_yaml(conf_yaml_str):
        print ('---------------Parsing yaml------------------')
        _opt = yaml.load(conf_yaml_str, Loader=Loader) #
        opt = edict(_opt)
        return opt

    def write_yaml():
        # output = dump(data, Dumper=Dumper)
        raise NotImplementedError



def show_dtype(dtype_list, dtype_name=''):
    this_dt = np.dtype(dtype_list)

    name2dt_off = this_dt.fields
    name2dt_off = [(name,dt,off) for name,(dt,off) in name2dt_off.iteritems()]
    fields_list = sorted(name2dt_off, key=lambda x: x[2]) # fi, dt, offset

    print ('---------------------------------- %s' % dtype_name)
    print ('%-15s  %10s      %10s' % ('field_name', 'itemsize', 'offset')) # np.dtype(dt).itemsize
    print ('---------------------------------------------- ')
    for name, dt, offset in fields_list:
        print '%-15s  %10d      %10d' % (name, dt.itemsize, offset) # np.dtype(dt).itemsize
    print ('---------------------------------- Ttl size=', this_dt.itemsize)


def dtype_summery(dtype_list, dtype_name=''):

    from tabulate import tabulate
    this_dt = np.dtype(dtype_list)

    name2dt_off = this_dt.fields
    name2dt_off = [(name,dt.itemsize,off) for name,(dt,off) in name2dt_off.iteritems()]
    fields_list = sorted(name2dt_off, key=lambda x: x[2]) # ('field_name', 'itemsize', 'offset')
    # print tabulate(fields_list, headers=['field_name', 'itemsize', 'offset'], tablefmt='pipe')
    field_size = [(filedname,itemsize) for filedname, itemsize, offset in fields_list]
    print (tabulate(field_size, headers=['field_name', 'itemsize'], tablefmt='pipe'))


if __name__ == '__main__':
    from basic.common import env, add_path
    add_path(env.Home+'/working/cvpr18align/dataset/')
    from PASCAL3D import * # PASCAL3D_Dir, categories, category2nrmodel, get_anno

    dtype_summery(viewpoint  , 'viewpoint  ' )
    dtype_summery(proj_info  , 'proj_info  ' )
    dtype_summery(image_info , 'image_info ' )
    dtype_summery(object_anno, 'object_anno' )
    dtype_summery(pose_hypo  , 'pose_hypo  ' )

