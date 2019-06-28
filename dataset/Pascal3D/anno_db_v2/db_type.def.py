"""
 @Author  : Shuai Liao
"""

import numpy as np

MAX_STRING_LEN = 64

# struct
# Warning: the parameterization of viewpoint may not be correct (by using a,e,t to construct rotation matrix).
class viewpoint:
    #--[Extrinsic parameters]
    a         =     np.float64    # azimuth                     warning: this's new aet after adding displacement
    e         =     np.float64    # elevation                   warning: this's new aet after adding displacement
    t         =     np.float64    # theta (in-plane rotation )  warning: this's new aet after adding displacement
    d         =     np.float64    # camera distance.
    #--[Intrinsic parameters]
    px        =     np.float64    # principle offset x
    py        =     np.float64    # principle offset y
    f         =     np.float64    # focal length
    mx        =     np.float64    # viewport: nr of pixel per unit in image coordinate in x direction.
    my        =     np.float64    # viewport: nr of pixel per unit in image coordinate in y direction.

# struct
class proj_info:
    obj_id        =  str        # <primary key>
    cad_id        =  str
    #---- some object annotation is not able to project.
    valid         =  np.bool
    #---- Rotation matrix
    R             =  np.float64    ,   (3,3)           # 3x3

#-------------------------------------------------------------------------------
# -------- Extra info. ---------
# base_dir = env.Home + '/working/cvpr17pose/dataset/'+dataset
# imgID2path = ....
# ------------------------------
class image_info:
    image_id  =  str        # <primary key>
    #--- original image size.
    H         =  int        # (not the actually image size, but with maxium side 500.)
    W         =  int
    C         =  int        # image channel
    #--- the actual image size the annotation is based on (with resizing by ratio).
    h         =  int
    w         =  int


# [Table] for storing object annotation for Pascal3D+/Objectnet3D
class object_anno:
    obj_id    =     str    # <primary key>  format: obj_id  = {image_id}-{x1},{y1},{x2},{y2}  e.g. 2008_000251-24,14,410,245
    #---------------------
    category  =     str    # e.g. aeroplane, car, bike
    cad_id    =     str    # e.g. aeroplane01  {category}{cad_idx}
    bbox      =     np.float64    ,  4     # xmin,ymin,xmax,ymax  (float is because of scaling)
    #----- source image info. (src_img.ratio indicate what size of image the anno based on.)
    src_img   =     image_info
    #----- viewpoint (camera model parameters)
    gt_view   =     viewpoint
    #----- projection infor.
    # proj    =     proj_info              # not in use for this project.
    #----- Other annotation.
    difficult =     int
    truncated =     bool
    occluded  =     bool



if __name__=="__main__":
    import os, sys
    from basic.common import env, add_path
    add_path(env.Home+'/WorkGallery/pytools/numpy_db')
    from formater import get_code_blocks, block2list_code

    this_script = os.path.basename(__file__)
    assert this_script.endswith('.def.py')
    out_script = this_script.replace('.def.py', '.py')  # db_type.def.py  -->  db_type.py

    code_blocks = get_code_blocks(open(this_script).read())
    open(out_script,'w').write('\n\n'.join( map(block2list_code, code_blocks[:-1])) )
     # the last code_block is this  'if __name__=="__main__":'

