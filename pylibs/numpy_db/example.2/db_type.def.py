import numpy as np

# N_ANCHOR = 22
# MAX_PATH_LEN # 128
MAX_STRING_LEN = 64

# struct
# Warning: the parameterization of viewpoint may not be correct (by using a,e,t to construct rotation matrix).
class viewpoint:
    #--[Extrinsic parameters]
    a         =     np.float32    # azimuth                     warning: this's new aet after adding displacement
    e         =     np.float32    # elevation                   warning: this's new aet after adding displacement
    t         =     np.float32    # theta (in-plane rotation )  warning: this's new aet after adding displacement
    d         =     np.float32    # camera distance.
    #--[Intrinsic parameters]
    px        =     np.float32    # principle offset x
    py        =     np.float32    # principle offset y
    f         =     np.float32    # focal length
    mx        =     np.float32    # viewport: nr of pixel per unit in image coordinate in x direction.
    my        =     np.float32    # viewport: nr of pixel per unit in image coordinate in y direction.

# struct
class proj_info:
    valid         =  np.bool     # some object annotation is not able to project.
    #---- Rotation matrix
    R             =  np.float32    ,   (3,3)           # 3x3
    # anchor_cam3d  =  np.float32    ,   (N_ANCHOR,3)    # N*3    projected 3d location of anchor in camera space.
    # anchor_img2d  =  np.float32    ,   (N_ANCHOR,2)    # N*2    projected 2d location of anchor on image space.
    # anchor_visib  =  np.int8       ,   (N_ANCHOR,)     # N*1    visibility

#-------------------------------------------------------------------------------
class image_info:
    image_id  =  str        # <primary key>
    # path      =  str
    #--- original image size.
    H         =  int        # (not the actually image size, but with maxium side 500.)
    W         =  int
    C         =  int        # image channel
    #--- the actual image size the annotation is based on (with resizing by ratio).
    # ratio     =  float      # H=int(h*ratio), W=int(w*ratio)
    h         =  int
    w         =  int


# [Table] for storing object annotation for Pascal3D+/Objectnet3D
class object_anno:
    obj_id    =     str    # <primary key>  format: obj_id  = {image_id}-{x1},{y1},{x2},{y2}  e.g. 2008_000251-24,14,410,245
    #---------------------
    category  =     str    # e.g. aeroplane, car, bike
    cad_id    =     str    # e.g. aeroplane01  {category}{cad_idx}
    # nr_anchor =     int
    bbox      =     np.float32    ,  4     # xmin,ymin,xmax,ymax  (float is because of scaling)
    #----- source image info. (src_img.ratio indicate what size of image the anno based on.)
    src_img   =     image_info
    #----- viewpoint (camera model parameters)
    gt_view   =     viewpoint
    # #----- projection info.
    # proj      =     proj_info
    #==Other annotation.
    difficult =     int
    truncated =     bool
    occluded  =     bool


# [Table] that feed into dataengine.
class pose_hypo:   # hypothesis pose
    pose_hypo_id  =  str            # <primary key>   format:   pose_id = {obj_id}_{a},{e},{t}     e.g. 2008_000251-24,14,410,245_144,0,2
    pose_gt_id    =  str            # point to gt inside this table (for convenience)
    obj_id        =  str            # <foreign key>
    #----- view param and deviation.
    hypo_view     =  viewpoint
    delta_view    =  viewpoint      # hypo_view = gt_view + delta_view
    #----- projection info.
    proj          =  proj_info
    # #----- Flow array (Differece w.r.t ground truth pose)
    # anchor_flow3d =  np.float32    ,   (N_ANCHOR,3)   # Flow of 'anchor_prj2d' between this and gt
    # anchor_flow2d =  np.float32    ,   (N_ANCHOR,2)   # Flow of 'anchor_prj2d' between this and gt
    #----- Distance (scalar)   if dist=0 --> gt
    dist_geo      =  np.float32                       # Geodesic distance to GT po
    # dist_flow3d   =  np.float32                       # Sum of flow3d differences to GT pose.
    # dist_flow2d   =  np.float32                       # Sum of flow2d differences to GT pose.
    # #--------------------------Obj info.
    # rendered_path =  str                              # Warning: string should less than MAX_PATH_LEN
    # pnccimg_path  =  str                              # Warning: string should less than MAX_PATH_LEN
    # fused_path    =  str                              # Warning: string should less than MAX_PATH_LEN
    # roiimg_path   =  str                              # Warning: string should less than MAX_PATH_LEN



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

