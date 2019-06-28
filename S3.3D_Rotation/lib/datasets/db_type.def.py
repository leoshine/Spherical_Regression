import numpy as np

# N_ANCHOR = 22
# MAX_PATH_LEN # 128
MAX_STRING_LEN = 64


# struct
class axisAngle:
    axis      =     np.float32     ,   (3,) # unit 3d vector
    angle     =     np.float32              # in degree
#
# class S2xS1:
#     S2        =     np.float32     ,   (3,) # unit 3d vector
#     S1        =     np.float32     ,   (2,) # unit 2d vector, same as the in-plane rotation in pascal3d: but in form of (cos_t, sin_t)
#

# struct
class rot3d:
    #---- Rotation matrix
    R             =  np.float32    ,   (3,3)           # 3x3
    #---- Axis-Angle representation
    axis_angle    =  axisAngle
    #---- Quaternion representation
    quaternion    =  np.float32    ,   (4,)        # unit quternion
    #---- S2xS1      representation
    # s2xs1         =  S2xS1
    #----
    euler         =  np.float32    ,   (3,)


# [Table] for storing object annotation for Pascal3D+/Objectnet3D
class img_view_anno:
    img_id    =     str    # <primary key>  format: obj_id  = {cad_id}-v{viewInd}  e.g. 2008_000251-24,14,410,245
    #---------------------
    category  =     str    # e.g. aeroplane, car, bike
    cad_id    =     str    # e.g. aeroplane01  {category}{cad_idx}
    #----- viewpoint (camera model parameters)
    so3       =     rot3d



if __name__=="__main__":
    import os, sys
    from basic.common import env, add_path
    add_path(env.Home+'/WorkGallery/pytools/numpy_db')
    from formater import get_code_blocks, block2list_code
    #
    this_script = os.path.basename(__file__)
    assert this_script.endswith('.def.py')
    out_script = this_script.replace('.def.py', '.py')  # db_type.def.py  -->  db_type.py
    #
    code_blocks = get_code_blocks(open(this_script).read())
    open(out_script,'w').write('\n\n'.join( map(block2list_code, code_blocks[:-1])) )
     # the last code_block is this  'if __name__=="__main__":'

