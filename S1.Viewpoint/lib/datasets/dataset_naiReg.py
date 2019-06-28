"""
 @Author  : Shuai Liao
"""

import numpy as np
from .Dataset_Base import Dataset_Base, netcfg


#===============================================================
def pred2angle(a, e, t):
    return (a*180./np.pi) % 360 ,  e*180./np.pi,  t*180./np.pi
    # return (a*180./np.pi + 360) % 360 ,  e*180./np.pi,  t*180./np.pi

def pred2angle_shift45(a, e, t):
    # return (a*180./np.pi) % 360 ,  e*180./np.pi,  t*180./np.pi
    # shift 45 degree back
    return  (a*180./np.pi -45) % 360 ,  e*180./np.pi,  t*180./np.pi
    # return (a*180./np.pi + 360) % 360 ,  e*180./np.pi,  t*180./np.pi


def quat_pred2angle(a, e, t):
    """only for quaternion prediction."""
    return a, e, t # a,e,t already in degree and in original convention.

class Dataset_reg1D(Dataset_Base):
    # Implement interface method
    def __getitem__(self, idx):
        rcobj    = self.recs[idx]
        cate     = rcobj.category
        obj_id   = rcobj.obj_id
        image_id = rcobj.src_img.image_id

        # In degree
        a = rcobj.gt_view.a # * np.pi/180.
        e = rcobj.gt_view.e # * np.pi/180.
        t = rcobj.gt_view.t # * np.pi/180.
        # Re-Mapping a: [0,180]->[180,360]->[0,180]    [180,360]->[0,180]->[-180,0]
        a = (a+180.) % 360. - 180.
        # To radius
        a = (a * np.pi/180.).astype(np.float32)
        e = (e * np.pi/180.).astype(np.float32)
        t = (t * np.pi/180.).astype(np.float32)

        sample = dict( idx   = idx,
                       label = self.cate2ind[cate],
                       a =a,  # cosine
                       e =e,
                       t =t,
                       data  = self._get_image(rcobj) )  # sub_trs(self.datadb[obj_id]) )
        return sample

    @staticmethod
    def pred2angle(a, e, t):
        return pred2angle(a, e, t)


class Dataset_reg2D(Dataset_Base):
    # Implement interface method
    def __getitem__(self, idx):
        rcobj    = self.recs[idx]
        cate     = rcobj.category
        obj_id   = rcobj.obj_id
        image_id = rcobj.src_img.image_id

        # In degree
        a = rcobj.gt_view.a # * np.pi/180.
        e = rcobj.gt_view.e # * np.pi/180.
        t = rcobj.gt_view.t # * np.pi/180.
        # Re-Mapping a: [0,180]->[180,360]->[0,180]    [180,360]->[0,180]->[-180,0]
        a = (a+180.) % 360. - 180.
        # To radius
        a = (a * np.pi/180.).astype(np.float32)
        e = (e * np.pi/180.).astype(np.float32)
        t = (t * np.pi/180.).astype(np.float32)

        sample = dict( idx   = idx,
                       label = self.cate2ind[cate],
                       cos_a = np.cos(a),  # cosine
                       cos_e = np.cos(e),
                       cos_t = np.cos(t),
                       sin_a = np.sin(a),  # sine
                       sin_e = np.sin(e),
                       sin_t = np.sin(t),
                       data  = self._get_image(rcobj) )  # sub_trs(self.datadb[obj_id]) )
        return sample

    @staticmethod
    def pred2angle(a, e, t):
        return pred2angle(a, e, t)


class Dataset_reg3D(Dataset_Base):
    # Implement interface method
    def __getitem__(self, idx):
        rcobj = self.recs[idx]
        cate = rcobj.category
        obj_id = rcobj.obj_id

        # In degree
        a = rcobj.gt_view.a # * np.pi/180.
        e = rcobj.gt_view.e # * np.pi/180.
        t = rcobj.gt_view.t # * np.pi/180.
        # Re-Mapping a: [0,180]->[180,360]->[0,180]    [180,360]->[0,180]->[-180,0]
        a = (a+180.) % 360. - 180.
        # To radius
        a = (a * np.pi/180.).astype(np.float32)
        e = (e * np.pi/180.).astype(np.float32)
        t = (t * np.pi/180.).astype(np.float32)

        sample = dict( idx    = idx,
                       label  = self.cate2ind[cate],
                       cos_a1 = np.float32( np.cos(a-np.pi/3) ),  # cosine
                       cos_e1 = np.float32( np.cos(e-np.pi/3) ),
                       cos_t1 = np.float32( np.cos(t-np.pi/3) ),
                       cos_a2 = np.float32( np.cos(a        ) ),
                       cos_e2 = np.float32( np.cos(e        ) ),
                       cos_t2 = np.float32( np.cos(t        ) ),
                       cos_a3 = np.float32( np.cos(a+np.pi/3) ),
                       cos_e3 = np.float32( np.cos(e+np.pi/3) ),
                       cos_t3 = np.float32( np.cos(t+np.pi/3) ),
                       data   = self._get_image(rcobj)         )
        return sample

    @staticmethod
    def pred2angle(a, e, t):
        return pred2angle(a, e, t)


class Dataset_regS1xy(Dataset_Base):
    def __getitem__(self, idx):
        rcobj = self.recs[idx]
        cate = rcobj.category
        obj_id = rcobj.obj_id

        # In degree
        a = rcobj.gt_view.a # * np.pi/180.
        e = rcobj.gt_view.e # * np.pi/180.
        t = rcobj.gt_view.t # * np.pi/180.
        # Re-Mapping a: [0,180]->[180,360]->[0,180]    [180,360]->[0,180]->[-180,0]
        a = (a+180.) % 360. - 180.
        # To radius
        a = (a * np.pi/180.).astype(np.float32)
        e = (e * np.pi/180.).astype(np.float32)
        t = (t * np.pi/180.).astype(np.float32)

        sample = dict( idx   = idx,
                       label = self.cate2ind[cate],
                       a     = a,
                       e     = e,
                       t     = t,
                       data   = self._get_image(rcobj)         )
        return sample

    @staticmethod
    def pred2angle(a, e, t):
        return pred2angle(a, e, t)


class Dataset_regQuat(Dataset_Base):
    def __init__(self, *args, **kwargs):
        Dataset_Base.__init__(self, *args, **kwargs)

        from basic.common import add_path, env
        add_path(env.Home+'/working/eccv18varpose/code/prepare_data/rotationAnno/')
        from prepare_rot3d_anno import get_rotAnno_tbls

        self.rotAnno_tb = get_rotAnno_tbls(self.cates, collection=self.collection, filter='all', quiet=True)


    def __getitem__(self, idx):
        rcobj = self.recs[idx]
        cate = rcobj.category
        obj_id = rcobj.obj_id

        quat = self.rotAnno_tb[obj_id].quaternion

        sample = dict( idx   = idx,
                       label = self.cate2ind[cate],
                       quat  = quat,
                       data  = self._get_image(rcobj)         )
        return sample

    @staticmethod
    def pred2angle(a, e, t):
        return quat_pred2angle(a, e, t)




class Dataset_regSquaredProb(Dataset_Base):
    # Implement interface method
    def __getitem__(self, idx):
        rcobj    = self.recs[idx]
        cate     = rcobj.category
        obj_id   = rcobj.obj_id
        image_id = rcobj.src_img.image_id

        # In degree
        a = rcobj.gt_view.a # * np.pi/180.
        e = rcobj.gt_view.e # * np.pi/180.
        t = rcobj.gt_view.t # * np.pi/180.
        # shift 45 degree
        a,e,t = a+45,e,t
        # Re-Map a: [0,180]->[180,360]->[0,180]    [180,360]->[0,180]->[-180,0]
        a = (a+180.) % 360. - 180.
        # To radius
        a = (a * np.pi/180.).astype(np.float32)
        e = (e * np.pi/180.).astype(np.float32)
        t = (t * np.pi/180.).astype(np.float32)

        cos_sin_a = np.array([np.cos(a), np.sin(a)], np.float32)
        cos_sin_e = np.array([np.cos(e), np.sin(e)], np.float32)
        cos_sin_t = np.array([np.cos(t), np.sin(t)], np.float32)

        signs2label = {(1,1):0, (-1,1):1, (-1,-1):2, (1,-1):3,     (1,0):0, (0,1):0, (-1,0):1, (0,-1):3}
        sign_a = signs2label[tuple(np.sign( cos_sin_a ).astype(np.int32).tolist())]
        sign_e = signs2label[tuple(np.sign( cos_sin_e ).astype(np.int32).tolist())]
        sign_t = signs2label[tuple(np.sign( cos_sin_t ).astype(np.int32).tolist())]

        sample = dict( idx    = idx,
                       label  = self.cate2ind[cate],
                       ccss_a = cos_sin_a**2,  # cosine
                       ccss_e = cos_sin_e**2,
                       ccss_t = cos_sin_t**2,
                       sign_a = np.int(sign_a), # quadrant_label
                       sign_e = np.int(sign_e), # quadrant_label
                       sign_t = np.int(sign_t), # quadrant_label
                       data  = self._get_image(rcobj) )  # sub_trs(self.datadb[obj_id]) )
        return sample

    @staticmethod
    def pred2angle(a, e, t):
        return pred2angle_shift45(a, e, t)


# XP version share the same dataset class
Dataset_regQuatXP=Dataset_regQuat
Dataset_regQuatSXP=Dataset_regQuat
Dataset_reg2DXP=Dataset_reg2D
Dataset_reg2DSXP=Dataset_reg2D

Dataset_regSquaredProbV2=Dataset_regSquaredProb

#
Dataset_regSquaredProbV2_bigFC8=Dataset_regSquaredProb
Dataset_regSquaredProbV2_FC8FC9=Dataset_regSquaredProb


Dataset_reg1D_bigFC8=Dataset_reg1D
Dataset_reg2D_bigFC8=Dataset_reg2D
Dataset_reg3D_bigFC8=Dataset_reg3D
