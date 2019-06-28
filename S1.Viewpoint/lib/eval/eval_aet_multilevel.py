"""
 @Author  : Shuai Liao
"""

import os, sys
from basic.common import add_path,env
import numpy as np
from scipy.linalg import logm, norm
from math import pi, sqrt
from multiprocessing import Pool
from txt_table_v1 import TxtTable

# add_path(env.Home+'/working/eccv18varpose/dataset')
# from PASCAL3D import get_anno_dbs_tbl, get_anno, categories

this_dir = os.path.dirname(os.path.realpath(__file__))
add_path(this_dir+'/../../../dataset')
from Pascal3D import get_anno_dbs_tbl, get_anno, categories


def compute_RotMats(a, e, t):
    """
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
    #                                          Warning from Shuai                                             #
    #                                                                                                         #
    #   This function is just a replication of matlab implementation for reproducibility purpose only!        #
    #   However, I believe the logic is not correct. But since Pascal3D+ dataset itself is annotated          #
    #   in such way, we have to follow this definition for evaluation purpose.                                #
    #                                                                                                         #
    #   In short words: The resulting rotation matrix can still be valid since it guarantees the CAD model    #
    #   to be projected roughly aligned with the 2D object in image. However, the way in interpreting         #
    #   a, e, t used in this function to construct the rotation matrix is deviated from the true definition   #
    #   of Azimuth, Elevation and In-plane rotation.                                                          #
    #                                                                                                         #
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
    """
    assert len(a)==len(e)==len(t)
    M = len(a)

    # camera intrinsic matrix
    Rz  = np.zeros((M, 3, 3), dtype=np.float32)
    Rx  = np.zeros((M, 3, 3), dtype=np.float32)
    Rz2 = np.zeros((M, 3, 3), dtype=np.float32)
    # C   = np.zeros((M, 1, 3), dtype=np.float32)
    # initial "1" positions.
    Rz [:, 2, 2] = 1
    Rx [:, 0, 0] = 1
    Rz2[:, 2, 2] = 1
    #
    R  = np.zeros((M, 3, 3), dtype=np.float32)

    # convert to radius
    a = a * pi / 180.
    e = e * pi / 180.
    t = t * pi / 180.

    # update a, e, t
    a = -a
    e = pi/2.+e
    t = -t
    #
    sin_a, cos_a = np.sin(a), np.cos(a)
    sin_e, cos_e = np.sin(e), np.cos(e)
    sin_t, cos_t = np.sin(t), np.cos(t)

    # ===========================
    #   rotation matrix
    # ===========================
    """
      # [Transposed]
      Rz = np.matrix( [[  cos(a), sin(a),       0 ],     # model rotate by a
                       [ -sin(a), cos(a),       0 ],
                       [      0,       0,       1 ]] )
      # [Transposed]
      Rx = np.matrix( [[      1,       0,       0 ],    # model rotate by e
                       [      0,   cos(e), sin(e) ],
                       [      0,  -sin(e), cos(e) ]] )
      # [Transposed]
      Rz2= np.matrix( [[ cos(t),   sin(t),      0 ],     # camera rotate by t (in-plane rotation)
                       [-sin(t),   cos(t),      0 ],
                       [      0,        0,      1 ]] )
      R = Rz2*Rx*Rz
    """

    # Original matrix (None-transposed.)
    # No need to set back to zero?
    Rz[:, 0, 0],  Rz[:, 0, 1]  =  cos_a, -sin_a
    Rz[:, 1, 0],  Rz[:, 1, 1]  =  sin_a,  cos_a
    #
    Rx[:, 1, 1],  Rx[:, 1, 2]  =  cos_e, -sin_e
    Rx[:, 2, 1],  Rx[:, 2, 2]  =  sin_e,  cos_e
    #
    Rz2[:, 0, 0], Rz2[:, 0, 1] =  cos_t, -sin_t
    Rz2[:, 1, 0], Rz2[:, 1, 1] =  sin_t,  cos_t
    # R = Rz2*Rx*Rz
    R[:] = np.einsum("nij,njk,nkl->nil", Rz2, Rx, Rz)

    # Return the original matrix without transpose!
    return R



#-# def geodesic_dist(R, R_gt):  # _geo_err
#-#     R, R_gt = map(np.matrix, [R, R_gt])
#-#     R_angle = norm(logm(R.transpose()*R_gt), 2) / sqrt(2)
#-#     # About different of numpy/scipy norm and matlab norm:
#-#     #  http://stackoverflow.com/questions/26680412/getting-different-answers-with-matlab-and-python-norm-functions
#-#     #  https://nl.mathworks.com/help/matlab/ref/norm.html
#-#     return R_angle   # R_angle_results < pi/6.  is treated as correct in VpsKps


def geodesic_dist(R, R_gt):  # _geo_err
    R, R_gt = map(np.matrix, [R, R_gt])
    # With out disp annoying error
    _logRR, errest = logm(R.transpose()*R_gt, disp=False)
    R_angle  = norm(_logRR, 2) / sqrt(2)
    # This will do print("logm result may be inaccurate, approximate err =", errest)
    # R_angle  = norm(logm(R.transpose()*R_gt), 2) / sqrt(2)
    #
    # About different of numpy/scipy norm and matlab norm:
    #  http://stackoverflow.com/questions/26680412/getting-different-answers-with-matlab-and-python-norm-functions
    #  https://nl.mathworks.com/help/matlab/ref/norm.html
    return R_angle

def geodesic_dist_new(R, R_gt):  # _geo_err
    '''ICCV17, From 3D Pose Regression using Convolutional Neural Networks.
        Note: the geodesic distance used by vpskps: d(R1, R2)
              the simplified version by this paper: d_A(R1, R2)
            Their relation is:  d(R1, R2) = d_A(R1, R2) / sqrt(2)
    '''
    R, R_gt = map(np.matrix, [R, R_gt])
    # Do clipping to [-1,1].
    # For a few cases, (tr(R)-1)/2 can be a little bit less/greater than -1/1.
    logR_F = np.clip( (np.trace(R.transpose()*R_gt)-1.)/2., -1, 1)
    R_angle = np.arccos( logR_F ) / np.sqrt(2)
    # This can return nan when inside is out of range [-1,1]
    # R_angle = np.arccos( (np.trace(R.transpose()*R_gt)-1.)/2. ) / np.sqrt(2)
    return R_angle

def _geodesic_dist(args):
    R, R_gt = args
    return geodesic_dist(R, R_gt)


def compute_geo_dists(GT_aet, Pred_aet):
    geo_dists= []
    gt_As, gt_Es, gt_Ts = GT_aet
    pr_As, pr_Es, pr_Ts = Pred_aet

    gt_Rs = compute_RotMats(gt_As, gt_Es, gt_Ts)
    pr_Rs = compute_RotMats(pr_As, pr_Es, pr_Ts)
    # for gt_a, gt_e, gt_t,  pr_a, pr_e, pr_t in zip(gt_As, gt_Es, gt_Ts,  pr_As, pr_Es, pr_Ts):
    for gt_R, pr_R in zip(gt_Rs,  pr_Rs):
        geo_dists.append( geodesic_dist_new(gt_R, pr_R) )
    return np.array(geo_dists)


def parse_rslt_txt(rslt_txt_file):
    lines = [x.strip() for x in open(rslt_txt_file).readlines() if not x.strip().startswith('#')]
    objID2aet = {}
    for line in lines:
        lineSp = line.split()
        objID = lineSp[0]
        a,e,t = map(float, lineSp[1:])
        objID2aet[objID] = (a,e,t)
    return objID2aet



def eval_one(objID2aet_pred, cate='aeroplane', theta_levels=[pi/6.], nr_worker=20):
    # objID2aet_pred = parse_rslt_txt(rslt_txt_file)

    keys, rcobjs = get_anno(cate, collection='val', filter='easy')
    # print('--->[eval_one] %s  '%cate, len(keys))
    vps = rcobjs.gt_view
    gt_rot_Mats = compute_RotMats(vps.a, vps.e, vps.t)

    a_preds, e_preds, t_preds = [],[],[]
    for rcobj in rcobjs:
        _a,_e,_t = objID2aet_pred[rcobj.obj_id]
        a_preds.append(_a)
        e_preds.append(_e)
        t_preds.append(_t)

    a_preds = np.array(a_preds, np.float32)
    e_preds = np.array(e_preds, np.float32)
    t_preds = np.array(t_preds, np.float32)
    pred_rot_Mats = compute_RotMats(a_preds, e_preds, t_preds)

    # pool = Pool(nr_worker)
    # geo_dists = pool.map(_geodesic_dist, zip(pred_rot_Mats,gt_rot_Mats))
    geo_dists = []
    for pr_R, gt_R in zip(pred_rot_Mats,gt_rot_Mats):
        geo_dists.append(geodesic_dist_new(pr_R,gt_R))
    #
    geo_dists = np.array(geo_dists)
    #
    MedError = np.median(geo_dists) / pi*180.
    Acc_at_ts = [sum(geo_dists<theta_level)/float(len(keys)) for theta_level in theta_levels]
    return MedError, Acc_at_ts


def eval_cates(rslt_file, cates=['aeroplane','boat','car'], theta_levels_str='pi/6.'):

    theta_strs   = [theta_str for theta_str in theta_levels_str.split()]
    theta_values = [eval(theta_str) for theta_str in theta_strs]

    objID2aet_pred = parse_rslt_txt(rslt_file)

    cate2eval = {}
    for cate in cates:
        MedError, Acc_at_ts = eval_one(objID2aet_pred, cate=cate, theta_levels=theta_values) # eval string express: to convert to float number.
        cate2eval[cate] = (MedError, Acc_at_ts)


    #-- Write result to file  (Format: # {obj_id}  {a} {e} {t} )
    # txtTbl = TxtTable('{cate:<20s}   {MedError:>6.3f}   {Acc@pi/6:>6.3f}   {Acc@pi/12:>6.3f}  {Acc@pi/24:>6.3f}')
    tb_format = '{cate:<15s}   {MedError:>6.3f}   ' + ''.join('{Acc@%s:>14.3f}' % x for x in theta_strs)
    txtTbl = TxtTable(tb_format)
    rslt_lines = [ txtTbl.getHeader() ]
    list_MedError = []
    theta_level2list_Acc_at_t = {}
    for cate in cates:
        MedError, Acc_at_ts = cate2eval[cate]
        rslt_lines.append( txtTbl.format(cate, MedError, *['%10.3f'%(Acc_at_t*100) for Acc_at_t in Acc_at_ts] ) )
        list_MedError.append(MedError)
        for theta_level, acc_at_t in zip(theta_strs, Acc_at_ts):
            theta_level2list_Acc_at_t.setdefault(theta_level, []).append(acc_at_t)
    rslt_lines.append( txtTbl.format('MEAN', np.mean(list_MedError),
                        *['%10.3f' % (np.mean(theta_level2list_Acc_at_t[theta_level])*100) for theta_level in theta_strs] ) )
    summary_str =  '\n'.join(rslt_lines)+'\n'
    return summary_str


if __name__ == '__main__':
    pass
