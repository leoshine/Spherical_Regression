import os, sys
from basic.common import add_path,env
import numpy as np
from scipy.linalg import logm, norm
from math import pi, sqrt
from multiprocessing import Pool
from txt_table_v1 import TxtTable

def quaternion2rotMat(q):
    """Quternion to rotation Matrix"""
    a, b, c, d = q   # q = a + bi + cj + dk
    R = np.array([[ a**2 + b**2 - c**2 - d**2,       2*b*c - 2*a*d      ,       2*b*d + 2*a*c       ],
                  [       2*b*c + 2*a*d      , a**2 - b**2 + c**2 - d**2,       2*c*d - 2*a*b       ],
                  [       2*b*d - 2*a*c      ,       2*c*d + 2*a*b      , a**2 - b**2 - c**2 + d**2 ]], dtype=np.float64)
    return R



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
    # R_angle_results < pi/6.  is treated as correct in VpsKps



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
    return geodesic_dist_new(R, R_gt)
    # return geodesic_dist(R, R_gt)

def compute_geo_dists(GT_quats, Pred_quats):
    geo_dists= []
    for gt_q, pr_q in zip(GT_quats, Pred_quats):
        gt_R, pr_R = quaternion2rotMat(gt_q), quaternion2rotMat(pr_q)
        geo_dists.append( geodesic_dist_new(gt_R, pr_R) )
    return np.array(geo_dists)

def eval_one(_gt_quats, _pr_quats, theta_levels=[pi/6.]):
    geo_dists = []
    for _gt_quat, _pr_quat in zip(_gt_quats, _pr_quats):
        gt_R = quaternion2rotMat(_gt_quat)
        pr_R = quaternion2rotMat(_pr_quat)
        geo_dists.append(geodesic_dist_new(pr_R,gt_R))
    geo_dists = np.array(geo_dists)

    MedError = np.median(geo_dists) /pi*180.
    Acc_at_ts = [sum(geo_dists<theta_level)/float(len(geo_dists)) for theta_level in theta_levels]
    return MedError, Acc_at_ts, geo_dists


def eval_cates(rslt_txt_file, rc_tbl, cates=['aeroplane','boat','car'], theta_levels_str='pi/6'):
    theta_strs   = [theta_str for theta_str in theta_levels_str.split()]
    theta_values = [eval(theta_str) for theta_str in theta_strs] # ast.literal_eval cannot handle pi, np.pi, numpy.pi

    txtTbl = TxtTable()
    cols = txtTbl.load_as_recarr(rslt_txt_file, fields=['obj_id','a','b','c','d'])

    cate2eval = {}
    for cate in cates:
        select = (rc_tbl.recs['category']==cate)
        # print cols[select].a.shape # (B,)
        _pr_quats = np.vstack([cols[select].a, cols[select].b, cols[select].c, cols[select].d]).T
        _gt_quats = rc_tbl.recs[select].so3.quaternion
        #
        #print _pr_quat.shape, _gt_quat.shape
        #exit()
        MedError, Acc_at_ts, geo_dists = eval_one(_pr_quats, _gt_quats, theta_levels=theta_values) # eval string express: to convert to float number.
        # print cate, geo_dists
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
    # print summary_str
    return summary_str


