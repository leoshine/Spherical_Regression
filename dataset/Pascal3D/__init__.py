import os, sys
import numpy as np
from collections import OrderedDict as odict

# from basic.common import env, add_path
# base_dir = os.path.dirname(os.path.realpath(__file__))

from .anno_db_v2.db_type import viewpoint, proj_info, image_info, object_anno, MAX_STRING_LEN #, pose_hypo
from .anno_db_v2.util_v2 import get_all_imgIDs , get_imgIDs, get_imgID2size    # image info.
from .anno_db_v2.util_v2 import get_anno_db_tbl, get_anno  , get_anno_dbs_tbl  # anno_db info.
from .anno_db_v2.util_v2 import rescale_anno, rescale_viewpoint, is_anno_of_org, recale_anno_to_org # tools for anno
from .data_info          import categories, category2nrmodel # , cadId2anchornames, cadId2nrAnchors


from numpy_db import npy_table, npy_db, dtype_summary, reorder_dtype

# from anno_db_v2.util_v2 import get_anno, get_anno_db_tbl, Rend2Img_Canvas, find_tightest_bbox, blendImgs
# get_anno(cate, collection="val", filter='easy', quiet=True)
# rend_canvas = Rend2Img_Canvas()
# rend_canvas.render(vp, cadId, image=image, near=0.1,far=100, visualize=True)








# read in anchors.
# PASCAL3D_dir = os.getenv("HOME") + "/working/cvpr17pose/dataset/PASCAL3D+_release1.1"
# PASCAL3D_dir = os.getenv("Home") + "/working/cvpr17pose/dataset/PASCAL3D"



# if __name__=="__main__":
#     category = 'aeroplane'
#     print ('category2nrmodel[category]', category2nrmodel[category])
#     print (category2anchors[category])
#     print ([(cate,len(category2anchors[cate])) for cate in categories])
