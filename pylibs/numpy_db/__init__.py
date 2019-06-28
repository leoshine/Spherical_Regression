"""
 @Author  : Shuai Liao
"""

from basic.common import is_py3

if is_py3: # python3
    from .numpy_db_py3 import npy_table, npy_db, dtype_summary, reorder_dtype
else:      # python2
    from .numpy_db     import npy_table, npy_db, dtype_summary, reorder_dtype
