#!/bin/bash

# TODO: add PyLib path.
PyLibs_path=$(cd ../../..;pwd)/pylibs   # here just happens to be this.
# Check if "basic" folder found.
if [ ! -d $PyLibs_path ]; then
  echo "[Error] python lib directory 'pylibs' is not found!  Abort!"
  exit
fi
echo "[Add PYTHONPATH]: $PyLibs_path"

export PYTHONPATH=$PyLibs_path:$PYTHONPATH

# Dump matlab annotation to python pickle
python mat2py.py

# First organize per category object annotation records, for pascal-voc part and imagenet part.
# Then build a annotation table that merges the two parts.
python prepare_anno_db.py


# build up image lmdb.
python build_imdb.py


# generate ground truth box based on the actual image size.
python gen_gt_box.py
