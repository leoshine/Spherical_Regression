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


python prepare_r4cnn_syn_data.py
