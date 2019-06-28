#!/bin/bash

#!/bin/bash
#-------------------------------------------------------------
# Useful one-liner which will give you the full directory name
# of the script no matter where it is being called from.
this_dir=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )  # get directory path to this script.
#-------------------------------------------------------------
# To expose the python lib directory of 'basic' to PYTHONPATH environment variable.
# Change it to yours, or just keep this directory structure.
PyLibs_path=$(dirname $this_dir)/pylibs  # here just happens to be this.
# Check if "basic" folder found.
if [ ! -d $PyLibs_path ]; then
  echo "[Error] python lib directory 'pylibs' is not found!  Abort!"
  exit
fi
echo "[Add PYTHONPATH]: $PyLibs_path"
export PYTHONPATH=$PyLibs_path
export PYTHONPATH=$PYTHONPATH:$OpenCV_INSTALL/lib/python3.5/site-packages:$OpenCV_INSTALL/lib/python3.7/site-packages  # just to add my python3 env.
#-------------------------------------------------------------

gpu_ids=$1
remain_args="${@:2}"


CUDA_VISIBLE_DEVICES=$gpu_ids  python trainval_workdir.py --gpu_ids=[$gpu_ids]  $remain_args


