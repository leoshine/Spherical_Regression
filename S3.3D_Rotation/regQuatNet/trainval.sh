#!/bin/bash
#-------------------------------------------------------------
# Useful one-liner which will give you the full directory name
# of the script no matter where it is being called from.
this_dir=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )  # get directory path to this script.
#-------------------------------------------------------------
# To expose the python lib directory of 'basic' to PYTHONPATH environment variable.
# Change it to yours, or just keep this directory structure.
PyLibs_path=$(dirname $(dirname $this_dir))/pylibs  # here just happens to be this.
# Check if "basic" folder found.
if [ ! -d $PyLibs_path ]; then
  echo "[Error] python lib directory 'pylibs' is not found!  Abort!"
  exit
fi
echo "[Add PYTHONPATH]: $PyLibs_path"
export PYTHONPATH=$PyLibs_path
export PYTHONPATH=$PYTHONPATH:$OpenCV_INSTALL/lib/python3.5/site-packages:$OpenCV_INSTALL/lib/python3.7/site-packages  # just to add my python3 env.
#-------------------------------------------------------------
# Get directory name of this script.
cur_dir_name=${this_dir##*/}
#
if [ $this_dir = $(pwd) ]; then
  echo "[Error] You should not call this bash from where it is!"
  echo "[Usage] cd regQuatNet/reg_Sexp; bash ../trainval.sh  alexnet  0 "
  echo "Abort!"
  exit
fi
#---------------------------------------------------


net_arch=$1    #  'alexnet'  or 'vgg16'

# Extra args received from command line.
work_dir=./snapshots.ModelNetSO3_v2/$net_arch
device_ids=$2
remain_args="${@:3}"


CUDA_VISIBLE_DEVICES=$device_ids  python $this_dir/../trainval_workdir.py  $work_dir/conf.cache.yml  $work_dir  $device_ids  --net_arch=$net_arch  $remain_args

