#!/bin/bash 

if test "$#" -lt 1; then
  echo "No GPU specified! Please pass gpu ids, i.e. 0,1,2"
  echo "I will grant the docker container access to all gpus now"
  gpus='all'
else
  gpus='"device='"$1"'"'
fi

echo $gpus

./build.sh
USERNAME=user

SRC_DIR="$PWD/.."
DATASET_DIR="/home/group-cvg/datasets"
WORK_DIR="/home/group-cvg"

#  --privileged\ gives access to all devices!
docker run\
    --gpus $gpus\
    --shm-size="42g"\
    -v "$SRC_DIR":/home/$USERNAME/baselines\
    -v "$DATASET_DIR":/media/datasets\
    -v "$WORK_DIR":/media/work2\
    -e PYTHONPATH=/home/$USERNAME/baselines\
    --rm -it\
    andoer/pt21-baselines \
    /bin/bash
