#!/bin/bash
USERNAME=user

./build.sh

SRC_DIR="$PWD/.."
WORK_DIR="/home/group-cvg/"
DATA_DIR="/media/data/"

docker run\
    --shm-size="68g"\
    -v "$SRC_DIR":/home/$USERNAME/MOT_Evaluation\
    -v "$WORK_DIR":/home/group-cvg\
    -e PYTHONPATH=/home/$USERNAME/MOT_Evaluation\
    --rm -it\
    mot_evaluation \
    /bin/bash

