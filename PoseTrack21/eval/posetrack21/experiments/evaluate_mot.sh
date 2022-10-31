#!/bin/bash

GT_FOLDER="/globalscratch/ucl/elen/bstandae/Yolov5_StrongSORT_OSNet/PoseTrack21/data/posetrack_mot/mot/tiny_val"
TRACKERS_FOLDER="/globalscratch/ucl/elen/bstandae/reconn.ai.ssance/runs/val/exp0/mot/"

python3 scripts/run_mot.py --GT_FOLDER $GT_FOLDER --TRACKERS_FOLDER $TRACKERS_FOLDER \
       --USE_PARALLEL False \
       --NUM_PARALLEL_CORES 8 \
