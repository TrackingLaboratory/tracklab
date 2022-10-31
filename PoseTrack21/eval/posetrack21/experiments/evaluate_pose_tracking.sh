#!/bin/bash

GT_FOLDER="/globalscratch/ucl/elen/bstandae/Yolov5_StrongSORT_OSNet/PoseTrack21/data/posetrack_data/tiny_val"
TRACKERS_FOLDER="/globalscratch/ucl/elen/bstandae/reconn.ai.ssance/runs/val/exp0/pose_tracking"

python3 scripts/run_posetrack_challenge.py --GT_FOLDER $GT_FOLDER --TRACKERS_FOLDER $TRACKERS_FOLDER \
       --USE_PARALLEL False \
       --NUM_PARALLEL_CORES 8
