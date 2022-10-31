#!/bin/bash

GT_FOLDER="/globalscratch/ucl/elen/bstandae/Yolov5_StrongSORT_OSNet/PoseTrack21/data/posetrack_data/tiny_val"
TRACKERS_FOLDER="/globalscratch/ucl/elen/bstandae/reconn.ai.ssance/runs/val/exp0/pose_tracking"

python3 scripts/run_posetrack_reid_challenge.py --GT_FOLDER $GT_FOLDER --TRACKERS_FOLDER $TRACKERS_FOLDER \
       --PRINT_RESULTS True \
       --OUTPUT_PAPER_SUMMARY True \
       --PRINT_PAPER_SUMMARY True

