#!/bin/bash

GT_FOLDER="${1:-/home/group-cvg/doering/2022/PoseTrackReIDEvaluationData/dummy_gt/}"
TRACKERS_FOLDER="${2:-/home/group-cvg/doering/2022/PoseTrackReIDEvaluationData/dummy_pr/}"

python3 scripts/run_posetrack_reid_challenge.py --GT_FOLDER $GT_FOLDER --TRACKERS_FOLDER $TRACKERS_FOLDER \
       --PRINT_RESULTS False \
       --OUTPUT_PAPER_SUMMARY True \
       --PRINT_PAPER_SUMMARY True

