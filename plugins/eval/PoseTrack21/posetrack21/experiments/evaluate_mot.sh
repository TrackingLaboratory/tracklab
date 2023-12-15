#!/bin/bash

GT_FOLDER="${1:-/home/group-cvg/doering/2022/PoseTrackReIDDataset/release/PoseTrack21/posetrack_mot/mot/val/}"
TRACKERS_FOLDER="${2:-/home/group-cvg/doering/2022/PoseTrackReIDEvaluationData/test_mot_results/trackers/}"

python3 scripts/run_mot.py --GT_FOLDER $GT_FOLDER --TRACKERS_FOLDER $TRACKERS_FOLDER \
       --USE_PARALLEL True \
       --NUM_PARALLEL_CORES 8 \
