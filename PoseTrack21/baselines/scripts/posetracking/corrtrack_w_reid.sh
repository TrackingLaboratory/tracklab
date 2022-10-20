#!/bin/bash 

cd $PWD/docker

# build contaienr
./build.sh
cd $PWD/..

USERNAME=user 

# host paths
SRC_DIR="$PWD"
WORK_DIR="/home/group-cvg/"

# container paths
MOUNT_DESTINATION="/home/$USERNAME/baselines/"
MODEL_DIR="$MOUNT_DESTINATION/data/models/"
DATA_DIR="$MOUNT_DESTINATION/data/detections/"

POSE_ESTIMATION_MODEL_PATH="$MODEL_DIR/pose_estimation_model_3_stage_lr_1e-5_wo_vis.pth"
DATASET_PATH="/home/group-cvg/doering/2022/PoseTrackReIDDataset/release/PoseTrack21/"
CORR_MODEL_PATH="$MODEL_DIR/corrtrack_model.pth"
SEQNET_MODEL_PATH="$MODEL_DIR/seqnet.pth"
BBOX_ANNOTATION_FILE_PATH="$DATA_DIR/PoseTrack21_tracktor_bb_thres_0.5_val.json"

NUM_POSE_STAGES=3

# TODO: SET INFERENCE FOLDER TO CORRTRACK FOLDER, AS WE RE-USE ESTIMATED POSES!
INFERENCE_FOLDER_PATH="$MOUNT_DESTINATION/outputs/tracking_baselines/"
EXPERIMENT_FOLDER_NAME=corrtrack_baseline
REFINED_POSES_NMS_FOLDER=pose_3_stage_refined_nms
EXPERIMENT_SAVE_PATH="$MOUNT_DESTINATION/outputs/tracking_baselines/corrtrack_w_reid_baseline"

echo "RUN TRACKING"
docker run\
    --gpus $1 \
    --cpus 20 \
    --shm-size="20g"\
    -v "$SRC_DIR":$MOUNT_DESTINATION \
    -v "$WORK_DIR":$WORK_DIR\
    -e PYTHONPATH=$MOUNT_DESTINATION\
    --rm -it\
    andoer/pt21-baselines \
    python corrtrack/inference/run_corrtrack_w_reid.py \
    --save_path=${EXPERIMENT_SAVE_PATH}/${EXPERIMENT_FOLDER_NAME}/ \
    --sequences_path=${INFERENCE_FOLDER_PATH}/${EXPERIMENT_FOLDER_NAME}/${REFINED_POSES_NMS_FOLDER}/jt_0.05_oks_0.7_3_stage/sequences/ \
    --dataset_path=${DATASET_PATH} \
    --joint_threshold=0.1 \
    --oks_threshold=0.2 \
    --corr_threshold=0.3 \
    --min_keypoints=2 \
    --min_track_len=3 \
    --duplicate_ratio=0.6 \
    --post_process_joint_threshold=0.3 \
    --ckpt_path=${CORR_MODEL_PATH} \
    --seqnet_ckpt=${SEQNET_MODEL_PATH} \
    --inactive_patience=10 \
    --similarity_threshold=0.5 \
    --min_refinement_track_len=1 

