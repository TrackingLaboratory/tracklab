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
BBOX_ANNOTATION_FILE_PATH="$DATA_DIR/PoseTrack21_tracktor_bb_thres_0.5_val.json"

NUM_POSE_STAGES=3

INFERENCE_FOLDER_PATH="$MOUNT_DESTINATION/outputs/tracking_baselines/"
EXPERIMENT_FOLDER_NAME=corrtrack_baseline

POSE_FOLDER=pose_3_stage
POSE_NMS_FOLDER=pose_3_stage_nms  # NMS AFTER POSE ESTIMATION
WARPED_POSE_FOLDER=pose_3_stage_warped
REFINED_POSES_FOLDER=pose_3_stage_warped_and_refined
REFINED_POSE_NMS_FOLDER=pose_3_stage_refined_nms
CORR_TRACKING_FOLDER=pose_3_stage_corr_tracking

# 1) pose estimation
echo "RUN POSE ESTIMATION"
docker run\
    --gpus $1\
    --cpus 20 \
    --shm-size="20g"\
    -v "$SRC_DIR":$MOUNT_DESTINATION \
    -v "$WORK_DIR":$WORK_DIR\
    -e PYTHONPATH=$MOUNT_DESTINATION\
    --rm -it\
    andoer/pt21-baselines \
    python corrtrack/inference/pose_estimation.py \
    --save_path=${INFERENCE_FOLDER_PATH}/${EXPERIMENT_FOLDER_NAME}/${POSE_FOLDER}/sequences/ \
    --dataset_path=${DATASET_PATH} \
    --prefix=val \
    --annotation_file_path=${BBOX_ANNOTATION_FILE_PATH} \
    --checkpoint_path=${POSE_ESTIMATION_MODEL_PATH} \
    --joint_threshold=0.0 \
    --output_size_x=288 \
    --output_size_y=384 \
    --num_stages=${NUM_POSE_STAGES} \
    --batch_size=128\
    --num_workers=20

#2) nms
echo "RUN NMS"
docker run\
    --gpus $1 \
    --cpus 20 \
    --shm-size="20g"\
    -v "$SRC_DIR":$MOUNT_DESTINATION \
    -v "$WORK_DIR":$WORK_DIR\
    -e PYTHONPATH=$MOUNT_DESTINATION\
    --rm -it\
    andoer/pt21-baselines \
    python corrtrack/tools/pose_nms.py \
    --result_path=${INFERENCE_FOLDER_PATH}/${EXPERIMENT_FOLDER_NAME}/${POSE_FOLDER}/sequences/ \
    --save_path=${INFERENCE_FOLDER_PATH}/${EXPERIMENT_FOLDER_NAME}/${POSE_NMS_FOLDER}/ \
    --joint_threshold=0.05 \
    --oks_threshold=0.7

# 3) pose warping
echo "RUN WARPING"
docker run\
    --gpus $1\
    --cpus 20 \
    --shm-size="20g"\
    -v "$SRC_DIR":$MOUNT_DESTINATION \
    -v "$WORK_DIR":$WORK_DIR\
    -e PYTHONPATH=$MOUNT_DESTINATION\
    --rm -it\
    andoer/pt21-baselines \
    python corrtrack/inference/warp_poses.py \
    --corr_ckpt_path=${CORR_MODEL_PATH} \
    --pose_ckpt_path=${POSE_ESTIMATION_MODEL_PATH} \
    --num_stages=${NUM_POSE_STAGES} \
    --sequences_path=${INFERENCE_FOLDER_PATH}/${EXPERIMENT_FOLDER_NAME}/${POSE_NMS_FOLDER}/jt_0.05_oks_0.7_3_stage/sequences/ \
    --save_path=${INFERENCE_FOLDER_PATH}/${EXPERIMENT_FOLDER_NAME}/${WARPED_POSE_FOLDER} \
    --dataset_path=${DATASET_PATH} \
    --oks_threshold=0.8 \
    --corr_threshold=0.1 \
    --joint_threshold=0.1 \
    --bb_thres=0.5

# 4) recovering missed detections
echo "Run pose recovery"
docker run\
    --gpus $1\
    --cpus 20 \
    --shm-size="20g"\
    -v "$SRC_DIR":$MOUNT_DESTINATION \
    -v "$WORK_DIR":$WORK_DIR\
    -e PYTHONPATH=$MOUNT_DESTINATION\
    --rm -it\
    andoer/pt21-baselines \
    python corrtrack/inference/recover_missed_detections.py \
    --sequences_path=${INFERENCE_FOLDER_PATH}/${EXPERIMENT_FOLDER_NAME}/${WARPED_POSE_FOLDER}/val_set_bb_0.5_jt_0.1_with_corr_0.1_at_oks_0.8/sequences/ \
    --dataset_path=${DATASET_PATH} \
    --save_path=${INFERENCE_FOLDER_PATH}/${EXPERIMENT_FOLDER_NAME}/${REFINED_POSES_FOLDER} \
    --oks=0.6 \
    --warp_oks=0.8 \
    --joint_th=0.1

# 5) NMS again
echo "RUN NMS AGAIN"
docker run\
    --gpus $1\
    --cpus 20 \
    --shm-size="20g"\
    -v "$SRC_DIR":$MOUNT_DESTINATION \
    -v "$WORK_DIR":$WORK_DIR\
    -e PYTHONPATH=$MOUNT_DESTINATION\
    --rm -it\
    andoer/pt21-baselines \
    python corrtrack/tools/pose_nms.py \
    --result_path=${INFERENCE_FOLDER_PATH}/${EXPERIMENT_FOLDER_NAME}/${REFINED_POSES_FOLDER}/recover_missed_detections_jt_th_0.1_oks_0.6/sequences/ \
    --save_path=${INFERENCE_FOLDER_PATH}/${EXPERIMENT_FOLDER_NAME}/${REFINED_POSE_NMS_FOLDER}/ \
    --joint_threshold=0.05 \
    --oks_threshold=0.7

# 6) Corr Tracking
echo "RUN TRACKING"
docker run\
    --gpus $1\
    --cpus 20 \
    --shm-size="20g"\
    -v "$SRC_DIR":$MOUNT_DESTINATION \
    -v "$WORK_DIR":$WORK_DIR\
    -e PYTHONPATH=$MOUNT_DESTINATION\
    --rm -it\
    andoer/pt21-baselines \
    python corrtrack/inference/run_corrtrack.py \
    --save_path=${INFERENCE_FOLDER_PATH}/${EXPERIMENT_FOLDER_NAME}/${CORR_TRACKING_FOLDER}/ \
    --sequences_path=${INFERENCE_FOLDER_PATH}/${EXPERIMENT_FOLDER_NAME}/${REFINED_POSE_NMS_FOLDER}/jt_0.05_oks_0.7_3_stage/sequences/ \
    --dataset_path=${DATASET_PATH} \
    --joint_threshold=0.1 \
    --oks_threshold=0.2 \
    --corr_threshold=0.3 \
    --min_keypoints=2 \
    --min_track_len=3 \
    --duplicate_ratio=0.6 \
    --post_process_joint_threshold=0.3 \
    --break_tracks \
    --ckpt_path=${CORR_MODEL_PATH}
