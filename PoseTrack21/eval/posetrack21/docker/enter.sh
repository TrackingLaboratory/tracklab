./build.sh

USERNAME=user
SRC_DIR="$PWD/../"
WORK="/home/group-cvg/"
DATASET="/home/group-cvg/datasets/"

MOUNT_FOLDER=eval 

docker run\
	--shm-size="2g"\
	-v "$SRC_DIR":/home/$USERNAME/$MOUNT_FOLDER\
	-v "$WORK":/home/group-cvg/\
	-v "$DATASET":/home/group-cvg/datasets\
	--rm -it\
    andoer/posetrack21_eval_kit \
	/bin/bash
