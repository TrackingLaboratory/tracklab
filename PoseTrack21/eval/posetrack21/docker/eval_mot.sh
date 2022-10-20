./build.sh
USERNAME=user
SRC_DIR="$PWD/.."
MOUNT_FOLDER=eval 

if test "$#" -lt 2; then
	echo "#############################################################"
	echo "# Expected min two keywords: \$gt_dir \$experiment_dir \$num_cores #"
	echo "#############################################################"
	echo ""
	exit 1
fi

GT_FOLDER="$1" 
EXP_FOLDER="$2"
NUM_CORES="${3:-8}"
echo $GT_FOLDER
echo $EXP_FOLDER
echo $NUM_CORES

docker run\
	--shm-size="2g"\
	-v "$SRC_DIR":/home/$USERNAME/$MOUNT_FOLDER\
	-v "$GT_FOLDER":/home/$USERNAME/gt_data\
	-v "$EXP_FOLDER":/home/$USERNAME/experiments\
	--rm -it\
	andoer/posetrack21_eval_kit\
	python3 /home/$USERNAME/eval/scripts/run_mot.py\
		--GT_FOLDER /home/$USERNAME/gt_data\
		--TRACKERS_FOLDER /home/$USERNAME/experiments\
		--USE_PARALLEL True\
		--NUM_PARALLEL_CORES $NUM_CORES


