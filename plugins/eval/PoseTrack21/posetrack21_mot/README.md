# MOT Evaluation Kit 

We adopted [motmetrics](https://github.com/cheind/py-motmetrics) to consider ignore regions for PoseTrack21MOT and used this code for the evaluation of MOTA and IDF related metrics.

## Installation 
You can run `mot/docker/build.sh` to build the docker container. 
Alternatively, you can use `mot/docker/environment.yml` to install the required conda environment.

## Usage 
```
python evaluate_mot --dataset_path $PATH_TO_DATASET_ROOT \
                    --mot_path $PATH_TO_RESPECTIVE_MOT_FOLDER \
                    --result_PATH $FOLDER_WITH_YOUR_RESULTS \
                    --use_ignore_regions # whether to use ignore regions or not 
```
