# TrackLab
<p align="center"></p>
TrackLab is an easy-to-use modular framework for multi-object pose/segmentation/bbox tracking that supports many tracking datasets and evaluation metrics.  

<p align="center">
  <img src="docs/assets/gifs/PoseTrack21_008827.gif" width="400" />
  <img src="docs/assets/gifs/PoseTrack21_016236.gif" width="400" /> 
</p>

## News
- [2024.02.01] Planned public release 

## Introduction
Welcome to this official repository of TrackLab, a modular framework for multi-object tracking.
TrackLab is designed for research purposes and supports many types of detectors (bounding boxes, pose, segmentation), datasets and evaluation metrics.
Every component of TrackLab, such as detector, tracker, re-identifier, etc, is configurable via standard yaml files ([Hydra config framework](https://github.com/facebookresearch/hydra))
TrackLab is designed to be easily extended to support new methods.

TrackLab is composed of multiple modules:
1. A detector (YOLOv8, ...)
2. A re-identification model (BPBReID, ...)
3. A tracker (DeepSORT, StrongSORT, OC-SORT, ...)

Here's what makes TrackLab different from other existing tracking frameworks:
- Fully modular framework to quickly integrate any detection/reid/tracking method or develop your own
- It allows supervised training of the ReID model on the tracking training set
- It provides a fully configurable visualization tool with the possibility to display any dev/debug information
- It supports online and offline tracking methods (compared to MMTracking, AlphaPose, LightTrack and other libs who only support online tracking)
- It supports many tracking-related tasks:
  - multi-object (bbox) tracking
  - multi-person pose tracking
  - multi-person pose estimation
  - person re-identification


## Framework overview
### Hydra Configuration
TODO Describe TrackLab + Hydra configuration system 

### Architecture Overview
Important TrackLab classes include:
- **[TrackerState](pbtrack/datastruct/tracker_state.py)**: Core class that contains all the information about the current state of the tracker. All modules update the tracker_state sequentially. The TrackState contains one TrackingSet for each split of the dataset (train, val, test, etc).
- **[TrackingSet]()**: A tracking set contains four [Panda](https://pandas.pydata.org/) dataframes:
  - video_metadatas: contains one row of information per video (e.g. fps, width, height, etc)
  - image_metadatas: contains one row of information per image (e.g. frame_id, video_id, etc)
  - detections_gt: contains one row of information per ground truth detection (e.g. frame_id, video_id, bbox_ltwh, track_id, etc)
  - detections_pred: contains one row of information per predicted detection (e.g. frame_id, video_id, bbox_ltwh, track_id, reid embedding, etc)
- **[TrackingDataset](pbtrack/datastruct/tracking_dataset.py)**: 
  - Example: [SoccerNetMOT](pbtrack/wrappers/datasets/soccernet/soccernet_mot.py): The [SoccerNet Tracking](https://github.com/SoccerNet/sn-tracking) dataset
- **[TrackingEngine](pbtrack/engine/engine.py)**: This class is responsible for executing the entire tracking pipeline on the dataset. It loops over all videos of the dataset and calls all modules defined in the pipeline sequentially. The exact execution order (e.g. online/offline/...) is defined by the TrackingEngine subclass.
  - Example: **[OfflineTrackingEngine](pbtrack/engine/offline.py)**: The offline tracking engine performs tracking one module after another to speed up inference by leveraging large batch sizes and maximum GPU utilization. For instance, YoloV8 is first applied on an entire video by batching multiple images, then the re-identification model is applied on all detections of the video with large detections batches, etc. 
- **[Pipeline](pbtrack/pipeline/module.py)**: Define the order in which modules are executed by the TrackingEngine. If a tracker_state is loaded from disk, modules that should not be executed again should be removed.
  - Example: [bbox_detector, reid, track, jn_detect, jn_tracklet]
- **[VideoLevelModule](pbtrack/pipeline/videolevel_module.py)**: 
  - Example: [VotingTrackletJerseyNumber](pbtrack/wrappers/jn_detector/voting_tracklet_jn_api.py): To perform majority voting within each tracklet and compute a consistent tracklet level jersey number. Update the "jersey_number" column within detections_pred.
- **[ImageLevelModule](pbtrack/pipeline/imagelevel_module.py)**: 
  - Example 1: [YOLOv8](pbtrack/wrappers/detect_multiple/yolov8_api.py): To perform object detection on each image with [YOLOv8](https://github.com/ultralytics/ultralytics). Create a new row (i.e. detection) within detections_pred.
  - Example 2: [StrongSORT](pbtrack/wrappers/track/strong_sort_api.py): To perform online tracking with [StrongSORT](https://github.com/dyhBUPT/StrongSORT). Creates a new "track_id" column for each detection within detections_pred. 
- **[DetectionLevelModule](pbtrack/pipeline/detectionlevel_module.py)**: 
  - Example 1: [EasyOCR](pbtrack/wrappers/jn_detector/easyocr_api.py): To perform jersey number recognition on each detection with [EasyOCR](https://github.com/JaidedAI/EasyOCR). Create a new "jersey_number" column within detections_pred.
  - Example 2: [BPBReId](pbtrack/wrappers/reid/bpbreid_api.py): To perform person re-identification on each detection with [BPBReID](https://github.com/VlSomers/bpbreid). Create a new "embedding" column within detections_pred.
- **[Callback](pbtrack/callbacks/callback.py)**: Implement this class to add a callback to be called at a specific point during the tracking process, e.g. when task/dataset/video processing starts/ends.
  - Example: [VisualizationEngine](pbtrack/core/visualization_engine.py): Implements "on_video_loop_end" to save each video tracking results as a .mp4 or a list of .jpg. 
- **[Evaluator](pbtrack/core/evaluator.py)**: Implement this class to add a new evaluation metric, such as MOTA, HOTA, or any other (non-tracking related) metrics. 
  - Example: [SoccerNetMOTEvaluator](pbtrack/wrappers/eval/soccernet/soccernet_mot_evaluator.py): Evaluate performance of a tracker on the SoccerNet Tracking dataset using the official [evaluation library](https://github.com/SoccerNet/sn-tracking).
- **[]()**:

### Execution Flow Overview
Here is an overview of what happen when you run TrackLab:
[tracklab/main.py](tracklab/main.py) is the main entry point and receives the complete Hydra's configuration as input. 
[tracklab/main.py](tracklab/main.py) is usually called via the following command through the root [main.py](main.py) file: python main.py.
Within [tracklab/main.py](tracklab/main.py), all modules are first instantiated.
Training any tracking module (e.g. re-identification model) on the tracking training set is supported by calling the "train" method of the corresponding module.
Tracking is then performed on the validation or test set (depending on the configuration) via the TrackingEngine run() function.
For each video in the evaluated set, the TrackingEngine calls the "run" method of each module (e.g. detector, re-identifier, tracker, ...) sequentially.
The TrackingEngine is respnsible for batching the input data (e.g. images, detections, ...) before calling the "run" method of each module with the correct input data.
After a module has been called with a batch of input data, the TrackingEngine then updates the TrackerState object with the module outputs.
At the end of the tracking process, the TrackerState object contains the tracking results of each video.
Visualization (e.g. .mp4 results videos) are generated during the TrackingEngine.run() call, after a video has been tracked and before the next video is processed.
Evaluation is performed via the evaluator.run() function once the TrackingEngine.run() call is completed, i.e. after all videos have been processed.


## Documentation
You can find the documentation in the docs/ folder. After installing, you can run `make html` inside this folder
to get an html version of the documentation.

## Installation guide[^1]

[^1]: Tested on `conda 22.11.1`, `Python 3.10.8`, `pip 22.3.1`, `g++ 11.3.0` and `gcc 11.3.0`

### Clone the repository

```bash
git clone https://github.com/TrackingLaboratory/tracklab.git
cd tracklab
```

### Manage the environment

#### Create and activate a new environment

```bash
conda create -n tracklab pip python=3.10 pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda activate tracklab
```

You might need to change your torch installation depending on your hardware. Please check on 
[Pytorch website](https://pytorch.org/get-started/previous-versions/) to find the right version for you.

#### Install the dependencies
Get into your repo and install the requirements with :

```bash
pip install -e .
mim install mmcv-full
```

You might need to redo this if you update the repository, and some dependencies changed.

### External dependencies

- Get the **SoccerNet Tracking** dataset [here](https://github.com/SoccerNet/sn-tracking).
- Download the pretrained model weights [here](https://drive.google.com/drive/folders/1MmDkSHWJ1S-V9YcLMkFOjm3zo65UELjJ?usp=drive_link) and put the "pretrained_models" directory under the main project directory (i.e. "/path/to/tracklab/pretrained_models").

### Setup

You will need to set up some variables before running the code : 

1. In configs/config.yaml :
   - `data_dir`: the directory where you will store the different datasets (must be an absolute path !)
   - All the parameters under the "Machine configuration" header
2. In the corresponding modules (configs/modules/.../....yaml) :
   - The `batch_size`
   - You might want to change the model hyperparameters

All these variables are also configurable from the command-line, e.g. : (more info on Hydra's override grammar [here](https://hydra.cc/docs/advanced/override_grammar/basic/))
```bash
tracklab 'data_dir=${project_dir}/data' 'model_dir=${project_dir}/models' modules/reid=bpbreid pipeline=[bbox_detector,reid,track]
```
`${project_dir}` is a variable that is configured to be the root of the project you're running the code in. When using
it in a command, make sure to use single quotes (') as they would otherwise be seen as 
environment variables.

To find all the (many) configuration options you have, use :
```bash
tracklab --help
```

The first section contains the configuration groups, while the second section
shows all the possible options you can modify.
