![TrackLab](https://raw.githubusercontent.com/TrackingLaboratory/tracklab/refs/heads/main/docs/tracklab_banner.png)

TrackLab is an easy-to-use modular framework for multi-object pose/segmentation/bbox tracking that supports many tracking datasets and evaluation metrics.


<p align="center">
  <img src="https://raw.githubusercontent.com/TrackingLaboratory/tracklab/refs/heads/main/docs/assets/gifs/PoseTrack21_016236.gif" width="400" />
  <img src="https://raw.githubusercontent.com/TrackingLaboratory/tracklab/refs/heads/main/docs/assets/gifs/PoseTrack21_008827.gif" width="400" /> 
</p>

## News
- [2024.02.05] Public release

## Upcoming
- [x] Public release of the codebase
- [x] Add support for more datasets (DanceTrack, MOTChallenge, SportsMOT, ...)
- [ ] Add many more SOTA tracking methods and object detectors
- [ ] Improve documentation and add more tutorials

#### 🤝 How You Can Help
The TrackLab library is in its early stages, and we're eager to evolve it into a robust, mature tracking framework that can benefit the wider community.
If you're interested in contributing, feel free to open a pull-request or reach out to us!

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

## Documentation
You can find the documentation at [https://trackinglaboratory.github.io/tracklab/](https://trackinglaboratory.github.io/tracklab/) or in the docs/ folder. After installing, you can run `make html` inside this folder
to get an html version of the documentation.

## Installation guide[^1]

[^1]: Tested on `conda 22.11.1`, `Python 3.10.8`, `pip 22.3.1`, `g++ 11.3.0` and `gcc 11.3.0`

### Install using uv (recommended)

[Install uv](https://docs.astral.sh/uv/getting-started/installation/).

If you're creating a new project or using tracklab in an existing project, you can initialize a uv project with :

```bash
uv init
```

Then you can add tracklab : 

```bash
uv add tracklab
```

If you only want to use the virtual environment created by uv, you can use the following commands : 

```bash
uv venv --python 3.12
uv pip install tracklab
```

In both cases, you will then be able to run tracklab using the following command:
```bash
uv run tracklab
```

To update & run : 
```bash
uv run -U tracklab
```

### Install using conda

[Install conda](https://www.anaconda.com/docs/getting-started/miniconda/main).

Create a new virtual environment with conda, then install tracklab with pip : 
```bash
conda create -n tracklab pip python=3.10 pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda activate tracklab
pip install tracklab
```

You might need to change your torch installation depending on your hardware. Please check on 
[Pytorch website](https://pytorch.org/get-started/previous-versions/) to find the right version for you.

Update tracklab with : 

```bash
pip install -U tracklab
```

### Install additional dependencies
We support multiple extra packages: openmmlabs, transformers, yolox, that you can install with the following commands :

#### OpenMMLabs
```bash
(uv) pip install tracklab[openmmlab]
(uv run) mim install mmcv==2.0.1
```

#### Transformers & YOLOX
```bash
(uv) pip install tracklab[transformers,yolox]
```

### Manual installation

[Follow the above instructions to install uv](https://docs.astral.sh/uv/getting-started/installation/). Then clone the tracklab repository : 

```bash
git clone https://github.com/TrackingLaboratory/tracklab.git
cd tracklab
```

Run tracklab with : 

```bash
uv run tracklab
```

Since we're using uv under the hood, uv will automatically create a virtual environment for you, and
update the dependencies as you change them. You can also choose to install using conda, you'll then have
to run the following when inside a virtual environment:

```bash
pip install -e .
```

You might need to redo this if you update the repository, and some dependencies changed.

### External dependencies

- Get the **SoccerNet Tracking** dataset [here](https://github.com/SoccerNet/sn-tracking), rename the root folder as "SoccerNetMOT" and put it under the global dataset directory (specified under the `data_dir` config as explained below). Otherwise, you can modify the `dataset_path` config in [soccernet_mot.yaml](tracklab/configs/dataset/soccernet_mot.yaml) with your custom SoccerNet dataset directory.
- Download the pretrained model weights [here](https://drive.google.com/drive/folders/1MmDkSHWJ1S-V9YcLMkFOjm3zo65UELjJ?usp=drive_link) and put the "pretrained_models" directory under the main project directory (i.e. "/path/to/tracklab/pretrained_models").

### Setup

You will need to set up some variables before running the code : 

1. In configs/config.yaml :
   - `data_dir`: the directory where you will store the different datasets (must be an absolute path !)
   - All the parameters under the "Machine configuration" header
2. In the corresponding modules (tracklab/configs/modules/.../....yaml) :
   - The `batch_size`
   - You might want to change the model hyperparameters

To launch TrackLab with the default configuration defined in [configs/config.yaml](tracklab/configs/config.yaml), simply run: 
```bash
tracklab
```
This command will create a directory called `outputs` which will have a `${experiment_name}/yyyy-mm-dd/hh-mm-ss/` structure.
All the output files (logs, models, visualization, ...) from a run will be put inside this directory.

If you want to override some configuration parameters, e.g. to use another detection module or dataset, you can do so by modifying the corresponding parameters directly in the .yaml files under configs/.

All parameters are also configurable from the command-line, e.g. : (more info on Hydra's override grammar [here](https://hydra.cc/docs/advanced/override_grammar/basic/))
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


## Framework overview
### Hydra Configuration
TODO Describe TrackLab + Hydra configuration system 

### Architecture Overview
Here is an overview of the important TrackLab classes:
- **[TrackingDataset](tracklab/datastruct/tracking_dataset.py)**: Abstract class to be instantiated when adding a new dataset. The `TrackingDataset` contains one `TrackingSet` for each split of the dataset (train, val, test, etc).
  - Example: [SoccerNetMOT](tracklab/wrappers/dataset/soccernet/soccernet_mot.py). The [SoccerNet Tracking](https://github.com/SoccerNet/sn-tracking) dataset.
- **[TrackingSet](tracklab/datastruct/tracking_dataset.py)**: A tracking set contains three [Pandas](https://pandas.pydata.org/) dataframes:
  1. `video_metadatas`: contains one row of information per video (e.g. fps, width, height, etc).
  2. `image_metadatas`: contains one row of information per image (e.g. frame_id, video_id, etc).
  3. `detections_gt`: contains one row of information per ground truth detection (e.g. frame_id, video_id, bbox_ltwh, track_id, etc).
- **[TrackerState](tracklab/datastruct/tracker_state.py)**: Core class that contains all the information about the current state of the tracker. All modules in the tracking pipeline update the tracker_state sequentially. The tracker_state contains one key dataframe:
  1. `detections_pred`: contains one row of information per predicted detection (e.g. frame_id, video_id, bbox_ltwh, track_id, reid embedding, etc).
- **[TrackingEngine](tracklab/engine/engine.py)**: This class is responsible for executing the entire tracking pipeline on the dataset. It loops over all videos of the dataset and calls all modules defined in the pipeline sequentially. The exact execution order (e.g. online/offline/...) is defined by the TrackingEngine subclass.
  - Example: **[OfflineTrackingEngine](tracklab/engine/offline.py)**. The offline tracking engine performs tracking one module after another to speed up inference by leveraging large batch sizes and maximum GPU utilization. For instance, YoloV8 is first applied on an entire video by batching multiple images, then the re-identification model is applied on all detections in the video, etc. 
- **[Pipeline](tracklab/pipeline/module.py)**: Define the order in which modules are executed by the TrackingEngine. If a tracker_state is loaded from disk, modules that should not be executed again must be removed.
  - Example: [bbox_detector, reid, track]
- **[VideoLevelModule](tracklab/pipeline/videolevel_module.py)**: Abstract class to be instantiated when adding a new tracking module that operates on all frames simultaneously. Can be used to implement offline tracking strategies, tracklet level voting mechanisms, etc. 
  - Example: [VotingTrackletJerseyNumber](tracklab/wrappers/jn_detector/voting_tracklet_jn_api.py). To perform majority voting within each tracklet and compute a consistent tracklet level attribute (an attribute can be, for instance, the result of a detection level classification task).
- **[ImageLevelModule](tracklab/pipeline/imagelevel_module.py)**: Abstract class to be instantiated when adding a new tracking module that operates on a single frame. Can be used to implement online tracking strategies, pose/segmentation/bbox detectors, etc.
  - Example 1: [YOLOv8](tracklab/wrappers/bbox_detector/yolov8_api.py). To perform object detection on each image with [YOLOv8](https://github.com/ultralytics/ultralytics). Creates a new row (i.e. detection) within `detections_pred`.
  - Example 2: [StrongSORT](tracklab/wrappers/track/strong_sort_api.py). To perform online tracking with [StrongSORT](https://github.com/dyhBUPT/StrongSORT). Creates a new "track_id" column for each detection within `detections_pred`. 
- **[DetectionLevelModule](tracklab/pipeline/detectionlevel_module.py)**: Abstract class to be instantiated when adding a new tracking module that operates on a single detection. Can be used to implement pose estimation for top-down strategies, re-identification, attributes recognition, etc. 
  - Example 1: [EasyOCR](tracklab/wrappers/jn_detector/easyocr_api.py). To perform jersey number recognition on each detection with [EasyOCR](https://github.com/JaidedAI/EasyOCR). Creates a new "jersey_number" column within `detections_pred`.
  - Example 2: [BPBReId](tracklab/wrappers/reid/bpbreid_api.py). To perform person re-identification on each detection with [BPBReID](https://github.com/VlSomers/bpbreid). Creates a new "embedding" column within `detections_pred`.
- **[Callback](tracklab/callbacks/callback.py)**: Implement this class to add a callback that is triggered at a specific point during the tracking process, e.g. when dataset/video/module processing starts/ends.
  - Example: [VisualizationEngine](tracklab/core/visualization_engine.py). Implements "on_video_loop_end" to save each video tracking results as a .mp4 or a list of .jpg. 
- **[Evaluator](tracklab/core/evaluator.py)**: Implement this class to add a new evaluation metric, such as MOTA, HOTA, or any other (non-tracking related) metrics. 
  - Example: [TrackEvalEvaluator](tracklab/wrappers/eval/trackeval_evaluator.py). Evaluate the performance of a tracker using the official [TrackEval library](https://github.com/JonathonLuiten/TrackEval).

### Execution Flow Overview
Here is an overview of what happen when you run TrackLab:
[tracklab/main.py](tracklab/main.py) is the main entry point and receives the complete Hydra's configuration as input. 
[tracklab/main.py](tracklab/main.py) is usually called via the following command through the root [main.py](main.py) file: `python main.py`.
Within [tracklab/main.py](tracklab/main.py), all modules are first instantiated.
Then training any tracking module (e.g. the re-identification model) on the tracking training set is supported by calling the "train" method of the corresponding module.
Tracking is then performed on the validation or test set (depending on the configuration) via the TrackingEngine.run() function.
For each video in the evaluated set, the TrackingEngine calls the "run" method of each module (e.g. detector, re-identifier, tracker, ...) sequentially.
The TrackingEngine is responsible for batching the input data (e.g. images, detections, ...) before calling the "run" method of each module with the correct input data.
After a module has been called with a batch of input data, the TrackingEngine then updates the TrackerState object with the module outputs.
At the end of the tracking process, the TrackerState object contains the tracking results of each video.
Visualizations (e.g. .mp4 results videos) are generated during the TrackingEngine.run() call, after a video has been tracked and before the next video is processed.
Finally, evaluation is performed via the evaluator.run() function once the TrackingEngine.run() call is completed, i.e. after all videos have been processed.

## Tutorials
### Dump and load the tracker state to save computation time
When developing a new module, it is often useful to dump the tracker state to disk to save computation time and avoid running the other modules several times.
Here is how to do it:
1. First, save the tracker state by using the corresponding configuration in the config.yaml file:
```yaml
defaults:
    - state: save
# ...
state:
  save_file: "states/${experiment_name}.pklz"  # 'null' to disable saving. This is the save path for the tracker_state object that contains all modules outputs (bboxes, reid embeddings, jersey numbers, roles, teams, etc)
  load_file: null
```
2. Run Tracklab. The tracker state will be saved in the experiment folder as a .pklz file.
3. Then modify the load_file key in "config.yaml" to specify the path to the tracker state file that has just been created  (`load_file: "..."` config).
5. In config.yaml, remove from the pipeline all modules that should not be executed again. For instance, if you want to use the detections and reid embeddings from the saved tracker state, remove the "bbox_detector" and "reid" modules from the pipeline. Use `pipeline: []` if no module should be run again.
```yaml
defaults:
    - state: save
# ...
pipeline:
  - track
# ...
state:
  save_file: null  # 'null' to disable saving. This is the save path for the tracker_state object that contains all modules outputs (bboxes, reid embeddings, jersey numbers, roles, teams, etc)
  load_file: "path/to/tracker_state.pklz"
```
8. Run Tracklab again.


## Citation
If you use this repository for your research or wish to refer to our contributions, please use the following BibTeX entries:

[TrackLab](https://github.com/TrackingLaboratory/tracklab):
```
@misc{Joos2024Tracklab,
	title = {{TrackLab}},
	author = {Joos, Victor and Somers, Vladimir and Standaert, Baptiste},
	journal = {GitHub repository},
	year = {2024},
	howpublished = {\url{https://github.com/TrackingLaboratory/tracklab}}
}
```

[SoccerNet Game State Reconstruction](https://arxiv.org/abs/2404.11335):
```
@inproceedings{Somers2024SoccerNetGameState,
        title = {{SoccerNet} Game State Reconstruction: End-to-End Athlete Tracking and Identification on a Minimap},
        author = {Somers, Vladimir and Joos, Victor and Giancola, Silvio and Cioppa, Anthony and Ghasemzadeh, Seyed Abolfazl and Magera, Floriane and Standaert, Baptiste and Mansourian, Amir Mohammad and Zhou, Xin and Kasaei, Shohreh and Ghanem, Bernard and Alahi, Alexandre and Van Droogenbroeck, Marc and De Vleeschouwer, Christophe},
        booktitle = cvsports,
        month = Jun,
        year = {2024},
        address = city-seattle,
}
```

[BPBreID](https://arxiv.org/abs/2211.03679):
```
@article{bpbreid,
    archivePrefix = {arXiv},
    arxivId = {2211.03679},
    author = {Somers, Vladimir and {De Vleeschouwer}, Christophe and Alahi, Alexandre},
    doi = {10.48550/arxiv.2211.03679},
    eprint = {2211.03679},
    file = {:Users/vladimirsomers/Library/CloudStorage/OneDrive-SportradarAG/PhD/Scientific Literature/Mendeley/Somers, De Vleeschouwer, Alahi_2023_Body Part-Based Representation Learning for Occluded Person Re-Identification.pdf:pdf},
    isbn = {2211.03679v1},
    journal = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV23)},
    month = {nov},
    title = {{Body Part-Based Representation Learning for Occluded Person Re-Identification}},
    url = {https://arxiv.org/abs/2211.03679v1 http://arxiv.org/abs/2211.03679},
    year = {2023}
}
```
