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
Every component of TrackLab, such as detector, tracker, re-identifier, etc, is configurable via standard yaml files ([Hydra cofnig framework](https://github.com/facebookresearch/hydra))
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
