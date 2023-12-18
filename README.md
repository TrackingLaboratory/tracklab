# PbTrack

PbTrack is an easy-to-use modular framework for multi-object pose/segmentation/bbox tracking that supports many tracking datasets and evaluation metrics.  

<p align="center">
  <img src="docs/assets/gifs/PoseTrack21_008827.gif" width="400" />
  <img src="docs/assets/gifs/PoseTrack21_016236.gif" width="400" /> 
</p>

## News
- [2023.xx.yy] Planned public release 

## Introduction
Welcome to this official repository of PbTrack, a modular framework for multi-object tracking.
PbTrack is designed for research purposes and supports many types of detectors (bounding boxes, pose, segmentation), datasets, evaluation metrics.
Every component of PbTrack, such as detector, tracker, re-identifier, etc, is configurable via standard yaml files
PbTrack is designed to be easily extended to support new methods.

PbTrack is composed of multiple modules:
1. A detector (OpenPifPaf, YOLO, ...)
2. A re-identification model (BPBReID, ...)
3. A camera motion compensation algorithm (findTransformECC, ...)
4. A detection forecaster (Kalman filter, ...)
5. An appearance similarity metric (cosine similarity, ...)
6. An spatio-temporal similarity metric (IOU, OKS, ...)
7. An association algorithm (Hungarian algorithm, ...)


Here's what makes PbTrack different from other existing tracking frameworks:
- Fully modular framework to quickly integrate any detection/reid/tracking method or develop your own
- It allows supervised training of the ReID model on the tracking training set
- It provides a fully configurable visualization tool with the possibility to display any dev/debug information
- It supports online and offline tracking methods (compared to MMTracking, AlphaPose, LightTrack and other libs who only support online tracking)
- It supports many tracking-related tasks:
  - multi-object (bbox) tracking
  - multi-person pose tracking
  - multi-person pose estimation
  - person search
  - multi-person cross-video tracking
  - person re-identification


## Installation guide[^1]

[^1]: Tested on `conda 22.11.1`, `Python 3.10.8`, `pip 22.3.1`, `g++ 11.3.0` and `gcc 11.3.0`

### Clone the repository

```bash
git clone -b soccernet https://github.com/PbTrack/pb-track.git pbtrack-soccernet --recurse-submodules
cd pb-track
```

If you cloned the repo without using the `--recurse-submodules` option, you can still download the submodules with :

```bash
git submodule update --init --recursive
```

### Manage the environment

#### Create and activate a new environment

```bash
conda create -n pbtrack pip python=3.10 pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda activate pbtrack
```

You might need to change your torch installation depending on your hardware. Please check on 
[Pytorch website](https://pytorch.org/get-started/previous-versions/) to find the right version for you.

#### Install the dependencies
Get into your repo and install the requirements with :

```bash
pip install -e .
mim install mmcv-full
```

Note: if you re-install dependencies after pulling the last changes, and a new git submodule has been added, do not forget to recursively update all the submodule before running above commands:

```bash
git submodule update --init --recursive
```

#### Setup reid

```bash
cd plugins/reid/bpbreid/
python setup.py develop
```

### External dependencies

- Get the **PoseTrack21** dataset [here](https://github.com/anDoer/PoseTrack21/tree/35bd7033ec4e1a352ae39b9522df5a683f83781b#how-to-get-the-dataset).
- Or get out custom **TinyPoseTrack21** dataset with only two videos [here](https://drive.google.com/file/d/15aX67GAKpf8faaBE4SOJAs_KGghzfWl4/view?usp=sharing).
- Get the pretrained weights of **BPBReID** [here](https://github.com/VlSomers/bpbreid#download-the-pre-trained-models).


### Quick (dirty?) install guide for inference
1. Follow above instruction for setting up the environment
2. Download the pretrained weights of OpenPifPaf and BPBreID on [Google Drive](https://drive.google.com/drive/folders/1ZLKYpWIFPOw0-op0dNVP1Csw3CjKr-1B?usp=share_link)
3. Update configs to point to the downloaded weights:
4. 'configs/reid/torchreid.yaml' -> load_weights: "/path/to/job-35493841_85mAP_95r1_ta_model.pth.tar"
4. 'configs/reid/torchreid.yaml' -> hrnet_pretrained_path: "/path/to/weights/folder" # /!\ just put the folder name in which the weights 'hrnetv2_w32_imagenet_pretrained.pth' are stored, not the filename
5. 'configs/detect/openpifpaf.yaml' -> checkpoint: "/path/to/shufflenetv2k30_dense_default_wo_augm.f07de325"
6. Update config to point to your inference video: in 'configs/config.yaml', remplace '  - dataset: posetrack21'  (line 8) with '  - dataset: external_video'
7. In 'configs/dataset/external_video.yaml', update the path to your video file (under 'video_path')
8. Finally, execute 'python main.py' to run the inference on the video file 'test.mp4'
