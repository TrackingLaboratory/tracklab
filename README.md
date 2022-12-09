# PbTrack

Work in progress

## Installation guide

(tested on python 3.8.6)

### Clone the repo

```bash
git clone https://github.com/PbTrack/pb-track.git --recurse-submodules
```

or

```bash
git clone git@github.com:PbTrack/pb-track.git
cd pb-track
git submodule update --init --recursive
```

### Manage environment

#### Create a new environment

```bash
conda create --name pb-track
conda activate pb-track
```

#### Install the dependencies

**TODO**

Vérifier si un seul call à requirements.txt dans notre folder de base suffit.

**END TODO**

##### modules/detect/DEKR

**TODO**

(Sera viré si on arrive à utiliser MMPose)

**END TODO**

```bash
pip install -r modules/detect/DEKR/requirements.txt
```

[COCOAPI](https://github.com/cocodataset/cocoapi) installation

```bash
git clone https://github.com/cocodataset/cocoapi.git ../cocoapi
cd ../cocoapi/PythonAPI
pip install Cython
make install
cd ../../pb-track
```

[CrowdPose](https://github.com/Jeff-sjtu/CrowdPose) installation

```bash
git clone https://github.com/Jeff-sjtu/CrowdPose.git ../CrowdPose
cd ../CrowdPose/crowdpose-api/PythonAPI/
sh install.sh
cd ../../../pb-track
```

[download the weights](https://mailustceducn-my.sharepoint.com/:f:/g/personal/aa397601_mail_ustc_edu_cn/EmoNwNpq4L1FgUsC9KbWezABSotd3BGOlcWCdkBi91l50g?e=HWuluh) and make the modules/detect/DEKR/ directory look like

```text
${modules/detect/DEKR}
|-- model
`-- |-- imagenet
    |   |-- hrnet_w32-36af842e.pth
    |   `-- hrnetv2_w48_imagenet_pretrained.pth
    |-- pose_coco
    |   |-- pose_dekr_hrnetw32_coco.pth
    |   `-- pose_dekr_hrnetw48_coco.pth
    |-- pose_crowdpose
    |   |-- pose_dekr_hrnetw32_crowdpose.pth
    |   `-- pose_dekr_hrnetw48_crowdpose.pth
    `-- rescore
        |-- final_rescore_coco_kpt.pth
        `-- final_rescore_crowd_pose_kpt.pth
```

##### modules/detect/openpifpaf

```bash
pip3 install --editable '.[dev,train,test]'
```

You will certainly need to install `torch 1.9.0` and `torchvision 0.10`.

```bash
pip install torch==1.9.0+cu111
torchvision==0.10.0+cu111
torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

##### modules/eval/PoseTrack21

First, [get the dataset](https://github.com/anDoer/PoseTrack21/tree/35bd7033ec4e1a352ae39b9522df5a683f83781b#how-to-get-the-dataset).

Then, install requirements

```bash
pip install -r modules/eval/PoseTrack21/eval/posetrack21/requirements.txt
```

If you encounter the error `Could not find library geos_c or load any of its variants`, try to install geos:

```bash
sudo apt-get install libgeos-dev
```

or

```bash
brew install geos
```

##### modules/reid/bpreid

```bash
cd modules/reid/bpbreid/
pip install -r requirements.txt
python setup.py develop
cd ../../..
```

##### modules/track/yolov5

```bash
pip install -r modules/track/yolov5/requirements.txt
```

##### bptrack

```bash
pip install -r requirements.txt
```

## Directory structure

**TODO**

Finish this once it is fixed.

**END TODO**

```text
pb-track
|-- configs
|   |-- bpreid
|   |   `-- *.yaml
|   |-- strongsort
|   |    `-- *.yaml
|   `-- *.yaml
|-- modules
|   |-- detect
|   |   `-- DEKR
|   |       `-- *
|   |-- eval
|   |   `-- PoseTrack21
|   |       `-- *
|   |-- reid
|   |   `-- bpbreid
|   |       `-- *
|   |-- track
|   |   |-- strong_sort
|   |   |   `-- *
|   |   `-- yolov5
|   |       `-- *
|-- pbtrack
...
```