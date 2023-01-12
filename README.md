# PbTrack

Work in progress

## Installation guide[^1]

[^1]: Tested on `conda 22.11.1`, `Python 3.10.8`, `pip 22.3.1`, `g++ 11.3.0` and `gcc 11.3.0`

### Clone the repository

```bash
git clone https://github.com/PbTrack/pb-track.git --recurse-submodules
cd pb-track
```

If you cloned the repo without using the `--recurse-submodules` option, you can still download the submodules with :

```bash
git submodule update --init --recursive
```

### Manage the environment

#### Create and activate a new environment

```bash
conda create -y --name "pbtrack" python pip numpy
conda activate pbtrack
```

#### Install the dependencies
Get into your repo and install the requirements with :

```bash
pip install -r requirements.txt
```

#### Setup reid

```bash
cd plugins/reid/bpbreid/
python setup.py develop
```

### External dependencies

- Get the **PoseTrack21** dataset [here](https://github.com/anDoer/PoseTrack21/tree/35bd7033ec4e1a352ae39b9522df5a683f83781b#how-to-get-the-dataset).
- Get the pretrained weights of **BPBReID** [here](https://github.com/VlSomers/bpbreid#download-the-pre-trained-models).
