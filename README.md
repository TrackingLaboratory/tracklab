# PbTrack

Work in progress

## Installation guide

(tested on `Python 3.9.12`, `conda 22.11.1`, `pip 22.3.1`, `g++ 11.3.0`, `gcc 11.3.0`)

### Clone the repository

```bash
git clone https://github.com/PbTrack/pb-track.git --recurse-submodules
```

or

```bash
git clone git@github.com:PbTrack/pb-track.git
cd pb-track
git submodule update --init --recursive
```

### Manage the environment

#### Create and activate a new environment

```bash
conda create -n "pbtrack"
conda activate pbtrack
```

#### Install the dependencies
Get into your repo and install the requirements.

```bash
cd pb-track
pip install -r requirements.txt
```

### External dependencies

- Get the **PoseTrack21** dataset [here](https://github.com/anDoer/PoseTrack21/tree/35bd7033ec4e1a352ae39b9522df5a683f83781b#how-to-get-the-dataset).
- Get the pretrained weights of **BPBReID** [here](https://github.com/VlSomers/bpbreid#download-the-pre-trained-models).
