# reconn.ai.ssance

Work in progress.

Repo sources:
- [Yolov5 + StrongSORT with OSNet](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet)
- [PoseTrack21](https://github.com/anDoer/PoseTrack21)
- [DEKR](https://github.com/HRNet/DEKR)


## Installation

*TODO: verify if everything is ok*

1. Clone this repo

    ```git clone "https://github.com/bstandaert/reconn.ai.ssance.git"```

2. Install DEKR dependencies:
   ```
   pip install -r DEKR/requirements.txt
   ```
3. Install [COCOAPI](https://github.com/cocodataset/cocoapi) (/!\ 'pip install cython' might be needed to install cocoapi/crowdposeapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
4. Install [CrowdPoseAPI](https://github.com/Jeff-sjtu/CrowdPose) exactly the same as COCOAPI.
5. Manage DEKR folder:
    1. Download the pretrainded models from this [URL](https://mailustceducn-my.sharepoint.com/:f:/g/personal/aa397601_mail_ustc_edu_cn/EmoNwNpq4L1FgUsC9KbWezABSotd3BGOlcWCdkBi91l50g?e=HWuluh).
    2. Move the downloaded models to the right place. The structure of the folder should looks like:

        ```
        DEKR/
        ├── experiments/
            ├── ...
        ├── lib/
            ├── ...
        ├── model/
            ├── imagenet/
                ├── *.pth
            ├── pose_coco/
                ├── *.pth
            ├── pose_crowdpose/
                ├── *.pth
            ├── rescore/
                ├── *.pth
        ├── tools/
            ├── ...
        ├── ...
        ```
6. Install BPBreID requirements:
    ```pip install -r bpbreid/requirements.txt```

7. Setup BPBreID (required for fast Cython evaluation):
    ```
    cd bpbreid
    python setup.py develop
    ``` 
9. To remove the IDE-level error 'Unresolved reference 'torchreid'' when using the 'from torchreid... import ...' statement inside BPBreID python files:
   1. In PyCharm, right click on 'bpbreid' folder, then choose 'Mark Directory as' -> 'Sources root'
   2. In VSCode, ...

10. Install Reconnaissance requirements:
     ```pip install -r requirements.txt```

11. Download [PoseTrack21](https://github.com/anDoer/PoseTrack21) dataset. You can refer to [their documentation](https://github.com/anDoer/PoseTrack21#how-to-get-the-dataset) for the instructions. The structure of the folder should looks like:
     ```
     PoseTrack21/
     ├── baselines/
         ├── ...
     ├── doc/
         ├── ...
     ├── eval/
         ├── ...
     ├── download_dataset.py
     ├── ...
     ```
12. Weights from ```strong_sort/``` folder should be downloaded automatically. The structure of the folder should looks like:
     ```
     strong_sort/
     ├── configs/
         ├── ...
     ├── deep/
         ├── ...
     ├── results/
         ├── ...
     ├── sort/
         ├── ...
     ├── utils/
         ├── ...
     ├── weights/
         ├── *.pt
     ├── results/
         ├── ...
     ├── __init__.py
     ├── reid_multibackend.py
     ├── strong_sort.py
     ├── ...
     ```
*TODO: check if this is correct.*

11. YOLOv5 is not yet been implemented. *TODO: implement yolov5 module for track.py and val.py.*

## Repo structure

Your repository should look like:
```
reconn.ai.ssance/
├── DEKR/
    ├── ...
├── PoseTrack21/
    ├── ...
├── strong_sort/
    ├── ...
├── yolov5/
    ├── ...
├── datasets.py
├── dekr2detections.py
├── detections.py
├── track.py
├── val.py
├── ...
```