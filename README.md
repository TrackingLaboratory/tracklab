# reconn.ai.ssance

Work in progress.

Repo sources:
- [Yolov5 + StrongSORT with OSNet](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet)
- [PoseTrack21](https://github.com/anDoer/PoseTrack21)
- [DEKR](https://github.com/HRNet/DEKR)


## Installation

TODO: verify if everything is ok

1. Clone this repo

    ```git clone "https://github.com/bstandaert/reconn.ai.ssance.git"```

2. Install the requirements.

    ```pip install -r requirements.txt```

3. Manage DEKR folder
    1. Download the pretrainded models from this [URL](https://mailustceducn-my.sharepoint.com/:f:/g/personal/aa397601_mail_ustc_edu_cn/EmoNwNpq4L1FgUsC9KbWezABSotd3BGOlcWCdkBi91l50g?e=HWuluh)
    2. Move the downloaded models to the right place. The structure of the folder should looks like:

        ```
        DEKR/
        ├── experiments/
        ├── lib/
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
        ```
4. Download [PoseTrack21](https://github.com/anDoer/PoseTrack21) dataset. You can refer to [their documentation](https://github.com/anDoer/PoseTrack21#how-to-get-the-dataset) for the instructions. The structure of the folder should looks like:
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
5. Weights from ```strong_sort/``` folder should be downloaded automatically. TODO: check if this is correct. The structure of the folder should looks like:
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

6. Yolov5 is not yet been implemented. TODO: implement yolov5 module for track.py and val.py.

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