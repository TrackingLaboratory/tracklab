# Available features

## Features

- [x] Detection
- [x] Pose estimation
- [x] Tracking
- [x] Re-identification
- [x] Training models
- [x] Evaluation
- [x] Visualisation
- [x] Debugging
- [x] Logging

## Tasks

- [x] Bounding box detection
- [x] Pose estimation
- [x] Appearance features extraction
- [x] Multi-Object Tracking
- [x] Re-identification
- [ ] Segmentation

## Models

### Bouding Box Detection

- [x] [MMDetection zoo](https://mmdetection.readthedocs.io/en/latest/model_zoo.html) [Deprecated]
  - Mask, Fast, Faster, Cascade R-CNN
  - SSD
  - Yolox
  - ...
- [x] [Yolo from ultralytics](https://docs.ultralytics.com/)
- [x] [RTDETR](https://github.com/lyuwenyu/RT-DETR)
- [x] [RTMDet](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet)
- [x] [YOLOX](https://arxiv.org/pdf/2107.08430)

### Pose Estimation (bottom-up & top-down)

- [x] [OpenPifPaf](https://openpifpaf.github.io/intro.html) [Deprecated]
- [x] [MMPose zoo](https://mmpose.readthedocs.io/en/latest/) [Deprecated]
  - HRNet
  - LiteHRNet
  - HRFormer
  - HigherHRNet
  - DEKR
  - ...
- [x] [Yolo pose from ultralytics](https://docs.ultralytics.com/tasks/pose/)
- [x] [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)
- [x] [RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo)
- [x] [VitPose](https://github.com/ViTAE-Transformer/ViTPose)

### Apparence features extraction

- [x] [BPBReID](https://arxiv.org/abs/2211.03679)
- [x] [KPReID](https://arxiv.org/pdf/2407.18112)

### Trackers

- [x] [StrongSORT](https://arxiv.org/abs/2202.13514)
- [x] [OCSort](https://arxiv.org/abs/2203.14360)
- [x] [Deep OCSORT](https://arxiv.org/pdf/2302.11813)
- [x] [ByteTrack](https://arxiv.org/abs/2110.06864)
- [x] [BotSort](https://arxiv.org/abs/2206.14651)

## Datasets

- [x] [PoseTrack21](https://openaccess.thecvf.com/content/CVPR2022/papers/Doring_PoseTrack21_A_Dataset_for_Person_Search_Multi-Object_Tracking_and_Multi-Person_CVPR_2022_paper.pdf)
- [x] [PoseTrack18](https://openaccess.thecvf.com/content_cvpr_2018/html/Andriluka_PoseTrack_A_Benchmark_CVPR_2018_paper.html)
- [x] [MOT20](https://motchallenge.net/data/MOT20/)
- [x] [MOT17](https://motchallenge.net/data/MOT17/)
- [x] [SportsMOT](https://github.com/MCG-NJU/SportsMOT)
- [x] [DanceTrack](https://github.com/DanceTrack/DanceTrack)
- [x] [BEE24](https://ieeexplore.ieee.org/document/10851814)
- [x] [Soccernet-GS](https://arxiv.org/pdf/2404.11335)
- [x] [Soccernet](https://www.soccer-net.org/)

## Visualisations
- [x] Detections
- [x] Pose estimation
- [x] Tracking (Ellipses, Bounding boxes, Tracking Lines)
- [x] Debugging stats
