# PoseTrack21 Evaluation Kit 

## Installation 
Please install the required packages specified in requirements.txt
```
pip install -r requirements.txt
```

If you want to use the evaluation kit as an API inside your projects, please consider installing the evaluation kit as follows:
```
pip install -e .
```

Alternatively, you can rely on our provided docker files. Simply run 
```
cd eval/posetrack21/docker/ && ./build.sh
```

## Evaluation 
We provide scripts for the evaluation for multi-object tracking, multi-person pose tracking, multi-person reid pose tracking and pose estimation in eval/posetrack21/experiments/

```
./eval/experiments/evaluate_*.sh $PATH_TO_GT_FOLDER $RESULT_FOLDER 
```

You can also directly run the respective python files 
```
python eval/posetrack21/scripts/run_*.py -h 
```

To run your evaluation inside docker, run 
```
eval/posetrack21/docker/eval_.sh $GT_FOLDER $EXP_FOLDER $NUM_CPU_CORES
```

___
The PoseTrack21 evaluation code build upon

```
@misc{luiten2020trackeval,
  author =       {Jonathon Luiten, Arne Hoffhues},
  title =        {TrackEval},
  howpublished = {\url{https://github.com/JonathonLuiten/TrackEval}},
  year =         {2020}
}
```
