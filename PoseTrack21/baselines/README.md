# Baselines
You can find the respective scripts to run the baselines in `baselines/scripts/posetracking/`. Each script contains pre-defined paths to the respective model weights and person detections. 

Please download the [required weights and detections](https://github.com/anDoer/PoseTrack21/releases/download/v0.2/PoseTrack21Release.zip) and place the folders `data` and `outputs` `into PoseTrack21/baselines`.

# Leader Board 
You can request to include your results into this leader board by generating a pull request! 
Your pull-request should include a method name and a reference to your accepted publication. 

We show the results of two protocols: 
* Protocol 1: The results are calculated on a sequence level
* Protocol 2: The results are calculated across the entire dataset

*Note that results can vary between protocols (i.e. DetA) as in protocol 1, metrics are calculated for each sequence individually and then averaged, where as in protocol 2, metrics are calculated for the entire valdation set.*

### Protocol 1: Multi-Person Re-ID Pose Tracking
| Method                 | DetA          | AssA          | FragA         | HOTA          | FA-HOTA       | 
| ------------------     | ------------- | ------------- | ------------- | ------------- | ------------- |
| CorrTrack [1]          | 45.48         | 58.02         | 57.75         | 51.13         | 51.07         |
| CorrTrack w. Reid [1]  | 46.56         | 60.21         | 59.66         | 52.71         | 52.59         |
| Tracktor w. Pose [1]   | 46.30         | 59.41         | 58.61         | 52.21         | 52.03         |
| Tracktor w. Corr [1]   | 44.67         | 54.05         | 54.05         | 48.90         | 48.43         |

### Protocol 2: Multi-Person Re-ID Pose Tracking with lookup table

| Method                  | DetA          | AssA          | FragA         | HOTA          | FA-HOTA       | 
| ------------------      | ------------- | ------------- | ------------- | ------------- | ------------- |
| CorrTrack [1]           | 45.40         | 48.34         | 48.11         | 46.48         | 46.40         |
| CorrTrack w. Reid [1]   | 46.48         | 50.19         | 49.74         | 47.93         | 47.82         |
| Tracktor w. Pose [1]    | 46.24         | 49.56         | 48.88         | 47.49         | 47.32         |
| Tracktor w. Corr [1]    | 44.62         | 44.65         | 42.96         | 44.23         | 43.79         |

In an additional post-processign step, we implemented a naive lookup-table based approach to associate tracks across the entire validation set. 
The code will be released soon.

## References 
```
[1]  @inproceedings{doering22,
     title={Pose{T}rack21: {A} Dataset for Person Search, Multi-Object Tracking and Multi-Person Pose Tracking},
     author={Andreas Doering and Di Chen and Shanshan Zhang and Bernt Schiele and Juergen Gall},
     booktitle={CVPR},
     year={2022}
}
```
