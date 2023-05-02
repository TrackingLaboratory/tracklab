# New evaluator integration

The evaluator must extend the abstract class `EvaluatorBase` which is described in `core/evaluator.py`. 
Specifically, the function `run(tracker_state)` must be implemented for the evaluation to be done.

## `EvaluatorBase` extension

The `tracker_state` object is the data structure that aggregates all the 
information and results related to the tracking. 
It contains in particular the `predicitions`, the metadata related to the 
dataset `image_metadas` and `video_metadas` but also the ground truths
`detections_gt`, which were created during the dataset initialization. 
All these attributes are `pd.DataFrame`, which makes sorting for evaluation 
much easier.

You can find an example of an evaluator for the MOT challenge in the file 
`wrappers/eval/mot/mot20_evaluator.py`.

## Config file

Our framework works with the [Hydra](https://hydra.cc/) configuration system which 
takes advantage of a hierarchical configuration via files. This is very convenient
to easily change modules.

Again, don't forget to add the evaluator in the `__init__.py` file and create 
the configuration file. As example, the new created one is available in file 
`configs/eval/mot20.yaml`. 

You will also have to change in the main config file `configs/config.yaml` the field
`eval: mot20` to evaluate with your new evaluator on this dataset.
