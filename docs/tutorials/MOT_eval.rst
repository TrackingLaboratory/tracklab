New evaluator integration  [outdated]
=====================================

The evaluator must extend the abstract class :class:`Evaluator<tracklab.core.evaluator.Evaluator>` which is described in `core/evaluator.py`.
Specifically, the function `run(tracker_state)` must be implemented for the evaluation to be done.

## `EvaluatorBase` extension

The `tracker_state` object is the data structure that aggregates all the 
information and results related to the tracking. 
It contains in particular the `predictions`, the metadata related to the
dataset `image_metadas` and `video_metadas` but also the ground truths
`detections_gt`, which were created during the dataset initialization. 
All these attributes are `pd.DataFrame`, which makes sorting for evaluation 
much easier.

You can find an example of an evaluator for the MOT challenge in the file 
`wrappers/eval/mot/mot20_evaluator.py`. You will then have to add the new 
object evaluator in the `__init__.py` file.

Config file
-----------

Our framework works with the [Hydra](https://hydra.cc/) configuration system which 
takes advantage of a hierarchical configuration via files. This is very convenient
to easily change modules.

You will have to create a new `.yaml` configuration file, add the required 
arguments in the `cfg` field and link the `_target_` to the new object for the 
instantiation. The completed example is available in `configs/eval/mot20.
yaml`. 

You will also have to change in the main config file `configs/config.yaml` the field
`eval: mot20` to evaluate with your new evaluator.
