Integrating a new module [outdated]
===================================

This tutorial aims to integrate a new pose detector in the framework. 
[Yolov8](https://docs.ultralytics.com/tasks/pose/) proposes a pose detector 
which takes as input an image and provides as output the positions of the 
detected joints and the bounding boxes. To do this, we will start by extending 
the abstract class :class:`ImageLevelModule<tracklab.pipeline.imagelevel_module.ImageLevelModule>`.

.. note ::

 The first thing we need to do is determine at what level we want to run the module :

  * :class:`VideoLevelModule<tracklab.pipeline.videolevel_module.VideoLevelModule>` : You get all the detections from a video at once.
  * :class:`ImageLevelModule<tracklab.pipeline.imagelevel_module.ImageLevelModule>` : You get all the detections from a single image.
  * :class:`DetectionLevelModule<tracklab.pipeline.detectionlevel_module.DetectionLevelModule>` : You get the information of a single detection

`ImageLevelModule` extension
----------------------------
The abstract class :class:`ImageLevelModule<tracklab.pipeline.imagelevel_module.ImageLevelModule>` is defined in the file `pipeline/imagelevel_module.py`.
It allows to define a common interface for the different detectors and the 
automation of the aggregation of the results in the pipeline.

Basically, its role is to take as input an image and to provide as output the 
`detections` objects that the model will have made.

It will be necessary to define the functions `__init__`, `preprocess` and 
`process` whose expected behavior is described in the docstring of the class.

The `__init__` function will be called once in `main.py` when the model is 
initialized. 

The `preprocess` function is called at each iteration of the pipeline, it allows 
to define what is the expected behavior of the `dataloader`. It is in this function 
that we load the image, the metadata and apply the processing (if required) before 
providing it to the `process` function. The default behavior of the 
collate function is that of Pytorch, but it can of course be modified by changing 
the `self.collate_fn` attribute. Finally, it takes as argument a `pd.Series` object 
which contains the metadata related to the image, and a `pd.DataFrame` which contains
the detections from the previous steps (if applicable).

The `process` function is called after the batching of the dataloader via the 
`self.collate_fn` function. It takes as argument the batch and the metadata linked 
to the input images. This is where the processing by the model must be done. 
This function must return the results of the detections in one of the following 
forms: `Union[pd.Series, List[pd.Series], pd.DataFrame]`. The mechanism for updating 
the results during the pipeline is based on the index of the `detections`. 
This is why it is important to assign a unique `id` to each object. This can be done by 
incrementing the `self.id` variable. This `id` should then be added as a `name` 
argument in the case of a `pd.Series` or as an `index` in the case of a `pd.DataFrame`. 
Finally, it is expected that at least the following attributes are assigned in 
the `detections`: `image_id`, `video_id`, `category_id`, `bbox_ltwh` and `bbox_conf`. 
Of course other elements can be added at this stage if needed.

An example of the implementation for Yolov8-Pose is available in the file 
`wrappers/detect_multi/yolov8_pose.py`. You will also need to add this new class 
to the `wrappers/detect_multi/__init__.py` file.

Config file
-----------

Our framework works with the [Hydra](https://hydra.cc/) configuration system which 
takes advantage of a hierarchical configuration via files. This is very convenient
to easily change modules.

You will have to create a new `.yaml` configuration file, add the required 
arguments in the `cfg` field and link the `_target_` to the new object for the 
instantiation. The completed example is available in 
`configs/modules/pose_bottomup/yolov8_pose.yaml`.

You will also have to change in the main config file `configs/config.yaml` the field
`modules/pose_bottomup: yolov8_pose` (name of the new file) to infer with the new
model.