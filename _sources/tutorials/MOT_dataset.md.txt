# New dataset integration [outdated]

For this tutorial we are going to integrate the [MOT20](https://motchallenge.net/data/MOT20/) dataset. 
Please refer to their documentation to download the dataset on your own machine. You can download all the 
data from [this link](https://motchallenge.net/data/MOT20.zip).

## `TrackingDataset` extension
The first step is to create a new class that extends the `TrackingDataset` class. 
This class will be used to load the data from the dataset and to provide the structure for
the pipeline to work on. The `TrackingDataset` class is defined in `tracklab/datastruct/tracking_dataset.py`.

A `TrackingDataset` is an abstract class that allows to work on a subset of the dataset. It is composed of 
`TrackingSet` which represent the different splits of the dataset. These `TrackingSet` contain the information 
about ground truth detections, images and videos metadata which are stored as of `pandas` dataframes.

Basically, you need to here define the way to load the files and fill the `pandas` dataframes that 
respectively contain the information about the detection ground truths, 
images metadata and videos metadata for each split of your dataset. The minimum required information to be 
included in the `pd.DataFrame` are the following:

| `detections`               | `image_metadatas`                  | `video_metadatas`     |
|----------------------------|------------------------------------|-----------------------|
| `[image_id, video_id, ..]` | `[video_id, file_path, frame, ..]` | `[name, nframes, ..]` |

/!\ do not forget to set the indexes of your dataframes to a unique ID. The 
update of the results accross the pipeline relies on the uniqueness of the 
indexes.

You can find the example of the MOT20 implementation of the `TrackingDataset` 
that we implemented in `tracklab/wrappers/dataset/mot20.py`. You will then 
need to add your new class to the `tracklab/wrappers/dataset/__init__.py` file.

## Config file

Our framework works with the [Hydra](https://hydra.cc/) configuration system which 
takes advantage of a hierarchical configuration via files. This is very convenient
to easily change modules.

Basically, you will need to add a `.yaml` file in `configs/dataset` which will be 
converted to a dictionary and will contain all the arguments required for your 
new class. The `_target_` element indicates the object you want to 
instantiate for the creation of your dataset. 
It  will point to our new class `tracklab.wrappers.MOT20`. You can find the configuration 
file in `configs/dataset/mot20.yaml`.

Then you have to change in the main config file (`configs/config.yaml`) the entry in 
datasets in defaults to `mot20` (the name of the new file without the extension).
