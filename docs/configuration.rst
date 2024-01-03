PbTrack Configuration
=====================

This framework uses `Hydra <http://hydra.cc/>`_ for its configuration.

Hydra combines commandline arguments with configuration directories,
to create a configuration dictionary. The config tree is determined by
the directory structure, and each file in a directory is a configuration choice.

By default, the main configuration file looks like this :

.. include:: ../configs/config.yaml
    :code: yaml

The `defaults` list in this file contains the default value for every category.
You can change these defaults either through the commandline.
For example, to change the dataset and the image detector, you could do::

    python -m tracklab.main dataset=posetrack18 detect_multiple=openpifpaf

If you want to change a single value inside a category, for example the bbox extension factor::

    python -m tracklab.main detect_multiple=openpifpaf "detect_multiple.cfg.bbox.extension_factor=[0.05,0.05,0.05]"

This will both change the detector to ``openpifpaf`` and change a parameter inside the openpifpaf configuration.

You can find out which values you can change by looking at the configuration directory or running::

    python -m tracklab.main --help

Thanks to Hydra, we can also run a grid-search (or other kind of hyper-parameter searches). The default grid-search
is activated with the ``-m`` flag, and by passing a comma-separated list of values, either at the category-level
or at the key-level::

    python -m tracklab.main -m dataset=posetrack18,posetrack21 detect_multiple=openpifpaf,yolox track.cfg.max_dist=0.2,0.5,0.7

This command will run tracking multiple times, to test two datasets, two image-level detectors with three different
tracking hyper-parameters (combining to 12 runs).

.. hint::

    The first ``-m`` is a python flag to run a module, while the second ``-m`` is the multirun flag from hydra


Configuration directory
-----------------------

The configuration for tracklab is located in the ``configs/`` directory, with the following structure::

    configs
    ├── dataset
    ├── detect_multiple
    ├── detect_single
    ├── reid
    ├── track
    ├── engine
    ├── eval
    ├── machine
    ├── state
    ├── visualization
    └── config.yaml

For example, when adding a new dataset, you should add a new file inside ``configs/dataset/``, which will be the basis
for your new dataset.

Automatic initialization
------------------------

In the code, we use :func:`hydra.initialize` in order to find the right class to initialize or function to call. This
requires the configuration to have a specific key, ``_target_``, which must contain the full name (including modules) of
