.. _installation:

Installation guide [1]_
=======================

Clone the repository
--------------------

.. code:: bash

   git clone https://github.com/PbTrack/pb-track.git
   cd pb-track

Manage the environment
----------------------

Create and activate a new environment with conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   conda create -n tracklab pip python=3.10 pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
   conda activate tracklab

You might need to change your torch installation depending on your
hardware. Please check on `Pytorch
website <https://pytorch.org/get-started/previous-versions/>`_ to find
the right version for you.

.. note::

 You can also install the framework in a regular python virtual environment, and can then install
 the dependencies directly, but make sure that you install it in a *separate* environment !

 For advanced users, this project can also be installed with `Poetry <https://python-poetry.org/>`_.

Install the dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

Get into your repo and install the requirements with :

.. code:: bash

   pip install -e .
   mim install mmcv-full


.. [1]
   Tested on ``conda 22.11.1``, ``Python 3.10.8``, ``pip 22.3.1``,
   ``g++ 11.3.0`` and ``gcc 11.3.0``

