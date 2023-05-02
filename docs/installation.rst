.. _installation:

Installation guide [1]_
=======================

Clone the repository
--------------------

.. code:: bash

   git clone https://github.com/PbTrack/pb-track.git --recurse-submodules
   cd pb-track

If you cloned the repo without using the ``--recurse-submodules``
option, you can still download the submodules with :

.. code:: bash

   git submodule update --init --recursive

Manage the environment
----------------------

Create and activate a new environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   conda create -n pbtrack pip python=3.10 pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
   conda activate pbtrack

You might need to change your torch installation depending on your
hardware. Please check on `Pytorch
website <https://pytorch.org/get-started/previous-versions/>`_ to find
the right version for you.

Install the dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

Get into your repo and install the requirements with :

.. code:: bash

   pip install -r requirements.txt
   mim install mmcv-full

Note: if you re-install dependencies after pulling the last changes, and
a new git submodule has been added, do not forget to recursively update
all the submodule before running above commands:

.. code:: bash

   git submodule update --init --recursive

Setup reid
~~~~~~~~~~

.. code:: bash

   cd plugins/reid/bpbreid/
   python setup.py develop

.. [1]
   Tested on ``conda 22.11.1``, ``Python 3.10.8``, ``pip 22.3.1``,
   ``g++ 11.3.0`` and ``gcc 11.3.0``