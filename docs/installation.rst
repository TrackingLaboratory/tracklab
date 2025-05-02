.. _installation:

Installation guide
=======================

Clone the repository
--------------------

.. code:: bash

   git clone https://github.com/TrackingLaboratory/tracklab.git
   cd tracklab

Manage the environment
----------------------

Create and activate a new environment with uv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash
   uv venv --python 3.12
   uv sync

You might need to change your torch installation depending on your
hardware. Please check on `Pytorch
website <https://pytorch.org/get-started/previous-versions/>`_ to find
the right version for you.

.. note::

 You can also install the framework in a regular python virtual environment, and can then install
 the dependencies directly, but make sure that you install it in a *separate* environment !

Install the dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

Get into your repo and install the requirements with :

.. code:: bash
    uv sync


Run your first tracklab pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the default configuration (on an example DanceTrack video) :

.. code:: bash
    uv run tracklab