# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = [
    'distances',
    'io',
    'lap',
    'metrics',
    'utils',
    'MOTAccumulator',
]

from . import distances
from . import io
from . import lap
from . import metrics
from . import utils
from .mot import MOTAccumulator

# Needs to be last line
__version__ = '1.2.0'
