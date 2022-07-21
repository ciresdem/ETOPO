# -*- coding: utf-8 -*-

"""validate_etopo_dataset.py -- Code wrapper for validate_dem_collection.py that
focuses on validating all the data files in an ETOPO_source_dataset object.

Created by: mmacferrin on 2022.07.20
"""

import argparse
import os

####################################
# Include the base /src/ directory of thie project, to add all the other modules.
import import_parent_dir; import_parent_dir.import_src_dir_via_pythonpath()
####################################
import icesat2.validate_dem_collection
import datasets.etopo_source_dataset
