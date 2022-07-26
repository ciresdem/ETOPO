# -*- coding: utf-8 -*-

"""create_global_lakes_layer.py -- Code to combine the GEBCO, GLOBathy, Copernicus,
and other layers to create lake-depth-elevations (and masks) of global inland water bodies.
"""
try:
    import cudem
    cudem
except (ImportError, ModuleNotFoundError):
    print("CUDEM executables required for this code. Go to https://github.com/ciresdem/cudem for download & documentation.")
    import sys; sys.exit(0)

import os

#####################################################
# Code snippet to import the base directory into
# PYTHONPATH to aid in importing from all the other
# modules in other subdirs.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
#####################################################
import
