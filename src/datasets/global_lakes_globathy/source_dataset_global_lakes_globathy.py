# -*- coding: utf-8 -*-

"""Source code for the global_lakes_globathy ETOPO source dataset class."""

import os

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset

class source_dataset_global_lakes_globathy(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "global_lakes_globathy_config.ini" )):
        """Initialize the global_lakes_globathy source dataset object."""

        super(source_dataset_global_lakes_globathy, self).__init__("global_lakes_globathy", configfile)