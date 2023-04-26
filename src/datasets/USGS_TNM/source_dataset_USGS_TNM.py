# -*- coding: utf-8 -*-

"""Source code for the USGS_TNM ETOPO source dataset class."""

import os

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset

class source_dataset_USGS_TNM(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "USGS_TNM_config.ini" )):
        """Initialize the USGS_TNM source dataset object."""

        super(source_dataset_USGS_TNM, self).__init__("USGS_TNM", configfile)
