# -*- coding: utf-8 -*-

"""Source code for the global_lakes ETOPO source dataset class."""

import os
import geopandas

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset
import utils.configfile

class source_dataset_global_lakes_gebco(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "global_lakes_config.ini" )):
        """Initialize the global_lakes source dataset object."""

        super(source_dataset_global_lakes, self).__init__("global_lakes", configfile)


    def create_gebco_global_lakes(self, resolution_s=15):
        """Create a global lakes outline dataset with GEBCO elevations in it.

        Test to see where this is valid and where it isn't.
        Run the create_globathy_global_lakes() method first to get all the lake outlines."""