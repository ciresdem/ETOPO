# -*- coding: utf-8 -*-

"""Source code for the CUDEM_CONUS_Sandy ETOPO source dataset class."""

import os
import pyproj

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset

class source_dataset_CUDEM_CONUS_Sandy(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "CUDEM_CONUS_Sandy_config.ini" )):
        """Initialize the CUDEM_CONUS_Sandy source dataset object."""

        super(source_dataset_CUDEM_CONUS_Sandy, self).__init__("CUDEM_CONUS_Sandy", configfile)

if __name__ == "__main__":
    ds = source_dataset_CUDEM_CONUS_Sandy()
    ds.reproject_tiles_from_nad83()