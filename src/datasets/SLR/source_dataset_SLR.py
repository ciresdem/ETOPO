# -*- coding: utf-8 -*-

"""Source code for the SLR ETOPO source dataset class."""

import os

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset

class source_dataset_SLR(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "SLR_config.ini" )):
        """Initialize the SLR source dataset object."""

        super(source_dataset_SLR, self).__init__("SLR", configfile)
        
if __name__ == "__main__":
    pass
