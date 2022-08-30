# -*- coding: utf-8 -*-

"""Source code for the BlueTopo_19N ETOPO source dataset class."""

import os

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset
import datasets.BlueTopo.source_dataset_BlueTopo as BlueTopo

class source_dataset_BlueTopo_19N(BlueTopo.source_dataset_BlueTopo):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "BlueTopo_19N_config.ini" )):
        """Initialize the BlueTopo_19N source dataset object."""

        etopo_source_dataset.ETOPO_source_dataset.__init__(self, "BlueTopo_19N", configfile)

if __name__ == "__main__":
    bt19 = source_dataset_BlueTopo_19N()