# -*- coding: utf-8 -*-

"""Source code for the ShallowBathyEverywhere ETOPO source dataset class."""

import os

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset

class source_dataset_ShallowBathyEverywhere(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "ShallowBathyEverywhere_config.ini" )):
        """Initialize the ShallowBathyEverywhere source dataset object."""

        super(source_dataset_ShallowBathyEverywhere, self).__init__("ShallowBathyEverywhere", configfile)

if __name__ == "__main__":
    gdf = source_dataset_ShallowBathyEverywhere().get_geodataframe()