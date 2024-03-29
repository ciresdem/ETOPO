# -*- coding: utf-8 -*-

"""Source code for the BedMachine_Surface ETOPO source dataset class."""

import os

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset

class source_dataset_BedMachine_Surface(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "BedMachine_Surface_config.ini" )):
        """Initialize the BedMachine_Surface source dataset object."""

        super(source_dataset_BedMachine_Surface, self).__init__("BedMachine_Surface", configfile)

if __name__ == "__main__":
    gdf = source_dataset_BedMachine_Surface().get_geodataframe(resolution_s=15)
    print(gdf)
    for fname in gdf.filename:
        print(fname)