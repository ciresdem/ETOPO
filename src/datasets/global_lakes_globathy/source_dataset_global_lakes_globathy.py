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

    def create_global_lakes_globathy_tiles(self,
                                           resolution_s = 15,
                                           crm_only_if_1s = True,
                                           verbose = True):

        # 1) Get the tile outlines and resolutions of each grid to create. (Use the etopo empty tiles to do this.)
        etopo_config = utils.configfile.config()
        if resolution_s == 1:
            etopo_gpkg_name = etopo_config.etopo_tile_geopackage_1s
        elif resolution_s == 15:
            etopo_gpkg_name = etopo_config.etopo_tile_geopackage_15s
        elif resolution_s == 60:
            etopo_gpkg_name = etopo_config.etopo_tile_geopackage_60s
        else:
            raise ValueError("Unhandled value for parameter 'resolution_s':", resolution_s)

        # 2) Get the directory of the output files and filenames for each output


        # 3) Generate all the output grids. Lakes everywhere!

        # 4) Then think about generating the gebco outlines where they should exist.
