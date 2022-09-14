# -*- coding: utf-8 -*-

"""Source code for the NOAA_estuarine ETOPO source dataset class."""

import os

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset

class source_dataset_NOAA_estuarine(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "NOAA_estuarine_config.ini" )):
        """Initialize the NOAA_estuarine source dataset object."""

        super(source_dataset_NOAA_estuarine, self).__init__("NOAA_estuarine", configfile)

if __name__ == "__main__":
    noaa = source_dataset_NOAA_estuarine()
    # noaa.print_unique_vdatum_ids()
    # noaa.move_estuarine_tiles()
    # noaa.convert_to_gtiff()
    # noaa.convert_to_wgs84_and_egm2008(overwrite=False)
    noaa.get_geodataframe()