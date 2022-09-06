# -*- coding: utf-8 -*-

"""Source code for the CUDEM_Puerto_Rico ETOPO source dataset class."""

import os

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset

class source_dataset_CUDEM_Puerto_Rico(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "CUDEM_Puerto_Rico_config.ini" )):
        """Initialize the CUDEM_Puerto_Rico source dataset object."""

        super(source_dataset_CUDEM_Puerto_Rico, self).__init__("CUDEM_Puerto_Rico", configfile)

if __name__ == "__main__":
    ds = source_dataset_CUDEM_Puerto_Rico()
    ds.convert_vdatum(numprocs=10)

    ds = source_dataset_CUDEM_Puerto_Rico()
    ds.reproject_tiles_from_nad83()