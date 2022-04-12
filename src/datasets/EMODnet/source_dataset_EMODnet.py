# -*- coding: utf-8 -*-

"""Source code for the EMODnet ETOPO source dataset class."""

import os
import re
import subprocess

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset

class source_dataset_EMODnet(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "EMODnet_config.ini" )):
        """Initialize the EMODnet source dataset object."""

        etopo_source_dataset.ETOPO_source_dataset.__init__(self, "EMODnet", configfile)

    def is_active(self):
        """A switch to see if thais dataset is yet being used."""
        # Switch to 'True' when the .ini is filled out and this is ready to go.
        return False

    def asc_to_gtif(self, overwrite=False):
        """All the EMODnet files (at least the MSL ones) come in ASCII format.
        Convert them all to .tif (compressed) using gdal_translate.
        """
        dirname = os.path.abspath(os.path.join(os.path.split(self.config._configfile)[0], self.config.source_datafiles_directory))
        print(dirname)
        asc_file_regex = r"\.asc\Z"

        asc_files = sorted([os.path.join(dirname, fn) for fn in os.listdir(dirname) if re.search(asc_file_regex, fn) != None])
        print(len(asc_files), "files.")

        tif_files = [os.path.splitext(fn)[0] + ".tif" for fn in asc_files]

        for i,(asc_file, tif_file) in enumerate(zip(asc_files, tif_files)):
            print(i+1, "of", len(asc_files))

            if os.path.exists(tif_file):
                if overwrite:
                    os.remove(tif_file)
                else:
                    continue

            cmd_args = ["gdal_translate", asc_file, tif_file, "-co", 'COMPRESS=LZW', "-co", 'PREDICTOR=2' ]
            # print(" ".join(cmd_args))
            subprocess.run(cmd_args)


if __name__ == "__main__":
    EMOD = source_dataset_EMODnet()
    # EMOD.asc_to_gtif(overwrite=True)
    EMOD.get_geodataframe(verbose=True)
