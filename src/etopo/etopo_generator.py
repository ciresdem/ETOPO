# -*- coding: utf-8 -*-

"""etopo_dataset.py -- parent code for managing, building and assessing/validating
   the ETOPO global topo-bathy DEM."""
import os

class ETOPO_Generator:

    # Tile naming conventions. ETOPO_2022_v1_N15E175.tif, e.g.
    fname_template_tif = r"ETOPO_2022_v1_{0:s}{1:02d}{2:s}{3:03d}.tif"
    fname_template_netcdf = r"ETOPO_2022_v1_{0:s}{1:02d}{2:s}{3:03d}.nc"

    # The base directory of the project is two levels up. Retrieve the absolute path to it on this machine.
    # This file resides in [project_basedir]/src/etopo
    project_basedir = os.path.abspath(os.path.join(os.path.split(__file__)[0], "..", ".."))
    empty_tiles_directory = os.path.join(project_basedir, "data", "empty_tiles") # Note, still need to add resolution "1s" or "15s" directory.
    source_tiles_directory = os.path.join(project_basedir, "data", "source_tiles") # Same as above
    finished_tiles_directory = os.path.join(project_basedir, "data", "finished_tiles") # Same as above

    def __init__(self):
        """Read the configuration file, get the info about our grid locations."""
        self.datasets = None

    def create_empty_grids(self, resolution_s=(1,15)):
        """Create empty grid-cells for the ETOPO data product.
        resolution_s can be 1, 15, or a list/tuple with both values (1,15).
        """
        # TODO: Finish
        pass

    def create_list_of_input_datasets(self):
        """Look through the /src/datasets/ directory, and get a list of the input datasets to use.
        This list will be saved to self.source_datasets, and will be instances of
        the datasets.source_dataset class, or sub-classes thereof.
        Each subclass must be called source_dataset_XXXX.py where XXX refers to the name
        of the dataset specified by the folder it's in. For instance, the CopernicusDEM
        is in the folder /src/datasets/CopernicusDEM/source_dataset_CopernicusDEM.py file.

        Any dataset that does not have a sub-class defined in such a file, will simply
        be instantiated with the parent class defined in source_dataset.py
        """
        # TODO: Finish
        # 1. Search the sub-directories in /src/datasets, anything with a
        # source_dataset_XXXXX.py class gets instantiated.
        #    Anything without the sub-class functions gets instantiated as the base class and marked "unimplemented"
        # 2. Add this list of classes to the self.datasets list.
