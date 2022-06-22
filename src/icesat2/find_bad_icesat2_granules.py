# -*- coding: utf-8 -*-

"""find_bad_icesat2_granules.py -- code for identifying and eliminating bad ICESat-2 granules from analyses."""

import os

####################################3
# Include the base /src/ directory of thie project, to add all the other modules.
import import_parent_dir; import_parent_dir.import_src_dir_via_pythonpath()
####################################3
import icesat2.validate_dem as validate_dem
import icesat2.atl_granules as atl_granules
import utils.configfile

my_config = utils.configfile.config()

def create_bad_granules_csv(bad_granule_list_fname = my_config._abspath(my_config.icesat2_bad_granules_list),
                            overwrite=False,
                            verbose=True):
    """Create an empty CSV file with a record of the bad granules we've identified."""


def add_file_to_bad_granule_list(granule_fname,
                                 bad_granule_list_fname = my_config._abspath(my_config.icesat2_bad_granules_list)):
    """If we've found a bad granule in the list, add it to the list of bad granules.
    In this case we're using the ATL03..._photons.h5 files."""

def identify_bad_granules_in_single_validation(dem_fname,
                                               add_to_list = True,
                                               verbose=True):
    """Run a single-validation on a DEM, on a photon by photon basis.
    Identify granules where the photon errors are consistently outside the ranges
    of others by more than 2 s.d. (3 s.d.?) away from the mean range of the overall photons."""

def get_summary_h5_fname_from_tilename(dem_fname):
    """Given a DEM fname, return the file path of validation results file in the granule_validation_results directory."""
    fname_short = os.path.split(dem_fname)
    fname_base, ext = os.path.splitext(fname_short)
    return os.path.join(my_config.icesat2_granule_validation_results_directory,
                        fname_base + "_granule_results.h5")


def generate_tile_granule_validation_summary(dem_fname,
                                             dem_vdatum,
                                             vdatum_out = "EGM2008",
                                             summary_h5_fname = None,
                                             overwrite = False,
                                             verbose = True):
    """Run a validation on the fname and tally up granule-level photon-by-photon
    validation results for each granule within the DEM bounding box.

    Return a dataframe with the results for that tile, delineated by granules within that tile.
    Also, write the dataframe out to an HDF5 in the granule_validation_results directory.
    """
    # Get the name of our summary file from the dem filename.
    if summary_h5_fname is None:
        summary_h5_fname = get_summary_h5_fname_from_tilename(dem_fname)

    # Check to make sure it's not already in there (skip if it is, unless we've specified to overwrite it.)
    if os.path.exists(summary_h5_fname):
        if overwrite:
            if verbose:
                print("Removing old", os.path.split(summary_h5_fname)[1] + ".")
            os.remove(summary_h5_fname)
        else:
            if verbose:
                print(os.path.split(summary_h5_fname)[1], "already exists.")
            return

    # TODO 1. Get the photon validation dataframe for that tile *on a photon basis*. *Not* aggregated by grid-cell.
    # TODO 2. Loop through all the unique granules and organize the photons by those.
    # TODO 3: Calculate the mean, std, range of



if __name__ == "__main__":
    pass
