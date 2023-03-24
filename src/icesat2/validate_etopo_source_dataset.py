# -*- coding: utf-8 -*-

"""validate_etopo_dataset.py -- Code wrapper for validate_dem_collection.py that
focuses on validating all the data files in an ETOPO_source_dataset object.

Created by: mmacferrin on 2022.07.20
"""

import argparse

####################################
# Include the base /src/ directory of thie project, to add all the other modules.
import import_parent_dir; import_parent_dir.import_src_dir_via_pythonpath()
####################################
import icesat2.validate_dem_collection
import icesat2.find_bad_icesat2_granules
import datasets.etopo_source_dataset

def validate_dataset(dataset_name,
                     output_vdatum = "egm2008",
                     results_subdir = "icesat2_results",
                     detect_bad_granules = True,
                     plot_outputs = True,
                     verbose = True):
    """Take an ETOPO dataset and validate the entire thing against the ICEsat-2 photon database.

    This is basically a wrapper for validate_dem_collection.py, but takes the various
    arguments from the source dataset config.ini file to run it.

    If detect_bad_granules, run the find_bad_granules.py code on it after validation
    and go from there.
    """
    # Create source dataset object
    dset = datasets.etopo_source_dataset.get_source_dataset_object(dataset_name, verbose=verbose)
    if dset is None:
        return

    # Get the list of files.
    dem_list = dset.retrieve_all_datafiles_list(verbose=verbose)
    # Get the other metadata variables needed.
    input_vdatum = dset.config.dataset_vdatum_name
    dset_name = dset.config.dataset_name
    # If we have a separate regex for land-only data files to validate, use that.
    if hasattr(dset.config, "datafiles_to_validate_regex"):
        dem_regex = dset.config.datafiles_to_validate_regex
    else:
        # Otherwise, just use the default regex.
        dem_regex = dset.config.datafiles_regex

    # First find what the original bad granules were, and then we'll see if anything was added.
    orig_bad_granule_list = icesat2.find_bad_icesat2_granules.get_list_of_granules_to_reject(verbose=verbose)

    # Run the icesat-2 validation
    icesat2.validate_dem_collection.validate_list_of_dems(dem_list, None, use_icesat2_photon_database=True,
                                                          results_h5=None, fname_filter=dem_regex,
                                                          output_dir=results_subdir, input_vdatum=input_vdatum,
                                                          output_vdatum=output_vdatum, place_name=dset_name,
                                                          create_individual_results=True, skip_icesat2_download=True,
                                                          include_photon_validation=True, omit_bad_granules=True,
                                                          verbose=verbose)
    # Run the find_bad_granules.py code to find bad granules in this dataset and add them to the whole dataset.
    icesat2.find_bad_icesat2_granules.find_bad_granules_in_a_dataset(dset, verbose=verbose)
    # Add any "bad granule" records to the whole dataset
    icesat2.find_bad_icesat2_granules.create_master_list_of_bad_granules(dset, append=True, verbose=verbose)
    # Get an updated list of granules to avoid.
    new_bad_granule_list = icesat2.find_bad_icesat2_granules.get_list_of_granules_to_reject(verbose=verbose)
    # Check to see if any *new* bad granules have been added to the list since running this validation.
    if len(new_bad_granule_list) > len(orig_bad_granule_list):
        # *IF* bad granules are detected in this dataset, delete the results
        #   files that contain photons from that dataset, and run them again
        #   with the bad granules filtered out.
        deleted_list = icesat2.find_bad_icesat2_granules.check_for_and_remove_bad_granules_after_validation(dset,
                                                                                                            results_subdir = results_subdir,
                                                                                                            verbose = verbose)
        # Then, we'll run the fuckin' validation again.
        if len(deleted_list) > 0:
            if verbose:
                print("*** Re-running ICESat-2 analysis after removing bad granule data. ***")
        # Re-run the icesat-2 validation
        icesat2.validate_dem_collection.validate_list_of_dems(dem_list, None, use_icesat2_photon_database=True,
                                                              results_h5=None, fname_filter=dem_regex,
                                                              output_dir=results_subdir, input_vdatum=input_vdatum,
                                                              output_vdatum=output_vdatum, place_name=dset_name,
                                                              create_individual_results=True,
                                                              skip_icesat2_download=True,
                                                              include_photon_validation=True, omit_bad_granules=True,
                                                              verbose=verbose)

    return

def define_and_parse_args():
    parser = argparse.ArgumentParser(description = "Validate an entire ETOPO DEM dataset against ICESat-2.")
    parser.add_argument("NAME", help="Name of the dataset. Look at the src/datasets directory for valid ETOPO input datasets.")
    parser.add_argument("-output_vdatum", "-ov", default="egm2008", help="The output vertical datum against which to validate this dataset. Default: egm2008")
    parser.add_argument("-results_subdir", "-rd", default="icesat2_results", help="A sub-directory in which to put the validation results and files. Default 'icesat2_results/'")
    parser.add_argument("--quiet", "-q", default=False, action="store_true", help="Run in quiet mode.")

    return parser.parse_args()

if __name__ == "__main__":
    args = define_and_parse_args()

    validate_dataset(args.NAME,
                     output_vdatum = args.output_vdatum,
                     results_subdir = args.results_subdir,
                     verbose = not args.quiet)
