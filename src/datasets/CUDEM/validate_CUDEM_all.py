# -*- coding: utf-8 -*-

"""
validate_CUDEM_CONUS.py

Script to automate going throug the various regions of the CUDEM CONUS tiles
and validating each region separately against ICESat-2. I can't do it all at once
since it's so much data covering such a huge, wide area.
"""
import os
import argparse
import subprocess
import sys
import traceback
import re

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import utils.configfile
import icesat2.validate_dem_collection
import utils.traverse_directory
import icesat2.plot_validation_results

cudem_config = utils.configfile.config(os.path.join(THIS_DIR, "CUDEM_config.ini"))
etopo_config = utils.configfile.config(os.path.abspath(os.path.join(THIS_DIR, "..","..","..","etopo_config.ini")))

# Give a human-readable place name for each of the sub-regions
conus_place_names_dict = {"AK": "Alaska",
                        "AL_nwFL": "Alabama & NW Florida",
                        "chesapeake_bay": "Chesapeake Bay",
                        "FL": "Florida",
                        "LA_MS": "Louisiana and Mississippi",
                        "MA_NH_ME": "Massachusetts, New Hampshire and Maine",
                        "NC": "North Carlonia",
                        # "northeast_sandy": "Northeast, post-Sandy",
                        # "prusvi": "Puerto Rico and US Virgin Islands",
                        "CA": "California",
                        "OR": "Oregon",
                        "rima": "Rhode Island and Massachusetts",
                        "southeast": "Southeast US",
                        "TX": "Texas",
                        "wash_bellingham": "Washington, Bellingham",
                        "wash_juandefuca": "Washington, Straight of Juan de Fuca",
                        "wash_outercoast": "Washington, Outer Coast"}

# Subdirs with each region
# All of these already have converted DEMs (usually in the "converted" sub-dir) that maintain
data_dirs_and_vdatums = [("AmericanSamoa", "ASVD02"),
                         ("CNMI/grids/1_9_arc_second", "NMVD03"),
                         ("CONUS/NCEI_ninth_Topobathy_2014_8483", "NAVD88"),
                         ("CONUS_Sandy/northeast_sandy", "NAVD88"),
                         ("Guam", "GUVD04"),
                         ("Hawaii", "EGM2008"),
                         ("Puerto_Rico", "PRVD02"),
                         ("US_Virgin_Islands", "VIVD09")]

def find_summary_results_h5s_in_directory(dirname, h5_regex=r"_results\.h5\Z"):
    """Given a directory (presumably with a bunch of validation results, find only the summary_results h5's, that sum up all the individual results.

    These are found because the [NAME] part of the [NAME]_results.h5 is named after a sub-directory in the tree where the file exists."""
    h5_names = utils.traverse_directory.list_files(dirname, regex_match=h5_regex)
    h5_summaries = []
    for h5 in h5_names:
        # Look specifically for .h5's that are named after one of the directories they're in. This is how we know
        # they are one of the total "summary" files and not just a single DEM result.
        h5_base = os.path.basename(h5[:re.search(h5_regex, h5).span()[0]])
        path_parts = [dn for dn in os.path.dirname(h5).split(os.sep) if len(dn) > 0]
        if h5_base in path_parts:
            h5_summaries.append(h5)

    return h5_summaries


def plot_all(basedir=cudem_config._abspath(cudem_config.source_datafiles_directory), cutoff_m = 25):
    master_list_of_h5s = []
    for subdir, vdatum in data_dirs_and_vdatums:
        # Don't do the Sandy DEMs separately. Combine them with the regular CONUS CU DEMs
        if subdir == "CONUS_Sandy/northeast_sandy":
            continue

        list_of_h5s = find_summary_results_h5s_in_directory(os.path.join(basedir, subdir))

        # If we're doing the CONUS dems, add the Sandy ones.
        if subdir.find("CONUS/NCEI") > -1:
            list_of_h5s.extend(find_summary_results_h5s_in_directory(os.path.join(basedir, "CONUS_Sandy/northeast_sandy")))

        # Add these summaries to the "master list" of summary h5s.
        master_list_of_h5s.extend(list_of_h5s)

        plotname = os.path.join(basedir, subdir.split(os.sep)[0] + "_summary_plot.png")

        for h5 in list_of_h5s:
            print(h5[len(basedir):])

        icesat2.plot_validation_results.plot_histogram_and_error_stats_4_panels(list_of_h5s,
                                                                                plotname,
                                                                                place_name = subdir.split(os.sep)[0],
                                                                                error_max_cutoff = cutoff_m)

    master_plotname = os.path.join(basedir, "CUDEM_summary_plot.png")

    icesat2.plot_validation_results.plot_histogram_and_error_stats_4_panels(master_list_of_h5s,
                                                                            master_plotname,
                                                                            place_name="CUDEM",
                                                                            error_max_cutoff=cutoff_m)


def validate_region(basedir,
                    region_subdir,
                    results_subdir="icesat2_results",
                    datafiles_regex = None,
                    place_name = None,
                    input_vdatum="EGM2008",
                    output_vdatum="EGM2008",
                    include_photon_validations = True,
                    reverse=False,
                    overwrite=False):
    """Validate one sub-region of ETOPO against ICESat-2 using the icesat2 validation scripts."""
    data_dir = os.path.abspath(os.path.join(basedir, region_subdir))
    results_dir = os.path.abspath(os.path.join(data_dir, results_subdir))

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
        print("Results directory", results_dir, "created.")

    # photon_h5 = os.path.join(results_dir, region_subdir + "_photons.h5")
    # If the sub_directory being used here has multiple parts, just use the first section of it.
    subdir_parts = os.path.normpath(region_subdir).split(os.sep)
    if len(subdir_parts) == 1:
        results_h5 = os.path.join(results_dir, region_subdir + "_results.h5")
    else:
        results_h5 = os.path.join(results_dir, subdir_parts[0] + "_results.h5")

    icesat2_dir = os.path.abspath(os.path.join(etopo_config.etopo_cudem_cache_directory, ".cudem_cache"))

    if not overwrite and os.path.exists(results_h5):
        print(place_name if (place_name is not None) else region_subdir, "results already generated. Moving on.\n" )
        return

    list_of_dems = utils.traverse_directory.list_files(data_dir, regex_match=datafiles_regex)

    if reverse:
        list_of_dems.reverse()
        # If we're running in reverse, do it in a temp-dir as the cwd.
        # tempdir = os.path.join(etopo_config._abspath(etopo_config.etopo_cudem_cache_directory), "temp_{0}".format(os.getpid()))
        # if not os.path.exists(tempdir):
        #     os.mkdir(tempdir)
        #
        #
        #
        # if os.path.exists(tempdir):
        #     rm_cmd = ["rm", "-rf", tempdir]
        #     print(" ".join(rm_cmd))
        #     subprocess.run(rm_cmd)

    success = False
    while not success:
        try:
            icesat2.validate_dem_collection.validate_list_of_dems(list_of_dems, photon_h5=None, use_icesat2_photon_database=True,
                                                                  results_h5=results_h5, fname_filter=datafiles_regex,
                                                                  fname_omit=None, output_dir=results_dir,
                                                                  icesat2_dir=icesat2_dir, input_vdatum=input_vdatum,
                                                                  output_vdatum=output_vdatum, overwrite=False,
                                                                  place_name=place_name, create_individual_results=True,
                                                                  skip_icesat2_download=False, delete_datafiles=False,
                                                                  include_photon_validation=include_photon_validations,
                                                                  write_result_tifs=True, shapefile_name=None,
                                                                  omit_bad_granules=True, verbose=True)
            success = True
        except AttributeError as e:
            tb = traceback.format_exc()
            # If we had an issue creating the coastline mask, sometimes it's because another process interfered with it.
            # What works is removing the coastline mask, and trying again. Do this.
            if (tb.find("coastline_mask_array = coastline_ds.GetRasterBand(1).ReadAsArray()") > -1) and \
                    (tb.find("'NoneType' object has no attribute 'GetRasterBand'") > -1):

                rm_cmd = ["rm", results_dir+"/*coastline_mask*"]
                print(" ".join(rm_cmd))
                subprocess.run(rm_cmd)
                continue
            else:
                raise e

# Default directory ... just validate the 1/9" tiles, not the 1/3" tiles (which are bathy only)
def validate_all(basedir=cudem_config._abspath(cudem_config.source_datafiles_directory),
                 reverse=False,
                 verbose=True):
    """Validate all the CUDEM regions, one at a time."""
    # Get a list of all the subdirs in this base directory (these are the various regions).
    # subdirs_list = [dirname for dirname in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, dirname))]

    output_vdatum = "EGM2008"
    datafiles_regex = cudem_config.datafiles_regex

    # First find what the original bad granules were, and then we'll see if anything was added.
    orig_bad_granule_list = icesat2.find_bad_icesat2_granules.get_list_of_granules_to_reject(verbose=verbose)

    datadir_list = []
    vdatum_list = []
    for datadir, vdatum in data_dirs_and_vdatums:
        if datadir == "CONUS/NCEI_ninth_Topobathy_2014_8483":
            datadir_list.extend([os.path.join(datadir, subdir) for subdir in conus_place_names_dict.keys()])
            vdatum_list.extend([vdatum] * len(conus_place_names_dict.keys()))
        else:
            datadir_list.append(datadir)
            vdatum_list.append(vdatum)

    if reverse:
        datadir_list.reverse()
        vdatum_list.reverse()

    for i, (subdir, input_vdatum) in enumerate(zip(datadir_list, vdatum_list)):
        # # Start with Alabama & Florida, for starters.
        # if subdir not in ("AL_nwFL", "FL"):
        # if subdir == "SF_Bay_navd88":
        #     continue
        # place_name = place_names_dict[subdir]
        print("\n===========", subdir, "{0} of {1} ===========".format(i+1, len(datadir_list)))

        # if subdir == "northeast_sandy":
        #     print("Hitting errors in post-Sandy DEMs, skipping for now.")
        #     continue
        validate_region(basedir,
                        subdir,
                        datafiles_regex = datafiles_regex,
                        place_name = None,
                        input_vdatum = input_vdatum,
                        output_vdatum = output_vdatum,
                        include_photon_validations = True,
                        reverse=reverse)

    # Run the find_bad_granules.py code to find bad granules in this dataset and add them to the whole dataset.
    # icesat2.find_bad_icesat2_granules.find_bad_granules_in_a_dataset("CUDEM_CONUS", verbose=verbose)
    # # Add any "bad granule" records to the whole dataset
    # icesat2.find_bad_icesat2_granules.create_master_list_of_bad_granules("CUDEM_CONUS", append=True, verbose=verbose)
    # # Get an updated list of granules to avoid.
    # new_bad_granule_list = icesat2.find_bad_icesat2_granules.get_list_of_granules_to_reject(verbose=verbose)
    # # Check to see if any *new* bad granules have been added to the list since running this validation.
    # if len(new_bad_granule_list) > len(orig_bad_granule_list):
    #     # *IF* bad granules are detected in this dataset, delete the results
    #     #   files that contain photons from that dataset, and run them again
    #     #   with the bad granules filtered out.
    #     deleted_list = icesat2.find_bad_icesat2_granules.check_for_and_remove_bad_granules_after_validation("CUDEM_CONUS",
    #                                                                                                         results_subdir = "icesat2_results",
    #                                                                                                         verbose = verbose)
        # Then, we'll run the fuckin' validation again.
        # if len(deleted_list) > 0:
        #     if verbose:
        #         print("*** Re-running ICESat-2 analysis after removing bad granule data. ***")
        # Re-run the icesat-2 validation

        # for subdir in subdirs_list:
        #     # # Start with Alabama & Florida, for starters.
        #     # if subdir not in ("AL_nwFL", "FL"):
        #     # if subdir == "SF_Bay_navd88":
        #     #     continue
        #     place_name = place_names_dict[subdir]
        #     print("\n===========", place_name, "===========")
        #
        #     if subdir == "northeast_sandy":
        #         print("Hitting errors in post-Sandy DEMs, skipping for now.")
        #         continue
        #     validate_region(basedir,
        #                     subdir,
        #                     datafiles_regex = datafiles_regex,
        #                     place_name = place_name,
        #                     input_vdatum = input_vdatum,
        #                     include_photon_validations = False,
        #                     output_vdatum = output_vdatum)

# def plot_regions(basedir =
#                  result_regex=r"ncei(\w+)_results\.h5")

def read_and_parse_args():
    parser = argparse.ArgumentParser(description="Validate all the CUDEM tiles. All of them.")
    parser.add_argument("--reverse", "-r", action="store_true", default=False, help="Run the tiles in reverse order. Useful for running more than 1 proc simultaneously.")
    parser.add_argument("--plot", "-p", action="store_true", default=False, help="Create summary plots.")

    return parser.parse_args()

if __name__ == "__main__":
    args= read_and_parse_args()
    if args.plot:
        plot_all()
    else:
        validate_all(reverse = args.reverse)
