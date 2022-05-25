# -*- coding: utf-8 -*-

"""
validate_CUDEM_CONUS.py

Script to automate going throug the various regions of the CUDEM CONUS tiles
and validating each region separately against ICESat-2. I can't do it all at once
since it's so much data covering such a huge, wide area.
"""
import os

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import utils.configfile
import icesat2.validate_dem_collection

cudem_config = utils.configfile.config(os.path.join(THIS_DIR, "CUDEM_CONUS_config.ini"))
etopo_config = utils.configfile.config(os.path.abspath(os.path.join(THIS_DIR, "..","..","..","etopo_config.ini")))

# Give a human-readable place name for each of the sub-regions
place_names_dict = {"AK": "Alaska",
                    "AL_nwFL": "Alabama & NW Florida",
                    "chesapeake_bay": "Chesapeake Bay",
                    "FL": "Florida",
                    "LA_MS": "Louisiana and Mississippi",
                    "MA_NH_ME": "Massachusetts, New Hampshire and Maine",
                    "NC": "North Carlonia",
                    "northeast_sandy": "Northeast, post-Sandy",
                    "prusvi": "Puerto Rico and US Virgin Islands",
                    "rima": "Rhode Island and Massachusetts",
                    "southeast": "Southeast US",
                    "TX": "Texas",
                    "wash_bellingham": "Washington, Bellingham",
                    "wash_outercoast": "Washington, Outer Coast"}

def validate_region(basedir,
                    region_subdir,
                    results_subdir="icesat2_results",
                    datafiles_regex = None,
                    place_name = None,
                    input_vdatum="NAVD88",
                    output_vdatum="EGM2008",
                    overwrite=False):
    """Validate one sub-region of ETOPO against ICESat-2 using the icesat2 validation scripts."""
    # TODO: Fill in

    data_dir = os.path.abspath(os.path.join(basedir, region_subdir))
    results_dir = os.path.abspath(os.path.join(data_dir, results_subdir))

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
        print("Results directory", results_dir, "created.")

    # photon_h5 = os.path.join(results_dir, region_subdir + "_photons.h5")
    results_h5 = os.path.join(results_dir, region_subdir + "_results.h5")
    icesat2_dir = os.path.abspath(os.path.join(etopo_config.etopo_cudem_cache_directory, ".cudem_cache"))

    if not overwrite and os.path.exists(results_h5):
        print(place_name if (place_name is not None) else region_subdir, "results already generated. Moving on.\n" )
        return

    icesat2.validate_dem_collection.validate_list_of_dems(data_dir,
                                                         photon_h5=None,
                                                         use_icesat2_photon_database=True,
                                                         results_h5=results_h5,
                                                         fname_filter=datafiles_regex,
                                                         fname_omit=None,
                                                         output_dir=results_dir,
                                                         icesat2_dir=icesat2_dir,
                                                         input_vdatum=input_vdatum,
                                                         output_vdatum=output_vdatum,
                                                         overwrite=False,
                                                         place_name = place_name,
                                                         create_individual_results = True,
                                                         date_range=["2021-01-01","2021-12-31"],
                                                         skip_icesat2_download=False,
                                                         delete_datafiles=False,
                                                         write_result_tifs=True,
                                                         shapefile_name = None,
                                                         verbose=True)

# Default directory ... just validate the 1/9" tiles, not the 1/3" tiles (which are bathy only)
def validate_all(basedir=os.path.join(cudem_config._abspath(cudem_config.source_datafiles_directory), "NCEI_ninth_Topobathy_2014_8483")):
    """Validate all the CUDEM_CONUS regions, one at a time."""
    # Get a list of all the subdirs in this base directory (these are the various regions).
    subdirs_list = [dirname for dirname in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, dirname))]

    input_vdatum = cudem_config.dataset_vdatum_name
    output_vdatum = "EGM2008"
    datafiles_regex = cudem_config.datafiles_regex

    for subdir in subdirs_list:
        place_name = place_names_dict[subdir]
        print("\n===========", place_name, "===========")
        validate_region(basedir,
                        subdir,
                        datafiles_regex = datafiles_regex,
                        place_name = place_name,
                        input_vdatum = input_vdatum,
                        output_vdatum = output_vdatum)

if __name__ == "__main__":
    validate_all()
