
import os
import re

THIS_DIR = os.path.dirname(__file__)

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################
import utils.traverse_directory
import utils.configfile
cudem_config = utils.configfile.config(os.path.join(THIS_DIR, "CUDEM_config.ini"))
etopo_config = utils.configfile.config()

# A dictionary giving each region name, and a list of directories where to look for the ICESat-2 results files within that region
region_subdirs = {"CONUS and Alaska": ["CONUS", "CONUS_Sandy"],
                  "Hawaii": ["Hawaii"],
                  "Puerto Rico": ["Puerto_Rico"],
                  "US Virgin Islands": ["US_Virgin_Islands"],
                  "Guam": ["Guam"],
                  "American Samoa": ["AmericanSamoa"],
                  "Central Northern Mariana Islands": ["CNMI"]}

results_h5_regex = "ncei(\w+)_results.h5"

def plot_regions():
    for region in sorted(list(regions.keys())):
        for subdir in region_subdirs[region]:
            subdir_path = os.path.join(cudem_config.source_datafiles_directory, subdir)
            files = utils.traverse_directory.