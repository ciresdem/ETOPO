
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
import icesat2.plot_validation_results
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

results_h5_regex = r"ncei(\w+)_results.h5"
results_h5_omit = "photon_level_results"

def plot_regions():
    all_files = []
    for region in sorted(list(region_subdirs.keys())):
        files = []
        for subdir in region_subdirs[region]:
            subdir_path = os.path.join(cudem_config.source_datafiles_directory, subdir)
            files = files + utils.traverse_directory.list_files(subdir_path, regex_match=results_h5_regex, include_base_directory=True)

        files = [fn for fn in files if fn.find(results_h5_omit) == -1]

        all_files.extend(files)

        print(region, ",", len(files), "files.")
        # for fname in files:
        #     print(" ", fname[len(os.path.dirname(os.path.dirname(os.path.dirname(fname)))):])

        figname = os.path.join(cudem_config.source_datafiles_directory, region + "_validation_results.png")
        icesat2.plot_validation_results.plot_histogram_and_error_stats_4_panels(files,
                                                                                figname,
                                                                                place_name = region,
                                                                                dpi = 1200)

    print("CUDEM all,", len(all_files), "files.")
    figname_all = os.path.join(cudem_config.source_datafiles_directory, "CUDEM_all_validation_results.png")
    icesat2.plot_validation_results.plot_histogram_and_error_stats_4_panels(all_files,
                                                                            figname_all,
                                                                            place_name="CUDEM",
                                                                            dpi=1200)

if __name__ == "__main__":
    plot_regions()