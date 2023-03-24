# Quick script to compute the local slope of a given DEM raster (or directory of rasters) using gdaldem.

import argparse
import re
import os
import subprocess

import traverse_directory

def calculate_slope(dir_or_fname: str,
                    filter_regex: str = r"\.tif\Z",
                    skip_if_empty_exists: bool = False,
                    recursive: bool = False,
                    overwrite: bool = False,
                    verbose: bool = True) -> None:
    """Given a DEM file, calculate the slope. Save to the identical file with "_slope" attached to the name.

    Can be a directory of file as well.
    If a directory and "filter_regex" is set, it will only use files that match the regular expression given.
    Defaults to all .tif files.
    """
    if not os.path.exists(dir_or_fname):
        raise FileNotFoundError(dir_or_fname + " does not exist locally.")
    elif os.path.isdir(dir_or_fname):
        if recursive:
            dem_list = traverse_directory.list_files(dir_or_fname, regex_match = filter_regex)
        else:
            dem_list = sorted([os.path.join(dir_or_fname, fn) for fn in os.listdir(dir_or_fname) if re.search(filter_regex, fn) is not None])
    else:
        dem_list = [dir_or_fname]

    if skip_if_empty_exists:
        # If a dem had no results, the file generated is [DEM_basename]_results_EMPTY.txt
        # Look for those, and omit the dem if the directory contains such a file.
        updated_dem_list = []
        for dem in dem_list:
            empty_fname = os.path.splitext(dem)[0] + "_results_EMPTY.txt"
            if not os.path.exists(empty_fname):
                updated_dem_list.append(dem)

        if verbose and len(updated_dem_list) > 0:
            print(len(dem_list) - len(updated_dem_list), "DEM files omitted with no results.", len(updated_dem_list), "files remaining.")

        dem_list = updated_dem_list

    for i, dem in enumerate(dem_list):
        dem_base, dem_ext = os.path.splitext(dem)
        slope_fname = dem_base + "_slope" + dem_ext
        if os.path.exists(slope_fname):
            if overwrite:
                os.remove(slope_fname)
            else:
                print("{0}/{1} {2} already exists.".format(i+1, len(dem_list), os.path.basename(slope_fname)))
                continue

        gdal_command = ["gdaldem", "slope",
                        dem, slope_fname,
                        "-compute_edges",
                        "-scale", str(111120), # For lat/lon DEMs, use 111,120 for the vertial-to-horizontal scale ratio.
                        "-co", "COMPRESS=DEFLATE", # Compress the files using Deflate.
                        "-co", "PREDICTOR=2",
                        "-co", "TILED=YES"]

        subprocess.run(gdal_command, capture_output=True)

        print("{0}/{1} {2} {3}written.".format(i+1, len(dem_list), os.path.basename(slope_fname), "" if os.path.exists(slope_fname) else "NOT "))

    return


def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Caculate slope for a DEM file or directory of files. Output as a compressed GeoTiff.")
    parser.add_argument("dir_or_filename", type=str, help="A DEM file name or directory containing DEM files.")
    parser.add_argument("-regex", type=str, default=r'\.tif\Z', help="Regular expression to search for files. Default: r'\.tif\Z', finding all files ending in .tif. Only used if dir_or_filename is a directory.")
    parser.add_argument("--skip_if_EMPTY_exists", default=False, action="store_true", help="In DEM ICESat-2 validations, we produce an '_EMPTY.txt' file for each DEM that had no results. If this is set, skip any DEM for which an adjacent '_EMPTY.txt' file exists in the same directory.")
    parser.add_argument("--overwrite", default=False, action="store_true", help="Overwrite existing output files. Default: Just skip processing if the output file exists already.")
    parser.add_argument("--recursive", default=False, action="store_true", help="Search recursively for DEM files in the directory. (Default: find only files in the local directory.")
    parser.add_argument("--quiet", default=False, action="store_true", help="Run quietly.")

    return parser.parse_args()

if __name__ == "__main__":
    args = define_and_parse_args()

    calculate_slope(args.dir_or_filename,
                    filter_regex = args.regex,
                    skip_if_empty_exists = args.skip_if_EMPTY_exists,
                    overwrite = args.overwrite,
                    recursive = args.recursive,
                    verbose = not args.quiet)