# -*- coding: utf-8 -*-
"""gtif_to_netcdf.py -- Quick utility using the "gmt" commands to convert GeoTiff DEMs into NetCDF files."""

import os
import re
import subprocess
import argparse
import time
from osgeo import gdal

#####################################################
# Code snippet to import the base directory into
# PYTHONPATH to aid in importing from all the other
# modules in other subdirs.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
#####################################################

import utils.traverse_directory as traverse_directory
import utils.progress_bar as progress_bar
import etopo.coastline_mask

def check_for_gmt():
    """Check to make sure the command-line utility 'gmt' (general mapping tools) is installed on this machine."""

    try:
        p = subprocess.run(["gmt","--help"], capture_output=True)
        if p.returncode == 0:
            return True
        else:
            return False
    except FileNotFoundError:
        return False

def print_progress(i: int, N: int) -> None:
    progress_bar.ProgressBar(i, N, suffix="{0}/{1}".format(i, N))

def gtiff_to_netcdf(dir_or_file_name: str,
                    driver: str = 'gdal',
                    recurse_subdirs: bool = False,
                    gtif_regex: str = r'\.tif\Z',
                    omit_regex: str = r'\Z\A',
                    dest_subdir: str = '',
                    sids: bool = True,
                    vdatum_name: str = "EGM2008",
                    vdatum_epsg: int = 3855,
                    overwrite: bool = False,
                    verbose: bool = True) -> int:
    """Take all the geotiff files in a directory, convert them to netcdf. Return the number of files converted.

        Parameters:
            dir_or_file_name (str): Directory to look for geotiff files, or a geotiff file explicitly. Geotiff file names are presumed to end in .tif.
            driver (str) = 'gdal': The utility to use for the transformation. Options 'gdal' or 'gmt'.
            recurse_subdirs (bool): If True, recurse all sub-directories for files to convert. If False, stay in the 1st level local directory.
            gtif_regex (str) = r'\.tif\Z': A regular expression (see python 're' library) for identifying geotiff files and omitting other files. Defaults to r'\.tif\Z' (files that end in ".tif"). Useful to narrow down the search to only a subset of files in the directory. Ignored if only a single file is given.
            omit_regex (str) = r'\Z\A': A regular expression to omit certain files from processling. Defaults to an unmatchable regex target. Ignored if only a single file is given.
            dest_subdir (str) = '': A sub-directory in which to put each output file, with respect to the original filename. Default: put .nc files in the same directory.
            overwrite (bool) = False: Overwrite existing .nc files. Default: leave .nc files if they already exist. Already existing files are added to the output numfiles result.
            verbose (bool) = True: If False, run quietly. If true, output some results.
        Returns:
            numfiles (int): The number of geotiff files that were converted into netcdfs.
    """
    # does_gmt_exist = check_for_gmt()
    # if not does_gmt_exist:
    #     raise FileNotFoundError("Package 'gmt' (general mapping tools) required to run this script.")

    if not os.path.exists(dir_or_file_name):
        raise FileNotFoundError(dir_or_file_name)
    elif os.path.isdir(dir_or_file_name):
        fnames = traverse_directory.list_files(dir_or_file_name, regex_match=gtif_regex, include_base_directory=True, depth=-1 if recurse_subdirs else 0)
        if omit_regex != r'\Z\A':
            fnames = [fn for fn in fnames if (re.search(omit_regex, os.path.basename(fn)) is None)]
    else:
        fnames = [dir_or_file_name]

    driver_name = driver.strip().lower()

    num_written = 0
    for i, fname in enumerate(fnames):
        outfile_name = os.path.splitext(fname)[0] + ".nc"
        if dest_subdir:
            dest_dir = os.path.join(os.path.dirname(fname), dest_subdir)
            if not os.path.exists(dest_dir):
                os.mkdir(dest_dir)
            outfile_name = os.path.join(dest_dir, os.path.basename(outfile_name))

        if os.path.exists(outfile_name):
            if overwrite:
                os.remove(outfile_name)
            else:
                if verbose:
                    print_progress(i+1, len(fnames))
                num_written += 1
                continue

        if driver_name == "gmt":
            gmt_cmd = ["gmt", "grdconvert",
                       fname, "-G" + outfile_name]
            if verbose:
                print(" ".join(gmt_cmd))

            subprocess.run(gmt_cmd, capture_output=True)

            if sids:
                ncatted_sn_cmd = ["ncatted",
                                  "-a", "long_name,sid,o,c,sid",
                                  "-a", "standard_name,sid,a,c,source_id",
                                  "-h",
                                  outfile_basename]
            else:
                ncatted_sn_cmd = ["ncatted",
                                  "-a", "units,z,a,c,meters",
                                  "-a", "positive,z,a,c,up",
                                  "-a", "standard_name,z,a,c,height",
                                  "-a", "vert_crs_name,z,a,c," + vdatum_name,
                                  "-a", "vert_crs_epsg,z,a,c,EPSG:" + str(vdatum_epsg),
                                  "-h", # -h makes sure not to append the history, so we don't see this whole command in there.
                                  outfile_basename]

            if verbose:
                print(" ".join(ncatted_sn_cmd))

            subprocess.run(ncatted_sn_cmd, cwd=outfile_dir, capture_output=not verbose)

        elif driver_name == "gdal":
            # outfile_base, outfile_ext = os.path.splitext(outfile_name)

            ds = gdal.Open(fname, gdal.GA_ReadOnly)
            bbox, step = etopo.coastline_mask.get_bounding_box_and_step(ds)
            outfile_basename = os.path.basename(outfile_name)

            data_min, data_max, data_mean, data_std = ds.GetRasterBand(1).GetStatistics(0, 1)

            gdal_cmd = ["gdal_translate",
                        "-of", "NetCDF",
                        "-co", "COMPRESS=DEFLATE",
                        "-co", "ZLEVEL=5",
                        "-a_srs", "EPSG:4326", # NetCDFs don't support compound vertical+horizontal translations, so just use the horizontal.
                        "-a_ullr", str(bbox[0]), str(bbox[3]), str(bbox[2]), str(bbox[1]),
                        fname, outfile_name]
            if verbose:
                print(" ".join(gdal_cmd))
            subprocess.run(gdal_cmd, capture_output=True)

            time.sleep(0.01)
            assert os.path.exists(outfile_name)

            outfile_dir = os.path.dirname(outfile_name)
            outfile_basename = os.path.basename(outfile_name)

            # Then, we execute some nc commands to edit the file header.
            # 1. Rename the band to "z" rather than "Band1".
            ncrename_cmd = ["ncrename", outfile_basename,
                            "-v", ("Band1,sid" if sids else "Band1,z")]
            subprocess.run(ncrename_cmd, cwd=outfile_dir) #, capture_output=True)

            # # 2. Make the units 'meters'.
            # ncatted_m_cmd = ["ncatted", outfile_basename,
            #                 "-a", "units,z,a,c,meters"]
            # subprocess.run(ncatted_m_cmd, cwd=outfile_dir) #, capture_output=True)
            #
            # # 3. Make the positive direction 'up'.
            # ncatted_up_cmd = ["ncatted", outfile_basename,
            #                   "-a", "positive,z,a,c,up"]
            # subprocess.run(ncatted_up_cmd, cwd=outfile_dir) #, capture_output=True)
            #
            # # 4. Make the long_name 'z'.
            # ncatted_ln_cmd = ["ncatted", outfile_basename,
            #                   "-a", "long_name,z,o,c,z"]
            # subprocess.run(ncatted_ln_cmd, cwd=outfile_dir) #, capture_output=True)
            #
            # # 5. Make the standard_name 'height'.
            # ncatted_sn_cmd = ["ncatted", outfile_basename,
            #                   "-a", "standard_name,z,a,c,height"]
            # subprocess.run(ncatted_sn_cmd, cwd=outfile_dir) #, capture_output=True)

            # 6. Set a variable for the vertical reference name, the vertical reference epsg, and then copy (without the command history) over to the non-temp output file.
            if sids:
                ncatted_sn_cmd = ["ncatted",
                                  "-a", "long_name,sid,o,c,sid",
                                  "-a", "standard_name,sid,a,c,source_id",
                                  "-a", "actual_range,lat,c,d,{0},{1}".format(bbox[1], bbox[3]),
                                  "-a", "actual_range,lon,c,d,{0},{1}".format(bbox[0], bbox[2]),
                                  "-h",
                                  outfile_basename]
            else:
                ncatted_sn_cmd = ["ncatted",
                                  "-a", "units,z,a,c,meters",
                                  "-a", "positive,z,a,c,up",
                                  "-a", "long_name,z,o,c,z",
                                  "-a", "standard_name,z,a,c,height",
                                  "-a", "vert_crs_name,z,a,c," + vdatum_name,
                                  "-a", "vert_crs_epsg,z,a,c,EPSG:" + str(vdatum_epsg),
                                  "-a", "actual_range,lat,c,d,{0},{1}".format(bbox[1], bbox[3]),
                                  "-a", "actual_range,lon,c,d,{0},{1}".format(bbox[0], bbox[2]),
                                  "-a", "actual_range,z,c,d,{0},{1}".format(data_min, data_max),
                                  "-a", "history,global,d,,",
                                  "-h", # -h makes sure not to append the history, so we don't see this whole command in there.
                                  outfile_basename]
            if verbose:
                print(" ".join(ncatted_sn_cmd))

            subprocess.run(ncatted_sn_cmd, cwd=outfile_dir) #, capture_output=True)

            # Check to make sure the output file exists.
            assert os.path.exists(outfile_name)

            # # 7. Set a variable for the vertical reference epsg.
            # ncatted_sn_cmd = ["ncatted", outfile_base,
            #                   ]
            # subprocess.run(ncatted_sn_cmd, cwd=outfile_dir) #, capture_output=True)

        else:
            raise ValueError("Unknown Driver '{0}' used.".format(driver))

        if os.path.exists(outfile_name):
            num_written += 1
        elif verbose:
            print("ERROR: {0} not written.".format(outfile_name))

        if verbose:
            print_progress(i+1, len(fnames))

    if verbose:
        print("{0} of {1} NetCDF files written.".format(num_written, len(fnames)))

    return num_written


def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Convert a directory of GeoTiff (.tif) files to NetCDF (.nc) using 'gmt grdconvert'.")
    parser.add_argument("dir_or_file", help="Geotiff file, or a directory name to look for GeoTiff files.")
    parser.add_argument("-driver", default="gdal", help="Tool to use. Can be 'gdal' or 'gmt'. Uses 'gdal_translate' and 'gmt grdconvert', respectively. Default: gdal")
    parser.add_argument("-dest_subdir", default="netcdf", help="A sub-directory (relative to each file's local directory) in which to put the destination .nc file. Default 'netcdf'.")
    parser.add_argument("-gtif_regex", default=r"\.tif\Z", help="A regular expression (see python 're' library) for which to search for geotiff files. Defaults to any file ending in '.tif'.")
    parser.add_argument("-omit_regex", default="_w\.tif\Z", help="A regular expression to omit certain files from processing. Defaults to r'_w\.tif\Z', while filters out weights files ending in _w.tif.")
    parser.add_argument("-vdatum_str", default="EGM2008", help="The string of the vertical datum of the dataset. Default: EMG2008")
    parser.add_argument("-vdatum_epsg", default=3855, type=int, help="The integer EPSG code of the vertical datum of the dataset. Default: 3855")
    # parser.add_argument("--omit_weights", default=False, action="store_true", help="Omit weights ('_w.tif') files.")
    parser.add_argument("--sid", default=False, action="store_true", help="Treat as a source_id (sid) file rather than DEM. This changes the header information accordingly.")
    parser.add_argument("--overwrite", "-o", default=False, action="store_true", help="Overwrite existing files.")
    parser.add_argument("--recurse", "-r", default=False, action="store_true", help="Recurse into sub-directories.")
    parser.add_argument("--quiet", "-q", default=False, action="store_true", help="Execute quietly.")

    return parser.parse_args()

if __name__ == "__main__":
    args = define_and_parse_args()

    if not os.path.exists(args.dir_or_file):
        print(args.dir_or_file, "does not exist on local machine.")
        import sys
        sys.exit(0)

    gtiff_to_netcdf(args.dir_or_file,
                    recurse_subdirs=args.recurse,
                    driver=args.driver,
                    dest_subdir = args.dest_subdir,
                    gtif_regex=args.gtif_regex,
                    omit_regex=args.omit_regex,
                    sids=args.sid,
                    vdatum_name=args.vdatum_str,
                    vdatum_epsg=args.vdatum_epsg,
                    overwrite=args.overwrite,
                    verbose=not args.quiet)
