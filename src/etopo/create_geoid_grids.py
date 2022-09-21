# -*- coding: utf-8 -*-
"""map_finished_tiles_to_release_directory.py - creating symlinks to the final output locations in /data/ETOPO_2022_release."""

import os
import subprocess
from osgeo import gdal

#####################################################
# Code snippet to import the base directory into
# PYTHONPATH to aid in importing from all the other
# modules in other subdirs.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
#####################################################
import datasets.dataset_geopackage
import utils.configfile
etopo_config = utils.configfile.config()

def make_temp_zero_grid(infile, outfile, overwrite=False):
    """Create a copy of the input file to the output file, but give it all-zero values."""
    if os.path.exists(outfile):
        if overwrite:
            os.remove(outfile)
        else:
            return

    src = gdal.Open(infile, gdal.GA_ReadOnly)
    dst = gdal.GetDriverByName("GTiff").CreateCopy(outfile, src, strict=0)
    dst = None
    dst = gdal.Open(outfile, gdal.GA_Update)
    band = dst.GetRasterBand(1)
    array = band.ReadAsArray()
    array[:,:] = 0
    band.WriteArray(array)
    band = None
    dst = None
    return

def create_geoid_height_grids(resolution_s = (1,15,30,60), crm_only_if_1s=True, overwrite=False, verbose=True, silent_subprocs=True):
    """Create geoid-height grids for each ETOPO grid. This makes converting between EGM2008 and ITRF ellipsoid elevations easy."""

    if type(resolution_s) in (int, float):
        resolution_s = [int(resolution_s)]


    for res_s in resolution_s:
        gdf = datasets.dataset_geopackage.ETOPO_Geopackage(res_s).get_gdf(resolution_s = res_s,
                                                                          crm_only_if_1s=crm_only_if_1s,
                                                                          verbose=verbose)

        dest_dir = os.path.join(etopo_config._abspath(etopo_config.etopo_geoid_directory), "{0}s".format(res_s))
        temp_dir = os.path.join(dest_dir, "temp")
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)

        print("==", str(res_s) + "s", "resolution,", len(gdf), "files. ==")

        for i,row in gdf.iterrows():
            tempfile = os.path.join(temp_dir, os.path.splitext(os.path.basename(row.filename))[0] + "_ZERO_TEMP.tif")
            make_temp_zero_grid(row.filename, tempfile, overwrite=overwrite)

            destfile = os.path.join(dest_dir, os.path.splitext(os.path.basename(row.filename))[0] + "_geoid.tif")

            if os.path.exists(destfile):
                if overwrite:
                    os.remove(destfile)
                else:
                    print("{0}/{1} {2} already exists.".format(i+1, len(gdf), os.path.basename(destfile)))
                    continue

            convert_cmd = ["vertical_datum_convert.py",
                           "-i", "3855",\
                           "-o", "7912",
                           "-D", etopo_config._abspath(etopo_config.etopo_cudem_cache_directory),
                           "-k",
                           tempfile, destfile]

            if silent_subprocs:
                subprocess.run(convert_cmd, capture_output=True)
            else:
                print(" ".join(convert_cmd))
                subprocess.run(convert_cmd)

            print("{0}/{1} {2} ".format(i+1, len(gdf), os.path.basename(destfile)), end="")
            if os.path.exists(destfile):
                print("written.")
            else:
                print("NOT written.")

        # Get rid of the temp dir and all its contents.
        rm_cmd = ["rm", "-rf", temp_dir]
        subprocess.run(rm_cmd, capture_output=True)

if __name__ == "__main__":
    create_geoid_height_grids(resolution_s=(15,30,60,1), silent_subprocs=False)