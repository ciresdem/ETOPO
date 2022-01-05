# -*- coding: utf-8 -*-

"""validate_dem_collection.py
Code for validating and summarizing an entire list or directory of DEMs.
"""

import os
import pandas
import numpy
from osgeo import gdal

import validate_dem
import icepyx_download
import retrieve_land_photons

def read_or_create_photon_h5(dem_list, photon_h5, output_dir=None, verbose=True):
    """If the photon_h5 file exists, read it. Else, create one from the large bounding box of the DEMs."""
    # If the photon hdf5 file already exists, read it and return the dataframe.
    if os.path.exists(photon_h5):
        if verbose:
            print("Reading", photon_h5 + "...", end="")
        photon_df = pandas.read_hdf(photon_h5, mode='r')
        if verbose:
            print("Done.")
    # Otherwise, generate a large bounding box, and download icesat-2 data into the photon bounding box.
    else:
        # 1. Get the "master" bounding box of all the DEMs in this set.
        # NOTE OF WARNING: This will not work if DEM's straddle the longitudinal dateline (+/- 180* longitude).
        # We are presuming they don't. This will break if that is not true.
        # Start with nonsense min/max values.
        xmin_total = numpy.inf
        xmax_total = -numpy.inf
        ymin_total = numpy.inf
        ymax_total = -numpy.inf

        for dem_name in dem_list:
            if not os.path.exists(dem_name):
                print("File", dem_name, "does not appear to exist at the location specified. Skipping.")
                continue

            dset = gdal.Open(dem_name, gdal.GA_ReadOnly)
            gtf = dset.GetGeoTransform()
            xsize, ysize = dset.RasterXSize, dset.RasterYSize
            xleft, xres, xskew, ytop, yskew, yres = gtf
            xright = xleft + (xsize*xres)
            ybottom = ytop + (ysize*yres)

            xmin_total = min(xleft, xright, xmin_total)
            xmax_total = max(xleft, xright, xmax_total)
            ymin_total = min(ytop, ybottom, ymin_total)
            ymax_total = max(ytop, ybottom, ymax_total)

    return photon_df

def validate_list_of_dems(dem_list,
                          photon_h5,
                          output_dir=None,
                          input_vdatum="wgs84",
                          output_vdatum="wgs84",
                          skip_files_if_complete=True,
                          date_range=["2020-01-01","2020-12-31"],
                          verbose=True):
    """Take a list of DEMs, presumably in a single area, and output validation files for those DEMs.

    DEMs should encompass a contiguous area so as to use the same set of ICESat-2 granules for
    validation."""
    # If a common photon dataframe already exists, open and use it.

    photon_df = read_or_create_photon_h5(dem_list, photon_h5, output_dir=output_dir, verbose=verbose)

    for i, dem_path in enumerate(dem_list):
        if verbose:
            print("\n=======", dem_path, ", (" + str(i+1), "of", str(len(dem_list)) + ")", "=======")

        if output_dir != None:
            _, fname = os.path.split(dem_path)
            results_h5_file = os.path.join(output_dir, fname)


        results_h5_file = os.path.splitext(dem_path)[0] + ".h5"
        if os.path.exists(results_h5_file):
            print("Results complete, skipping.")
            continue
