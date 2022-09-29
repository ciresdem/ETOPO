# -*- coding: utf-8 -*-

"""trim_ndv_mask.py -- Trim edge-artifact grid-cells from a resampled dataset. Use the original dataset as a mask."""

import os
import argparse
from osgeo import gdal
import numpy
import pyproj

def trim_gtif(gtif_orig : str,
              gtif_to_mask : str,
              output_gtif : str,
              src_ndv : float = None,
              out_ndv : float = None,
              overwrite : bool = False,
              gt_tolerance : float = 1e-12,
              verbose: bool = True):
    """Trim a resampled geotiff of edge artifacts based on the original dataset. Set edge artifacts to NDV.

    gtif_orig -- Original geotiff with the correct mask that you want to use to mask out the other one.
    gtif_to_mask -- Dataset that you would like to modify to mask out values that aren't in the valid-data mask of gtif_orig.
    output_gtif -- Where to write the output file."""
    if os.path.exists(output_gtif):
        if overwrite:
            os.remove(output_gtif)
        else:
            if verbose:
                print(os.path.basename(output_gtif), "already exists.")
            return

    # Input the original dataset.
    ds_orig = gdal.Open(gtif_orig, gdal.GA_ReadOnly)
    band_orig = ds_orig.GetRasterBand(1)
    proj_orig = ds_orig.GetProjection()
    if src_ndv is not None:
        ndv_orig = src_ndv
    else:
        ndv_orig = band_orig.GetNoDataValue()

    # If there is no source dataset NDV, there is nothing to mask, so just exit.
    if ndv_orig is None:
        print(os.path.basename(gtif_orig), "has no NoDataValue. Nothing to mask. Use the -src_ndv flag to use a NDV in the dataset, or skip this one.")
        ds_orig = None
        return

    array_orig = band_orig.ReadAsArray()
    gt_orig = ds_orig.GetGeoTransform()

    # Input the dataset to mask.
    ds_to_mask = gdal.Open(gtif_to_mask, gdal.GA_ReadOnly)
    band_to_mask = ds_to_mask.GetRasterBand(1)
    proj_to_mask = ds_to_mask.GetProjection()
    ndv_to_mask = band_to_mask.GetNoDataValue()
    if ndv_to_mask is None:
        ndv_to_mask = ndv_orig
    array_to_mask = band_to_mask.ReadAsArray()
    gt_to_mask = ds_to_mask.GetGeoTransform()

    # Check to make sure these datasets match, both in projection and geotransform.
    if proj_orig != proj_to_mask:
        # If the text of the projections don't match, look at the EPSG values. If those are the same, we're still good.
        epsg_orig = pyproj.crs.CRS.from_user_input(proj_orig).to_epsg()
        epsg_to_mask = pyproj.crs.CRS.from_user_input(proj_to_mask).to_epsg()
        if epsg_orig != epsg_to_mask:
            print("Projections do not seem to match:")
            print(os.path.basename(gtif_orig), ":", proj_orig)
            print(os.path.basename(gtif_to_mask), ":", proj_to_mask)
            print("Exiting...")
            return

    # Test the geotransforms, but pass as long as they're within a tolerance of 10e-10
    if not numpy.all(numpy.abs(numpy.array(gt_orig) - numpy.array(gt_to_mask)) < gt_tolerance):
        print("Dataset geotransforms do not seem to match:")
        print(os.path.basename(gtif_orig), ":", gt_orig)
        print(os.path.basename(gtif_to_mask), ":", gt_to_mask)
        print("Exiting...")
        return

    # Create the output dataset, and then read it in with gdal.GA_Update set to update the data.
    driver = gdal.GetDriverByName(gdal.Info(ds_to_mask, options="-json")['driverShortName'])
    # print(driver)
    ds_out = driver.CreateCopy(output_gtif, ds_to_mask)
    ds_out = None
    # print(os.path.basename(output_gtif), "created (not yet masked).")

    ds_out = gdal.Open(output_gtif, gdal.GA_Update)
    band_out = ds_out.GetRasterBand(1)
    ndv_out = band_out.GetNoDataValue()
    if ndv_out is None:
        if out_ndv:
            ndv_out = out_ndv
        else:
            ndv_out = ndv_to_mask
        band_out.SetNoDataValue(ndv_out)

    # array_out = band_out.ReadAsArray()
    # We just created this dataset as a copy. It should be exactly the same.
    # assert numpy.all(array_out == array_to_mask)
    # Get all the NDVs from the original dataset.
    ndv_mask = (array_orig == ndv_orig)
    # Set those as NDVs in the output dataset.
    array_to_mask[ndv_mask] = ndv_out
    band_out.WriteArray(array_to_mask)

    band_out = None
    ds_out = None
    if verbose:
        print(os.path.basename(output_gtif), "written.")

    return

def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Mask out edge artifacts from a resampled dataset, such as FABDEM or resampled-Copernicus where edge pixels were introduced.")
    parser.add_argument("GTIF_ORIG", help="The original dataset before resampling. The NDV mask from this dataset is used to clip the other.")
    parser.add_argument("GTIF_TO_MASK", help="The resampled dataset to mask out using the NDV-mask from the source data. Any extraneous edge pixels are set to NDV in this dataset. Both datasets must be set to the same horizontal projection and geotransform grids. An error will be tossed if they are not.")
    parser.add_argument("-output", "-O", default="", help="Output dataset to write. Default to the input dataset with '_trimmed' included in the filename.")
    parser.add_argument("-src_ndv", default=None, help="The nodata value of the source dataset. Useful if this is not set in the original source. Overrides any source dataset.")
    parser.add_argument("-out_ndv", default=None, help="Set the output NDV to a particular value. Default, same NDV as either the GTIF_TO_MASK, or -src_ndv (if set).")
    parser.add_argument("--overwrite", default=False, action="store_true", help="Overwrite output data. If not set and output data exists, just do nothing and move along.")
    parser.add_argument("--quiet", default=False, action="store_true", help="Run in quiet mode.")
    return parser.parse_args()

if __name__ == "__main__":
    args = define_and_parse_args()

    if args.output == "":
        base, ext = os.path.splitext(args.GTIF_TO_MASK)
        args.output = base + "_trimmed" + ext

    try:
        args.src_ndv = float(args.src_ndv)
    except (ValueError, TypeError):
        args.src_ndv = None

    trim_gtif(args.GTIF_ORIG,
              args.GTIF_TO_MASK,
              args.output,
              src_ndv=args.src_ndv,
              out_ndv=args.out_ndv,
              overwrite=args.overwrite,
              verbose=not args.quiet)