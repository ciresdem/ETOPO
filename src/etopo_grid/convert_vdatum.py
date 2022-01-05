# -*- coding: utf-8 -*-

""" convert_vdatum.py -- Code for converting vertical datums of GeoTiff raster grids.
Uses gdalwarp as the underlying tool.

Author: Mike MacFerrin, University of Colorado
Created: August 18, 2021
"""

# gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4326 -tr 0.0002777777777777778 0.0002777777777777778 -te -79.0000000000000000 26.0000000000000000 -78.0000000000000000 27.0000000000000000 -r cubic -of GTiff -overwrite /home/mmacferrin/Research/DATA/DEMs/geoids/egm96-15.tif /home/mmacferrin/Research/DATA/DEMs/AW3D30/data/tiles/N025W080_N030W075/ALPSMLC30_N026W079_EGM96.tif
# Creating output file that is 3600P x 3600L.
# Processing /home/mmacferrin/Research/DATA/DEMs/geoids/egm96-15.tif [1/1] : 0Using internal nodata values (e.g. -32767) for image /home/mmacferrin/Research/DATA/DEMs/geoids/egm96-15.tif.
# Copying nodata values from source /home/mmacferrin/Research/DATA/DEMs/geoids/egm96-15.tif to destination /home/mmacferrin/Research/DATA/DEMs/AW3D30/data/tiles/N025W080_N030W075/ALPSMLC30_N026W079_EGM96.tif.

import argparse
import os
import shutil
import sys
from osgeo import gdal

import config
my_config = config.config()

# Common global vertical datums, and their EPSG codes. Look in ../ini/config.ini for the list of grid files for each geoid datum.
SUPPORTED_VDATUMS = ["wgs84", "ellipsoid", 4326, "egm2008", 3855, "egm96", 5773, "egm84", 5798]
VDATUM_FILE_DICT = {"wgs84"     : None,
                    "ellipsoid" : None,
                     4326       : None,
                    "egm2008"   : my_config.egm2008_grid_file,
                     3855       : my_config.egm2008_grid_file,
                    "egm96"     : my_config.egm96_grid_file,
                     5773       : my_config.egm96_grid_file,
                    "egm84"     : my_config.egm84_grid_file,
                     5798       : my_config.egm84_grid_file,
                    }

def vdatum_name_lookup(vd_gridfile):
    # Just a quick script for lookup up the name of a vdatum given its reference grid file.
    # Useful for file naming, as "egm2008" and 3855 would be the exact same grid.
    if vd_gridfile is None:
        return "wgs84"
    else:
        return {my_config.egm2008_grid_file: "egm2008",
                my_config.egm96_grid_file: "egm96",
                my_config.egm84_grid_file: "egm84"}[vd_gridfile]


# def get_epsg_from_srs_string(srs_string):
#     """From the GDAL SRS string, retreive the primary EPSG code for this dataset.

#     It resides in the final 'AUTHORITY["EPSG","----"]] tag with "----" as the 4-digit number. Return as an integer.'
#     """
#     # Strip off extraneous whitespace
#     srs_s = srs_string.strip().upper()
#     epsg_str = srs_s[srs_s.rfind('AUTHORITY["EPSG","')+len('AUTHORITY["EPSG","'):srs_s.rfind('"]]')]
#     return int(epsg_str)

def get_dataset_metadata(dem_name, band_num=1, return_ds=True):
    """Retreive metadata from a GDAL GeoTiff file.

    Return: srs,
            resolution (xstep, ystep)
            extent (xmin, ymin, xmax, ymax),
            size (xsize, ysize),
            nodata_value (if present, None if not),
            dataset (if return_ds is True)
    """
    ds = gdal.Open(dem_name, gdal.GA_ReadOnly)
    # proj = ds.GetProjection()
    # print("proj:", proj)
    srs = ds.GetSpatialRef()
    # print("epsg:", epsg)
    xmin, xstep, _, ymin, _, ystep = ds.GetGeoTransform()
    b1 = ds.GetRasterBand(band_num)
    ndv = b1.GetNoDataValue()
    xsize = b1.XSize
    ysize = b1.YSize

    xmax = xmin + xstep*xsize
    ymax = ymin + ystep*ysize

    # Since the xstep or ystep can be negative, we need to make sure these are actuall min/max
    xmin, xmax = min(xmin,xmax), max(xmin, xmax)
    ymin, ymax = min(ymin,ymax), max(ymin, ymax)

    if return_ds:
        return (srs,
                (xstep, ystep),
                (xmin, ymin, xmax, ymax),
                (xsize, ysize),
                ndv,
                ds)
    else:
        return (srs,
                (xstep, ystep),
                (xmin, ymin, xmax, ymax),
                (xsize, ysize),
                ndv)


def gdal_resample_alg_lookup(alg_name):
    """Given a command-line interpolation scheme, return the GDAL API code for it."""
    alg_name = alg_name.lower().strip()
    return {"near"        : gdal.GRA_NearestNeighbour,
            "bilinear"    : gdal.GRA_Bilinear,
            "cubic"       : gdal.GRA_Cubic,
            "cubicspline" : gdal.GRA_CubicSpline,
            "lanczos"     : gdal.GRA_Lanczos,
            "average"     : gdal.GRA_Average,
            "mode"        : gdal.GRA_Mode,
            "max"         : gdal.GRA_Max,
            "min"         : gdal.GRA_Min,
            "med"         : gdal.GRA_Med,
            "q1"          : gdal.GRA_Q1,
            "q3"          : gdal.GRA_Q3,
            } [alg_name]

def convert_vdatum(input_dem,
                   input_vdatum_gridfile,
                   output_dem,
                   output_vdatum_gridfile,
                   tempdir,
                   band_num=1,
                   ndv=None,
                   keep_grids=False,
                   overwrite_grids=False,
                   interpolation_method="cubic",
                   verbose=True):
    # 1: Create the name of each projected vdatum file.
    dem1_fname = os.path.split(input_dem)[1]
    vd1_name = vdatum_name_lookup(input_vdatum_gridfile)
    vd2_name = vdatum_name_lookup(output_vdatum_gridfile)

    # If both the input and output datums are the same, just copy the file unchanged.
    if vd1_name == vd2_name:
        shutil.copyfile(input_dem, output_dem)
        return output_dem

    dem_srs, \
    dem_resolution, \
    dem_extent, \
    dem_size, \
    dem_ndv, \
    dem_ds       = get_dataset_metadata(input_dem, band_num=band_num, return_ds=True)

    # If we didn't explicitly set the NDV, get it from the DEM.
    if ndv is None:
        ndv = dem_ndv

    # Options for the GDAL Warp operations.
    vdatum_regrid_opts = gdal.WarpOptions(\
              dstSRS = dem_srs,
              xRes = dem_resolution[0], # target resolution (xstep, ystep)
              yRes = dem_resolution[1], # target resolution (xstep, ystep)
              outputBounds = dem_extent, # target extent (xmin, ymin, xmax, ymax)
              srcNodata = ndv, # source nodatavalue
              dstNodata = ndv, # dest nodatavalue
              resampleAlg = gdal_resample_alg_lookup(interpolation_method), # resampling method
              )

    if vd1_name != "wgs84":
        # Create the temporary vdatum grid file.
        vdatum_1_regridded_file_name = os.path.join(tempdir, vd1_name + "_" + interpolation_method + "_" + dem1_fname)
        # Delete the old grid if we've chosen to overwrite it.
        if overwrite_grids and os.path.exists(vdatum_1_regridded_file_name):
            os.remove(vdatum_1_regridded_file_name)
        # Do the interpolation.
        gdal.Warp(vdatum_1_regridded_file_name, input_vdatum_gridfile, options=vdatum_regrid_opts)
    else:
        vdatum_1_regridded_file_name = None

    if vd2_name != "wgs84":
        # Create the temporary vdatum grid file.
        vdatum_2_regridded_file_name = os.path.join(tempdir, vd2_name + "_" + interpolation_method + "_" + dem1_fname)
        # Delete the old grid if we've chosen to overwrite it.
        if overwrite_grids and os.path.exists(vdatum_2_regridded_file_name):
            os.remove(vdatum_2_regridded_file_name)
        # Do the interpolation.
        gdal.Warp(vdatum_2_regridded_file_name, output_vdatum_gridfile, options=vdatum_regrid_opts)
    else:
        vdatum_2_regridded_file_name = None


    # Read the original DEM array.
    result_array = dem_ds.GetRasterBand(band_num).ReadAsArray()

    # Create a mask of all nodata locations (for future masking)
    if ndv != None:
        ndv_mask = (result_array == ndv)

    # Add the first datum values (if they exist) to convert to WGS84
    if vdatum_1_regridded_file_name != None:
        vd1_ds = gdal.Open(vdatum_1_regridded_file_name, gdal.GA_ReadOnly)
        vd1_arr = vd1_ds.GetRasterBand(1).ReadAsArray()

        # Add the first vdatum to bring it into WGS84
        result_array = result_array + vd1_arr
        vd1_ds = None

        # Clean up the dataset
        if not keep_grids:
            os.remove(vdatum_1_regridded_file_name)

    # Subtract the second datum values (if they exist) to convert to the new datum
    if vdatum_2_regridded_file_name != None:
        vd2_ds = gdal.Open(vdatum_2_regridded_file_name, gdal.GA_ReadOnly)
        vd2_arr = vd2_ds.GetRasterBand(1).ReadAsArray()

        # Substract the second vdatum to convert it to the new geoid.
        result_array = result_array - vd2_arr

        vd2_ds = None
        # Clean up the dataset
        if not keep_grids:
            os.remove(vdatum_2_regridded_file_name)

    # Mask out NDV values (since we ignored them in the band math)
    if ndv != None:
        result_array[ndv_mask] = ndv

    # Create a copy of the DEM raster to write the output (this is the easiest way of doing this).
    out_ds = dem_ds.GetDriver().CreateCopy(output_dem, dem_ds, strict=0)
    out_band = out_ds.GetRasterBand(band_num)

    out_band.WriteArray(result_array)
    if ndv != None:
        out_band.SetNoDataValue(ndv)

    # This will get GDAL to calculate the band statistics and include it in the file.
    out_band.SetStatistics(*out_band.GetStatistics(0,1))

    # Close the datasets, write the file.
    dem_ds = None
    out_band = None
    out_ds = None

    # 2: Check to see if they already exist. If overwrite_grids, delete and overwrite them. Else, keep them and move along.
    # 3: Create newly interpolated geoid grids using gdal_warp. Use the interpolation_method given.
    # 4: Get the EPSG srs from the source file.
    # 5: Do band math to convert the original image to wgs84 (if needed) and then to new geoid (if needed)
    # 6: If not keep_grids, delete the temporary grid files.
    # Exit successfully.
    if verbose:
        print(output_dem, "written.")
    return output_dem

def get_vdatum_name(input_name):
    """Just a simple tool for identifying and translating vdatum names and references."""
    # If it's an integer string, just convert to an integer.
    try:
        mod_input_name = int(input_name)
    except ValueError:
        mod_input_name = input_name

    # If it's a string, make it lowercase and strip off all whitespace anywhere in the string.
    if type(mod_input_name) == str:
        mod_input_name = "".join(input_name.strip().lower().split())

    # If we don't recognize it, toss an error IF "check_name_error" parameter is set.
    if mod_input_name not in SUPPORTED_VDATUMS:
        raise ValueError(f"Unknown vertical datum '{input_name}'. Use the --help flag to see a list of supported vdatums.")

    return mod_input_name

def read_and_parse_arguments():
    # Collect and process command-line arguments.
    parser = argparse.ArgumentParser(description='Change the vertical datum on a DEM raster, using globally gridded datums.')
    parser.add_argument('input_dem', type=str,
                        help='The input DEM.')
    parser.add_argument('-output_dem','-o', type=str, default="",
                        help='The output DEM. Default: Same path as input_dem, with "_out" added to the filename.')
    parser.add_argument('-ndv', default="",
                        help='Nodata value. Default: None. If None, reads it from the input_dem metadata.')
    parser.add_argument('-input_vdatum','-ivd', type=str, default="wgs84",
                        help="Input DEM vertical datum. (Default: 'wgs84')" + \
                        " Currently supported datum arguments, not case-sensitive: ({})".format(",".join([(("'" + vd + "'") if type(vd) == str else str(vd)) for vd in SUPPORTED_VDATUMS]))
                        )
    parser.add_argument('-output_vdatum','-ovd', type=str, default="wgs84",
                        help="Output DEM vertical datum. (Default: 'wgs84')" + \
                        " Supports same datum list as input_vdatum")
    parser.add_argument('-tempdir', type=str, default="",
                        help="A scratch directory to write temporary resampled grid files. Useful if user would like to save temp files elsewhere. Defaults to the destination directory.")
    parser.add_argument('-interp_method', type=str, default="cubic",
                        help="Interpolation method passed to gdal_warp. Default 'cubic'.")
    parser.add_argument('-band_num', type=int, default=1,
                        help="The band number (1-indexed). (Default: 1)")
    parser.add_argument('--keep_grids', action='store_true', default=False,
                        help='Keep the temporary regridded geoid/ellipsoid surface file[s]. Sometimes useful for reuse (Default: erase temporary files)')
    parser.add_argument('--overwrite_grids', action='store_true', default=False,
                        help='If a vdatum grid for this file already exists, create it again. Default: Just use it as is. NOTE: If keep_grids is False, the vdatum grid will be deleted at the end.')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Suppress output messaging, including error messages (just fail quietly without errors, return status 1).')

    return parser.parse_args()

# Main tasks.
if __name__ == "__main__":
    # Retreive command-line arguments.
    args = read_and_parse_arguments()

    try:
        # Collect and check our input and output vdata.
        ivd = get_vdatum_name(args.input_vdatum)
        ovd = get_vdatum_name(args.output_vdatum)
        ivd_gridfile = VDATUM_FILE_DICT[ivd]
        ovd_gridfile = VDATUM_FILE_DICT[ovd]

        # Output DEM defaults to the input_dem with "_out" on the filename.
        if args.output_dem == "":
            base, ext = os.path.splitext(args.input_dem)
            output_dem = base + "_out" + ext
        else:
            output_dem = args.output_dem

        # The temporary directory for saving temp files. Default to the output directory.
        if args.tempdir == "":
            tempdir = os.path.split(output_dem)[0]
        else:
            tempdir = args.tempdir

        # Get the NoDataValue
        if args.ndv == "":
            NDV = None
        else:
            try:
                NDV = int(args.ndv)
            except ValueError:
                NDV = float(args.ndv)

        # Do the work.
        convert_vdatum(args.input_dem,
                       ivd_gridfile,
                       output_dem,
                       ovd_gridfile,
                       tempdir,
                       band_num=args.band_num,
                       ndv=NDV,
                       keep_grids=args.keep_grids,
                       overwrite_grids=args.overwrite_grids,
                       interpolation_method=args.interp_method,
                       verbose=not args.quiet)

    except Exception as e:
        # Handle exceptions quietly (fail quietly) if in quiet mode
        if args.quiet:
            sys.exit(1)
        # Else just re-raise it like normal, let the user or the calling-program read it.
        else:
            raise e
