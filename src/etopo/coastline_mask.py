# -*- coding: utf-8 -*-

"""coastline_mask.py -- Wrapper code for using the CUDEM "coastline" module for
creating resampled coastline masks for DEMs.

TODO: Look into covering interior water bodies with this, which we don't currently do
except over the US. There are some LandSat & MODIS-sourced datasets we could use for this, I think.
"""

try:
    # We don't actually use the cudem modules here, but we make command-line
    # calls to "waffles", so check here to make sure cudem is installed on this
    # machine. If we can import the waffles module, we can use it from the command-line.
    from cudem import waffles
except:
    raise ModuleNotFoundError("Module 'cudem/waffles.py' required. Update paths, or refer to https://github.com/ciresdem/cudem for installation instructions.")

import os
from osgeo import gdal
import rich.console
import subprocess
import argparse
import pyproj

####################################3
# Include the base /src/ directory of thie project, to add all the other modules.
import import_parent_dir; import_parent_dir.import_src_dir_via_pythonpath()
####################################3
# import utils.progress_bar as progress_bar
# Use config file to get the encrypted credentials.
import utils.configfile as configfile
etopo_config = configfile.config()


def is_this_run_in_ipython():
    """Tell whether we're running in an IPython console or not. Useful for rich.print()."""
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

def get_bounding_box_and_step(gdal_dataset, invert_for_waffles=False):
    """Get the [xmin, ymin, xmax, ymax] from the gdal geotransform, as well as the [xstep, ystep].
    If 'invert_for_waffles', bbox returned as [xmin, xmax, ymin, ymax]."""
    geotransform = gdal_dataset.GetGeoTransform()
    x_size, y_size = gdal_dataset.RasterXSize, gdal_dataset.RasterYSize

    xmin, xstep, _, ymin, _, ystep = geotransform

    # print("geotransform", geotransform)
    # print('x_size', x_size, "y_size", y_size)

    xmax = xmin + (xstep * x_size)
    ymax = ymin + (ystep * y_size)

    # The geotransform can be based on any corner with negative step-sizes.
    # Get the actual min/max by taking the min() & max() of each pair.
    if invert_for_waffles:
        # The waffles command wants (xmin,xmax,ymin,ymax)
        return [min(xmin, xmax),
                max(xmin, xmax),
                min(ymin, ymax),
                max(ymin, ymax)], \
            [abs(xstep), abs(ystep)]
    else:
        # Everything else is looking for (xmin,ymin,xmax,ymax)
        return [min(xmin, xmax),
                min(ymin, ymax),
                max(xmin, xmax),
                max(ymin, ymax)], \
            [abs(xstep), abs(ystep)]

def get_dataset_epsg(gdal_dataset, warn_if_not_present=True):
    """Get the projection EPSG value from the dataset, if it's defined."""

    # Testing some things out.
    wkt=gdal_dataset.GetProjection()
    prj = pyproj.crs.CRS(wkt)

    epsg = prj.to_epsg()
    # Some of the CUDEM tiles are in NAD83 but the CRS doesn't explicitly give an
    # EPSG code. Handle that manually here.
    if epsg is None:
        # print(wkt)
        if wkt.find('GEOGCS["NAD83",') >= 0:
            return 4269
    else:
        return epsg

    # print(prj)
    # assert prj.lower().find("authority") >= 0

    # srs=osr.SpatialReference(wkt=prj)

    # # Under the AUTHORITY tag, it should have "ESPG" as the first value.
    # assert srs.GetAttrValue('authority', 0).strip().upper() == "EPSG"

    # # Get the EPSG number as the second value, converted to an integer.
    # return int(srs.GetAttrValue('authority', 1))


# TODO: Get coastline mask from:
    # 1. Copernicus mask (from waffles) -- this is what's existing.
    # 2. Extra tiles over Azerbaijan missing from Copernicus
    # 3. Outlying minor island (ask Matt about that) -- GMRT!
    # 4. Subtract off lakes mask from GLOBathy. -- Matt has that too
    # 5. Subtract buildings (just for dem validation). Matt has this too!
    # Combine all these ^^ together to return a complete coastline mask (possibly using stacks?)

def create_coastline_mask(input_dem,
                          return_bounds_step_epsg = False,
                          mask_out_lakes = True,
                          include_gmrt = False, # include_gmrt will include more minor outlying islands, many of which copernicus leaves out but GMRT includes
                          mask_out_buildings = False,
                          output_file=None,
                          verbose=True):
    """From a given DEM (.tif or otherwise), generate a coastline mask at the same grid and resolution.

    Uses the cudem waffles utility, which is handy for this.

    If output_file is None, put it in the same directory as the input_dem with the same base name.

    If return_ds_bounds_step_espg,
        Then return, in addition to the final mask output path.
        - The GDAL dataset of the DEM
        - The DEM bounding box (xmin, xmax, ymin, ymax)
        - The DEM pixel size in tuple: (xstep,ystep)
        - The DEM EPSG code

    If "round_up_half_pixel" is True, we expand the bounding-box size when performing
    the waffles command by an extra half-pixel in each direction. This eliminates occasional
    rounding errors where the pixel size causes the grid to be shortened by one pixel
    along one of the axes. It does not affect the bounding box returned to the calling function.

    Return the .tif name of the mask file generated."""

    input_ds = gdal.Open(input_dem, gdal.GA_ReadOnly)
    if not input_ds:
        raise FileNotFoundError("Input file '{input_dem}' not found.")

    bbox, step_xy = get_bounding_box_and_step(input_ds, invert_for_waffles=True)
    # print(bbox, step_xy)

    epsg = get_dataset_epsg(input_ds)
    # print(epsg)

    if output_file:
        output_filepath_base = os.path.splitext(output_file)[0]
    else:
        output_filepath_base = os.path.splitext(input_dem)[0] + "_coastline_mask"

    # output_dir, output_filebase = os.path.split(output_filepath_base)

    # Run a rich-text console for the output.
    console = rich.console.Console(force_jupyter=(True if is_this_run_in_ipython() else None))
    # Sometimes waffles can give some rounding error effects if the boundaries aren't exactly right.
    # If we round up half a pixel on each file extent size, it can ensure we

    waffle_args = ["waffles",
                   "-M","coastline:polygonize=False" + \
                       (":want_gmrt=True" if include_gmrt else "") + \
                       (":want_lakes=True" if mask_out_lakes else "") + \
                       (":want_buildings=True" if mask_out_buildings else ""),
                   "-R", "{0}/{1}/{2}/{3}".format(*bbox),
                   "-O", os.path.abspath(output_filepath_base),
                   "-P", "epsg:{0:d}".format(epsg),
                   "-E", str("{0:.16f}/{1:.16f}".format(step_xy[0], step_xy[1])),
                   "-D", etopo_config.etopo_cudem_cache_directory,
                   "--keep-cache",
                   "--nodata", str(etopo_config.etopo_ndv)]
                   # input_dem]

    if verbose:
        console.print("Running: [bold green]" + waffle_args[0] + "[/bold green] " + " ".join(waffle_args[1:]))
        # console.print("...in directory {}".format(output_dir))

    if verbose:
        kwargs = {}
    else:
        # Capture output and direct to /dev/null if we're running non-verbose.
        kwargs = {"stdout": subprocess.DEVNULL,
                  "stderr": subprocess.DEVNULL}
    # Put the data files generated in this processin the same directory as the output file.
    # This will be the home working directory of the process.
    kwargs["cwd"] = os.path.split(output_filepath_base)[0]

    subprocess.run(waffle_args,
                   check=True, **kwargs)

    if os.path.splitext(output_filepath_base)[1].lower() != ".tif":
        final_output_path = os.path.join(output_filepath_base + ".tif")
    else:
        final_output_path = output_filepath_base
    assert os.path.exists(final_output_path)

    if return_bounds_step_epsg:
        return final_output_path, bbox, step_xy, epsg
    else:
        return final_output_path

# 2: ICESat-2 data acquisition:
    # a) Get a file list from the NSIDC/icepyx API
    # b) Check existing files to see what we already have.
    # c) Download additional files of ATL03/ATL06/ATL08 data, where applicable.

def create_coastal_mask_filename(dem_name, target_dir=None):
    """If given a DEM name, create a filename for the coastal mask."""
    fdir, fname = os.path.split(os.path.abspath(dem_name))
    base, ext = os.path.splitext(fname)
    coastline_mask_fname = os.path.join(fdir if (target_dir is None) else target_dir, base + "_coastline_mask" + ext)
    return coastline_mask_fname


def get_coastline_mask_and_other_dem_data(dem_name,
                                          mask_out_lakes = True,
                                          mask_out_buildings=False,
                                          include_gmrt = True,
                                          target_fname_or_dir = None,
                                          verbose=True):
    """Get data from the DEM and a generated/opened coastline mask.

    Return, in this order:
        1. DEM GDAL dataset,
        2. DEM array (read from the dataset already)
        3. DEM bounding box (xmin, xmax, ymin, ymax)
        4. DEM EPSG code
        5. DEM resolution (xstep, ystep)
        6. coastline mask array in the same grid as the dem.
            (here derived from Copernicus data using the CUDEM "waffles" command))
    """
    if (target_fname_or_dir is None) or os.path.isdir(target_fname_or_dir):
        coastline_mask_file = create_coastal_mask_filename(dem_name, target_fname_or_dir)
    else:
        coastline_mask_file = target_fname_or_dir

    dem_ds = gdal.Open(dem_name, gdal.GA_ReadOnly)
    dem_array = dem_ds.ReadAsArray()

    coastline_ds = None

    # Get a coastline mask (here from Copernicus). If the file exists, use it.
    # If not, generate it.
    if not os.path.exists(coastline_mask_file):
        if verbose:
            print("Creating", coastline_mask_file)
        coastline_mask_file_out = create_coastline_mask(dem_name,
                                                        mask_out_lakes = mask_out_lakes,
                                                        mask_out_buildings = mask_out_buildings,
                                                        include_gmrt = include_gmrt,
                                                        return_bounds_step_epsg=False,
                                                        output_file=coastline_mask_file,
                                                        verbose=verbose)

        assert coastline_mask_file == coastline_mask_file_out

    if verbose:
        print("Reading", coastline_mask_file + "...", end="")
    coastline_ds = gdal.Open(coastline_mask_file, gdal.GA_ReadOnly)
    if verbose:
        print("Done.")

    dem_bbox, dem_step_xy = get_bounding_box_and_step(dem_ds)
    dem_epsg = get_dataset_epsg(dem_ds)

    coastline_mask_array = coastline_ds.GetRasterBand(1).ReadAsArray()
    coastline_ds = None

    return dem_ds, dem_array, dem_bbox, dem_epsg, dem_step_xy, coastline_mask_file, coastline_mask_array


def read_and_parse_args():
    parser = argparse.ArgumentParser(
        description="A script for creating coastline water masks from a DEM file. Return array is (0,1) for (water,land).")
    parser.add_argument("dem_filename", type=str, help="Input DEM.")
    parser.add_argument("dest", nargs="?", default="",
                        help="Destination file name, or file directory. If name is omitted: adds '_coastline_mask' to the input file name.")
    parser.add_argument("--dont_mask_out_buildings", default=False, action="store_true",
                        help="DO NOT Mask out areas that are covered by building polygons in the OpenStreetMap dataset. Masking out buildings is useful when using this for IceSat-2 validation.")
    parser.add_argument("--dont_mask_out_lakes", default=False, action="store_true",
                        help="DO NOT Mask out areas that are covered by lake polygons in the global HydroLakes dataset. Masking out lakes is useful when using this for IceSat-2 validation.")
    parser.add_argument("--dont_include_gmrt", default=False, action="store_true",
                        help="DO NOT Include land areas covered by the GMRT land-cover dataset. Including GMRT is useful for including many small outlying islands that Copernicus may exclude.")
    parser.add_argument("--quiet", "-q", action="store_true", default=False,
                        help="Run quietly.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = read_and_parse_args()

    create_coastline_mask(args.dem_filename,
                          return_ds_bounds_step_epsg=False,
                          mask_out_buildings = not args.dont_mask_out_buildings,
                          mask_out_lakes = not args.dont_mask_out_lakes,
                          include_gmrt = not args.dont_include_gmrt,
                          output_file=None if (args.dest.strip() == "") else args.dest,
                          verbose=not args.quiet)
