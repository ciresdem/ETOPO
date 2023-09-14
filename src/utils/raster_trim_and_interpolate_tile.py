
import argparse
import os
import subprocess
import geopandas
import fiona
from osgeo import gdal
import pyproj
import shutil
import numpy

def bounding_raster_gt(poly, raster_gt, raster_size_xy, resolution_multiplier=1, pixel_buffer=1, return_indices=False):
    """For a given polygon and raster geotransform, create a geotransform and (xsize, ysize) for an output raster.

    Enhance the resolution by resolution_multiplier.
    Buffer by "pixel_buffer" pixels on each side.

    Return values:
    - raster_geotransform (6-tuple)
    - raster x-size, y-size (2-tuple)"""
    minx, stepx, _, maxy, _, stepy = raster_gt
    raster_xsize, raster_ysize = raster_size_xy

    if resolution_multiplier != 1.0:
        stepx = stepx / resolution_multiplier
        stepy = stepy / resolution_multiplier
        raster_xsize = raster_xsize * resolution_multiplier
        raster_ysize = raster_ysize * resolution_multiplier

    poly_xmin, poly_ymin, poly_xmax, poly_ymax = poly.bounds

    poly_xmin_i = max(int(numpy.floor((poly_xmin - minx) / stepx)) - pixel_buffer, 0)
    poly_xmax_i = min(int(numpy.ceil((poly_xmax - minx) / stepx)) + pixel_buffer, raster_xsize)
    poly_ymin_j = min(int(numpy.ceil((poly_ymin - maxy) / stepy)) + pixel_buffer, raster_ysize)
    poly_ymax_j = max(int(numpy.floor((poly_ymax - maxy) / stepy)) - pixel_buffer, 0)

    # return_gt = (minx + (poly_xmin_i * stepx), stepx, 0,
    #              maxy + (poly_ymax_j * stepy), 0, stepy)
    if return_indices:
        return_bounds = (poly_xmin_i, poly_ymin_j, poly_xmax_i, poly_ymax_j)
    else:
        return_bounds = (minx + (poly_xmin_i * stepx), maxy + (poly_ymin_j * stepy),
                         minx + (poly_xmax_i * stepx), maxy + (poly_ymax_j * stepy))

    return_size_xy = (poly_xmax_i - poly_xmin_i, poly_ymin_j - poly_ymax_j)

    return return_bounds, return_size_xy

def cut_and_fill_polygons(raster_fname,
                          shp_or_gpkg_fname,
                          pix_buffer=3,
                          outdir=None,
                          out_basename=None,
                          skip_interpolation=False,
                          si=0,
                          max_distance=100,
                          overwrite=False,
                          verbose=True):
    """Pull out tiles from a raster, cut holes in the tiles from polygons, and interpolate across them.

    This creates tiles that smooth over unwanted artifacts and can be laid over the original tiles to create smoothly-
    interpolated surfaces."""

    assert os.path.exists(raster_fname)
    assert os.path.exists(shp_or_gpkg_fname)

    # If it's a GPKG and a layer name is not given, simply select the firat layer in the file.
    # if os.path.splitext(shp_or_gpkg_fname)[1].lower() == ".gpkg":
    #     if layer_name is None:
    #         layer_name = fiona.listlayers(shp_or_gpkg_fname)[0]

    # Get raster metadata: geotransform, size
    ds = gdal.Open(raster_fname, gdal.GA_ReadOnly)
    raster_gt = ds.GetGeoTransform()
    raster_epsg = pyproj.CRS.from_user_input(ds.GetProjection()).to_epsg()
    xleft, xstep, _, ytop, _, ystep = raster_gt
    raster_size_xy = ds.RasterXSize, ds.RasterYSize
    raster_datatype = gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)

    gdf = geopandas.read_file(shp_or_gpkg_fname)

    if outdir is None:
        outdir = os.path.dirname(raster_fname)

    if out_basename is None:
        out_basename = os.path.splitext(os.path.basename(raster_fname))[0]

    for idx, row in gdf.iterrows():
        polygon = row.geometry

        out_tilename = os.path.join(outdir, out_basename +
                                    "_{0}.tif".format(idx))

        if os.path.exists(out_tilename) and not overwrite:
            if verbose:
                print(os.path.basename(out_tilename), "already exists.\n")
            continue

        # Get the pixel bounds around the polygon for which we'll pull out a tile.
        tile_bounds, tile_size_xy = bounding_raster_gt(polygon, raster_gt, raster_size_xy, pixel_buffer=pix_buffer)

        mask_tilename = os.path.join(outdir, out_basename + "_{0}_TEMP_mask.tif".format(idx))

        # Create a temp one-off GPKG for the single polygon.
        single_gdf = geopandas.GeoDataFrame(data={"geometry": [polygon]}, geometry="geometry", crs=gdf.crs)
        single_gdf_tempname = mask_tilename.replace("_TEMP_mask.tif", "_TEMP_poly.gpkg")
        if os.path.exists(single_gdf_tempname):
            os.remove(single_gdf_tempname)
        single_gdf.to_file(single_gdf_tempname, layer="poly", driver="GPKG")

        ############################################################
        # First, Use gdal_rasterize to create a polygon mask
        ############################################################
        # Pull out a tile from the raster. Save it to a small geotiff.
        gdal_raster_cmd = ["gdal_rasterize",
                           "-l", "poly",
                           "-ot", "Byte",
                           "-co", "COMPRESS=DEFLATE",
                           "-burn", "1",
                           "-a_srs", f"EPSG:{raster_epsg}",
                           "-tr", str(abs(xstep)), str(abs(ystep)),
                           "-te"] + [str(b) for b in tile_bounds] + \
                           ["-init", "0",
                            single_gdf_tempname,
                            mask_tilename]

        if verbose:
            print(" ".join(gdal_raster_cmd))

        subprocess.run(gdal_raster_cmd, capture_output=not verbose)

        if os.path.exists(mask_tilename):
            if verbose:
                print(os.path.basename(mask_tilename), "written.")
        else:
            raise FileNotFoundError(mask_tilename)

        ############################################################
        # Second, clip the GEBCO tile to the same bounding box.
        ############################################################
        tile_tempname = mask_tilename.replace("_TEMP_mask.tif", "_TEMP_tile_orig.tif")
        if os.path.exists(tile_tempname):
            os.remove(tile_tempname)
        gdalwarp_cmd = ["gdalwarp", "-te"] + [str(b) for b in tile_bounds] + \
                       [raster_fname, tile_tempname]

        if verbose:
            print(" ".join(gdalwarp_cmd))

        subprocess.run(gdalwarp_cmd, capture_output=not verbose)

        if os.path.exists(tile_tempname):
            if verbose:
                print(os.path.basename(tile_tempname), "written.")
        else:
            raise FileNotFoundError(tile_tempname)

        ############################################################
        # Third, use the mask to cut out a hole from the temp tile.
        ############################################################
        hole_tilename = mask_tilename.replace("_TEMP_mask.tif", "_TEMP_hole.tif")

        gdal_calc_cmd = ["gdal_calc.py", "--calc=(1-A)*B",
                         "-A", mask_tilename,
                         "--A_band", "1",
                         "-B", tile_tempname,
                         "--B_band", "1",
                         "--NoDataValue", "0",
                         "--type", raster_datatype,
                         "--outfile", hole_tilename]

        if os.path.exists(hole_tilename):
            os.remove(hole_tilename)

        if verbose:
            print(" ".join(gdal_calc_cmd))

        subprocess.run(gdal_calc_cmd, capture_output=not verbose)

        if os.path.exists(hole_tilename):
            if verbose:
                print(os.path.basename(hole_tilename), "written.")
        else:
            raise FileNotFoundError(hole_tilename)

        ############################################################
        # Fourth, interpolate accross NDV with gdal_fillnodata.py
        ############################################################
        if skip_interpolation:
            # If we're not interpolating, then we're done, just move the file.
            shutil.move(hole_tilename, out_tilename)
        else:
            gdal_fillnodata_cmd = ["gdal_fillnodata.py",
                                   "-si", str(si)] + \
                                  ([] if max_distance is None else ["-md", str(max_distance)]) + \
                                  [hole_tilename, out_tilename]

            if os.path.exists(out_tilename):
                os.remove(out_tilename)

            if verbose:
                print(" ".join(gdal_fillnodata_cmd))

            subprocess.run(gdal_fillnodata_cmd, capture_output=not verbose)

            if os.path.exists(out_tilename):
                if verbose:
                    print(os.path.basename(out_tilename), "written.")
            else:
                raise FileNotFoundError(out_tilename)

        # Remove the temp files.
        if verbose:
            print("Removing temp tiles.")

        if os.path.exists(single_gdf_tempname):
            os.remove(single_gdf_tempname)
        if os.path.exists(mask_tilename):
            os.remove(mask_tilename)
        if os.path.exists(tile_tempname):
            os.remove(tile_tempname)
        if os.path.exists(hole_tilename):
            os.remove(hole_tilename)

        if verbose:
            print("Done.\n")


def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Cut holes in a raster using vector polygons, and save the a tile" + \
                                                 " surrounding each polygon. Interpolate over the missing values.")
    parser.add_argument("raster", help="The raster file to poke holes in.")
    parser.add_argument("polygon_shp_or_gpkg", help="The .shp or .gpkg vector file containing polygons" + \
                        " to cut out of the image.")
    parser.add_argument("-buffer", type=int, default=3, help="Buffer around the polygon (in pixels) to" + \
                                                             "create the tile. Default 3.")
    parser.add_argument("-layer", type=str, default=None, help="The layer name in a geopackage to read." + \
                                                               "Default: Just read the first layer, whichever it is.")
    parser.add_argument("-outdir", "-o", default=None, type=str, help="Output to put the resulting tiles.")
    parser.add_argument("-outbase", "-ob", default=None, type=str, help="Base of the output filenames. A" + \
                        " number will be added to the end of it. Default, just use the raster filename as a basename.")
    parser.add_argument("--skip_interpolation", default=False, action="store_true", help="Skip" + \
                        " interpolation and just leave NoData in the hole. Default: interpolate over the holes.")
    parser.add_argument("-si", default=0, type=int, help="Number of 3x3 average smoothing iterations to"
                                                         "dampen artifacts. Default: 0")
    parser.add_argument("-max_distance", "-md", default=100, type=int, help="The maximum distance (in" + \
                            " pixels) the the algorithm will search for values to interpolate. Default 100 pixels.")
    parser.add_argument("--overwrite", "-ov", default=False, action="store_true", help="Overwrite over" + \
                                                                                       " previously-written files.")
    parser.add_argument("--quiet", "-q", default=False, action="store_true", help="Operate in quiet mode.")

    return parser.parse_args()

if __name__ == "__main__":
    args = define_and_parse_args()

    cut_and_fill_polygons(args.raster,
                          args.polygon_shp_or_gpkg,
                          pix_buffer=args.buffer,
                          outdir=args.outdir,
                          out_basename=args.outbase,
                          skip_interpolation=args.skip_interpolation,
                          si=args.si,
                          max_distance=args.max_distance,
                          overwrite=args.overwrite,
                          verbose=not args.quiet)