#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 11:20:27 2022

create_geopackage.py - A utility for createing geopackages from a directory of DEM tiles.
This data is then used to search for tiles in each dataset that overlap the grid boxes of ETOPO.

@author: mmacferrin
"""
import re
import os, os.path
import argparse
from osgeo import gdal
import shapely.geometry
import geopandas

# Import the parent directory so I can import anything else I need.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
import utils.traverse_directory

class DatasetGeopackage:
    """A class for handling geopackage collections of DEMs from various sources."""
    def __init__(self, filename, base_dir=None):
        self.filename = filename
        # If we are given a directory of files, save it.
        if base_dir:
            self.base_dir = base_dir
        # Otherwise, assume that the directory in which the geopackage is being placed is where we should search.
        else:
            self.base_dir = os.path.split(filename)[0]

        self.gdf = None
        self.default_layer_name = "DEMs"

    def get_gdf(self, verbose=True):
        if not self.gdf is None:
            return self.gdf
        elif os.path.exists(self.filename):
            self.gdf = geopandas.read_file(self.filename, layer=self.default_layer_name)
            if verbose:
                print(self.filename, "read.")
        else:
            self.gdf = self.create_dataset_geopackage(dir_or_list_of_files=self.base_dir,
                                                      geopackage_to_write=self.filename,
                                                      verbose=verbose)

        return self.gdf

    def create_dataset_geopackage(self,
                                  dir_or_list_of_files = None,
                                  geopackage_to_write = None,
                                  recurse_directory = True,
                                  file_filter_regex=r"\.tif\Z",
                                  verbose=True):
        """Open a list of files and/or traverse a directory of raster files and create a geopackage of their bounding boxes.

        Use 'file_filter_regex' to filter out any subset of files you want, ignoring others.
        If file_filter_regex=None, will attempt to read all files in a directory.

        Returns the geodataframe created.

        If geopackage_to_write is given, writes out the geopackage."""
        if not dir_or_list_of_files:
            dir_or_list_of_files = self.filename

        if not geopackage_to_write:
            geopackage_to_write = self.base_dir

        if type(dir_or_list_of_files) == str:
            if os.path.isdir(dir_or_list_of_files):
                # We were given a directory. Get the list of all dataset files from the directory (recursively, or not)
                if recurse_directory:
                    list_of_files = utils.traverse_directory.list_files(dir_or_list_of_files,
                                                                        regex_match=file_filter_regex,
                                                                        ordered=True,
                                                                        include_base_directory=True)
                else:
                    list_of_files = [os.path.join(dir_or_list_of_files, fn) for fn in os.listdir(dir_or_list_of_files) if re.search(file_filter_regex, fn) != None]
            elif os.path.exists(dir_or_list_of_files):
                # We were given a file (not a directory). Just put this as the file.
                list_of_files = [dir_or_list_of_files]
            else:
                raise FileNotFoundError("'{0}' does not exist on the local file system.".format(dir_or_list_of_files))
        elif type(dir_or_list_of_files) in (list, tuple):
            # We are given a list of files.
            list_of_files = dir_or_list_of_files

        # Create an empty dictionary of values
        gdf_dict = {"filename": [],
                    "xleft"   : [],
                    "ytop"    : [],
                    "xres"    : [],
                    "yres"    : [],
                    "xsize"   : [],
                    "ysize"   : [],
                    "geometry": []}

        dset_crs = None
        for fname in list_of_files:
            # Get the bounding box of each file.
            try:
                polygon, crs, xleft, ytop, xres, yres, xsize, ysize = \
                    self.create_outline_geometry_of_geotiff(fname)
            except Exception as e:
                # If the file could not be read, warn and skip it.
                raise UserWarning("Unable to read file '{0}'. Skipping. (Error message: {1})".format(fname, str(e)))
                continue

            if dset_crs is None:
                dset_crs = crs
            else:
                # Get the CRS of each file, make sure they match the others (all files should have same CRS)
                if dset_crs != crs:
                    if verbose:
                        print("ERROR: Dataset files do not all have the same horizontal projection!\n" + \
                              "File {0}:\n".format(fname) + \
                              crs + "\n" +
                              "Previous file(s)' projection:\n" +
                              dset_crs)
                        raise ValueError("Projections do not match in dataset.")

            # Save each record with:
                # - the filename from whence it came
                # - the x,y resolution
                # - the x,y size
                # - the geometry
            gdf_dict["filename"].append(os.path.split(fname)[1])
            gdf_dict["xleft"].append(xleft)
            gdf_dict["ytop"].append(ytop)
            gdf_dict["xres"].append(xres)
            gdf_dict["yres"].append(yres)
            gdf_dict["xsize"].append(xsize)
            gdf_dict["ysize"].append(ysize)
            gdf_dict["geometry"].append(polygon)

        # Save the output file as a GeoDataFrame.
        gdf = geopandas.GeoDataFrame(gdf_dict, geometry="geometry", crs=dset_crs)

        # Save the output file as a GeoPackage.
        if geopackage_to_write:
            gdf.to_file(geopackage_to_write, layer=self.default_layer_name, driver="GPKG")
            if verbose:
                print(geopackage_to_write, "written with {0} data tile outlines.".format(len(gdf.index)))

        return gdf


    def create_outline_geometry_of_geotiff(self, gtif_name):
        """Given a geotiff file name, return a shapely.geometry.Polygon object of the outline and return it.

        If 'return_proj_string_too' is set, also return the projection string as a second return value."""
        dset = gdal.Open(gtif_name)
        xleft, xres, _, ytop, _, yres = dset.GetGeoTransform()
        xsize, ysize = dset.RasterXSize, dset.RasterYSize
        proj = dset.GetProjection()

        xright = xleft + (xres * xsize)
        ybottom = ytop + (yres * ysize)

        # Create shapely geometry polygon object from the points.
        polygon = shapely.geometry.Polygon([[xleft, ybottom],
                                            [xleft, ytop],
                                            [xright, ytop],
                                            [xright, ybottom],
                                            [xleft, ybottom]])

        # Return the geometry, and if requested, the projection wkt as well.
        return polygon, proj, xleft, ytop, xres, yres, xsize, ysize


    def subset_by_polygon(self, polygon, polygon_crs=None, check_if_same_crs=True):
        """Given a shapely polygon object, return all records that intersect the polygon."""
        gdf = self.get_gdf()

        if check_if_same_crs and polygon_crs != None:
            # TODO: Check to see if this is robust or not. May not be even if
            # the CRS's are basically the same. Perhaps get the EPSG number of each CRS and see if *those* are the same.
            # But for now, this quick check will suffice unless it starts breaking erroneously.
            assert gdf.crs == polygon_crs

        return gdf[gdf.intersects(polygon)]

    def subset_by_geotiff(self, gtif_file):
        """Given a geotiff file, return all records that intersect the bounding-box outline of this geotiff."""
        polygon, crs, _, _, _, _, _, _ = self.create_outline_geometry_of_geotiff(gtif_file)
        return self.subset_by_polygon(polygon, polygon_crs=crs, check_if_same_crs=True)

    def print_full_gdf(self):
        """Print a full geodataframe, with all rows and columnns. Resets display options back to defaults afterward."""
        orig_max_rows_opt = geopandas.get_option("max_rows")
        orig_max_cols_opt = geopandas.get_option("max_columns")

        geopandas.set_option('max_rows', None)
        geopandas.set_option('max_columns', None)
        print(self.get_gdf(verbose=False))

        geopandas.set_option('max_rows', orig_max_rows_opt)
        geopandas.set_option('max_columns', orig_max_cols_opt)
        return

def create_and_parse_args():
    parser = argparse.ArgumentParser(description="Create a GeoDataFrame and GeoPackage from a directory of geotiffs.")
    parser.add_argument("directory_name", type=str, help="Name of the directory to collect raster data files.")
    parser.add_argument("geopackage_file", type=str, help="Geopackage file to output (.gpkg).")
    parser.add_argument("-file_filter_regex", "-regex", type=str, default=r"\.tif\Z", help="A regular expression to select appropriate datafiles. Default to files ending in \".tif\"")
    parser.add_argument("--no_recurse", "-nr", action="store_true", default=False, help="Avoid recursion into sub-directories. (Default: False, go into all the sub-directories.)")
    parser.add_argument("--quiet", "-q", action="store_true", default=False, help="Suppress output.")

    return parser.parse_args()

if __name__ == "__main__":
    args = create_and_parse_args()
    dataset_object = DatasetGeopackage(args.geopackage_file,
                                       base_dir=args.directory_name)

    dataset_object.create_dataset_geopackage(dir_or_list_of_files = args.directory_name,
                                             geopackage_to_write = args.geopackage_file,
                                             recurse_directory= not args.no_recurse,
                                             file_filter_regex=args.file_filter_regex,
                                             verbose = not args.quiet
                                             )
