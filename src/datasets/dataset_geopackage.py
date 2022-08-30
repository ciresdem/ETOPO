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
import shapely.ops
import geopandas
import pyproj
import pygeos
import numpy
import warnings

# Import the parent directory so I can import anything else I need.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
import utils.traverse_directory
import utils.progress_bar
import utils.configfile
import warnings

class DatasetGeopackage:
    """A class for handling geopackage collections of DEMs from various sources."""
    def __init__(self, dataset_configfile):
        """Create the base class. Provide either the config object associated with
        the source dataset, or the name of the configfile .ini file and a config
        object will be created."""
        if type(dataset_configfile) == utils.configfile.config:
            dset_config = dataset_configfile
        else:
            dset_config = utils.configfile.config(configfile=dataset_configfile)

        self.config = dset_config
        self.filename = self.config._abspath(self.config.geopackage_filename)
        # Save the directory with files in it.
        self.base_dir = self.config._abspath(self.config.source_datafiles_directory)

        self.gdf = None
        self.default_layer_name = "DEMs"
        self.regex_filter = self.config.datafiles_regex

    def get_gdf_filename(self, resolution_s = None):
        if (self.filename.find("{0}") >= 0) and (resolution_s is not None):
            return self.filename.format(resolution_s)
        else:
            return self.filename

    def get_gdf(self, resolution_s = None, verbose=True):

        filename = self.get_gdf_filename(resolution_s = resolution_s)

        if not self.gdf is None:
            return self.gdf

        elif os.path.exists(filename):
            self.gdf = geopandas.read_file(filename, layer=self.default_layer_name)
            if verbose:
                print(filename, "read.")
        else:
            if verbose:
                print(filename, "does not exist. Creating...")
            self.gdf = self.create_dataset_geopackage(resolution_s = resolution_s,
                                                      verbose=verbose)
            if verbose:
                print("Done.")

        return self.gdf

    def create_dataset_geopackage(self, dir_or_list_of_files = None,
                                        resolution_s = None,
                                        recurse_directory = True,
                                        allow_epsg_matches_only = True,
                                        verbose = True):
        """Open a list of files and/or traverse a directory of raster files and create a geopackage of their bounding boxes.

        Use 'file_filter_regex' to filter out any subset of files you want, ignoring others.
        If file_filter_regex=None, will attempt to read all files in a directory.

        Returns the geodataframe created.

        If allow_epsg_matches_only is set, the projections can be considered "the same" even if the exact .prj text isn't identical.
        This helps some CUDEM tiles that are in various forms of NAD83 be identical even if their header info isn't.

        If geopackage_to_write is given, writes out the geopackage."""
        if not dir_or_list_of_files:
            if resolution_s is None:
                dir_or_list_of_files = self.base_dir
            else:
                if resolution_s == 15 and hasattr(self.config, "source_datafiles_directory_15s"):
                    dir_or_list_of_files = self.config._abspath(self.config.source_datafiles_directory_15s)
                elif resolution_s == 1 and hasattr(self.config, "source_datafiles_directory_1s"):
                    dir_or_list_of_files = self.config._abspath(self.config.source_datafiles_directory_1s)
                else:
                    dir_or_list_of_files = self.base_dir

        # geopackage_to_write = self.base_dir

        if type(dir_or_list_of_files) == str:
            if os.path.isdir(dir_or_list_of_files):
                # We were given a directory. Get the list of all dataset files from the directory (recursively, or not)
                if recurse_directory:
                    list_of_files = utils.traverse_directory.list_files(dir_or_list_of_files,
                                                                        regex_match=self.regex_filter,
                                                                        ordered=True,
                                                                        include_base_directory=True)
                else:
                    list_of_files = [os.path.join(dir_or_list_of_files, fn) for fn in os.listdir(dir_or_list_of_files) if re.search(self.regex_filter, fn) != None]
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
        previous_fname = None
        for i,fname in enumerate(list_of_files):
            # Get the bounding box of each file.
            try:
                polygon, crs, xleft, ytop, xres, yres, xsize, ysize = \
                    self.create_outline_geometry_of_geotiff(fname)
            except Exception as e:
                # If the file could not be read, warn and skip it.
                raise UserWarning("Unable to read file '{0}'. Skipping. (Error message: {1})".format(fname, str(e)))
                continue

            if previous_fname is None:
                previous_fname = fname

            if dset_crs is None:
                dset_crs = crs
            else:
                dset_epsg = pyproj.CRS.from_string(dset_crs).to_epsg()
                dem_epsg = pyproj.CRS.from_string(crs).to_epsg()
                # Get the CRS of each file, make sure they match the others (all files should have same CRS)
                # Some of the CUDEM tiles have different EPSG's even though they're both in NAD83 horizontal datums.
                # Make a special-case exception for those.
                if not (dset_crs == crs or (allow_epsg_matches_only and ((dem_epsg == dset_epsg) or ((dem_epsg in (4269,5498,None)) and (dset_epsg in (4269,5498,None)))))):
                    if verbose:
                        print("ERROR: Dataset files do not all have the same horizontal projection!\n" + \
                              "File {0}:\n".format(fname) + \
                              crs + "\n" + "EPSG: " + str(pyproj.CRS.from_string(crs).to_epsg()) + "\n\n" +
                              "Previous File {0}:\n".format(previous_fname) +
                              dset_crs + "\n" + "EPSG: " + str(pyproj.CRS.from_string(dset_crs).to_epsg()))
                        raise ValueError("Projections do not match in dataset.")

            # Save each record with:
                # - the filename from whence it came
                # - the x,y resolution
                # - the x,y size
                # - the geometry
            gdf_dict["filename"].append(fname)
            gdf_dict["xleft"].append(xleft)
            gdf_dict["ytop"].append(ytop)
            gdf_dict["xres"].append(xres)
            gdf_dict["yres"].append(yres)
            gdf_dict["xsize"].append(xsize)
            gdf_dict["ysize"].append(ysize)
            gdf_dict["geometry"].append(polygon)

            if verbose:
                utils.progress_bar.ProgressBar(i+1, len(list_of_files), suffix="{0}/{1}".format(i+1,len(list_of_files)))

        # Save the output file as a GeoDataFrame.
        gdf = geopandas.GeoDataFrame(gdf_dict, geometry="geometry", crs=dset_crs)

        # Save the output file as a GeoPackage.
        gpkg_fname = self.get_gdf_filename(resolution_s = resolution_s)

        gdf.to_file(gpkg_fname, layer=self.default_layer_name, driver="GPKG")
        if verbose:
            print(gpkg_fname, "written with {0} data tile outlines.".format(len(gdf.index)))

        return gdf

    def create_outline_geometry_of_geotiff(self, gtif_name):
        """Given a geotiff file name, return a shapely.geometry.Polygon object of the outline and return it.

        If 'return_proj_string_too' is set, also return the projection string as a second return value."""
        if not os.path.exists(gtif_name):
            raise FileNotFoundError(gtif_name + " not found.")
        dset = gdal.Open(gtif_name, gdal.GA_ReadOnly)
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

        # Return the geometry, and the projection wkt as well.
        return polygon, proj, xleft, ytop, xres, yres, xsize, ysize


    def subset_by_polygon(self, polygon, polygon_crs, resolution_s = None, verbose=True):
        """Given a shapely polygon object, return all records that intersect the polygon.

        If the polygon_crs does not match the geopackage crs, conver the polygon into
        the geopackage CRS before performing the intersection.
        """
        gdf = DatasetGeopackage.get_gdf(self, resolution_s = resolution_s, verbose=verbose)

        gdf_crs_obj  = pyproj.crs.CRS(gdf.crs)
        poly_crs_obj = pyproj.crs.CRS(polygon_crs)
        # If the CRS's are the same, use the polygon as-is
        if gdf_crs_obj.equals(poly_crs_obj):
            polygon_to_use = polygon
        # If the CRS's are not the same, convert the polygon into
        # the geopackage CRS and go from there.
        else:
            project = pyproj.Transformer.from_crs(poly_crs_obj, gdf_crs_obj, always_xy=True).transform
            # Interpolate the polygon into a lot more segments, to better handle warping in reprojected coordinates.
            polygon_interpolated = shapely.geometry.Polygon([polygon.boundary.interpolate(i, normalized=True) for i in numpy.linspace(0,1,101)])
            polygon_to_use = shapely.ops.transform(project, polygon_interpolated)
            # In some projected coordinate systems, some lat/lons will become infinities. Filter those out.
            Xs,Ys = polygon_to_use.exterior.coords.xy
            polygon_to_use = shapely.geometry.Polygon([[x,y] for x,y in zip(Xs,Ys) if (x!=numpy.inf and y!=numpy.inf)])

        # Get all the tiles that truly intersect but don't just "touch" the polygon on its boundary without overlapping.
        try:
            warnings.filterwarnings("error")
            gdf_partial = gdf[gdf.intersects(polygon_to_use) & ~gdf.touches(polygon_to_use)]
            warnings.filterwarnings("default")
            return gdf_partial
        except pygeos.GEOSException:
            # Sometimes the polygon is (at least partially) outside the bounds of this
            # projection, creating instances where coordinates are giving infinite values, causing issues here.
            # If this happens, just return the empty dataframe, as the polygon realistically (in the case
            # of all ETOPO's data sources) does not overlap any parts of this dataset.
            # This could omit things in devious cases where polygons cover large portions of the earth's surface.
            # However, for ETOPO, this is not really the case.
            return gdf[gdf.index == (gdf.index.min()-1)]
        except RuntimeWarning as e:
            print("Insersection warning in datasdet", self.config.dataset_name)
            print("polygon", polygon_to_use)
            print("ESPG:", gdf_crs_obj.to_epsg())
            return gdf[gdf.index == (gdf.index.min()-1)]


    def subset_by_geotiff(self, gtif_file, resolution_s = None, verbose = True):
        """Given a geotiff file, return all records that intersect the bounding-box outline of this geotiff."""
        polygon, crs, _, _, _, _, _, _ = self.create_outline_geometry_of_geotiff(gtif_file)
        return self.subset_by_polygon(polygon,
                                      crs,
                                      resolution_s = resolution_s,
                                      verbose = verbose)

    def print_full_gdf(self, resolution_s = None, verbose = True):
        """Print a full geodataframe, with all rows and columnns. Resets display options back to defaults afterward."""
        # Get the original default options for pandas displays
        orig_max_rows_opt = geopandas.get_option("max_rows")
        orig_max_cols_opt = geopandas.get_option("max_columns")

        # Set them to no-limit maximums
        geopandas.set_option('max_rows', None)
        geopandas.set_option('max_columns', None)
        # Print out the dataset
        print(self.get_gdf(resolution_s = resolution_s, verbose=verbose))

        # Re-set them to their original values.
        geopandas.set_option('max_rows', orig_max_rows_opt)
        geopandas.set_option('max_columns', orig_max_cols_opt)
        return

class ETOPO_Geopackage(DatasetGeopackage):
    """Inherited from the DatasetGeopackage, slightly modified __init__()
    interface to handle specifically the ETOPO source files."""
    def __init__(self, resolution):
        resolution = int(resolution)
        assert resolution in (1,15,60)

        ## These member variables are the same as the ones initiated in DatasetGeopackage.__init__()

        # The default configfile points to the etopo_config.ini in the project base directory
        self.config = utils.configfile.config()
        self.resolution = resolution

        if resolution == 1:
            self.filename = self.config.etopo_tile_geopackage_1s
        elif resolution == 15:
            self.filename = self.config.etopo_tile_geopackage_15s
        else:
            self.filename = self.config.etopo_tile_geopackage_60s

        self.base_dir = self.config._abspath(os.path.join(self.config.etopo_empty_tiles_directory, str(resolution) + "s"))

        self.gdf = None
        self.default_layer_name = "DEMs"
        self.regex_filter = r"ETOPO_(\d){4}_v(\d)_(\d){1,2}s_(\w){7}\.tif\Z"

        ## These member variables are unique to the ETOPO grids specifically.

        self.dlist_dir = self.config.etopo_datalist_directory

    def get_gdf(self, crm_only_if_1s=False, verbose=True):
        """Sub-class method of get_gdf, with the option to subset to only CRM tiles (US East Coast)
        if we're in 1-deg tile space."""
        # If we're not at 1s resolution, or we're not flagged to only get the CRM tiles, then just move along.
        if (self.resolution != 1) or (crm_only_if_1s == False):
            return super().get_gdf(resolution_s = self.resolution, verbose=verbose)

        # Otherwise, only send back the geodataframe that includes the CRM tiles.
        assert (self.resolution == 1) and (crm_only_if_1s is True)
        crm_gdf = geopandas.read_file(self.config.crm_tiles_outline_shapefile)
        # It should just be a one-feature shapefile. Sanity check here.
        assert len(crm_gdf) == 1

        # Get the one polygon from the crm outline shapefile.
        crm_polygon = crm_gdf.geometry.tolist()[0]
        # Return the subset of the geodataframe that intersects the CRM outline polygon.
        return self.subset_by_polygon(crm_polygon, crm_gdf.crs, verbose=verbose)

    def add_dlist_paths_to_gdf(self, save_to_file_if_not_already_there=True, crm_only_if_1s = True, verbose=True):
        """Add a 'dlist' column to the geodataframe that lists the location of the
        approrpriate source-datasets dlist for each ETOPO tile.

        Return the gdf with this new field.

        If "save_to_file_if_not_already_there", save this datalsit to the file
        if it doesn't already exist in the geodataframe."""
        gdf = self.get_gdf(crm_only_if_1s = crm_only_if_1s, verbose=verbose)

        # If the "dlist" column already exists, just return it.
        if 'dlist' in gdf.columns:
            return gdf

        # Little lambda function for converting the grid filename to a dlist filename.
        dlist_func = lambda fn: os.path.join(self.dlist_dir,
                                             str(self.resolution) + "s",
                                             os.path.splitext(os.path.split(fn)[1])[0] + ".datalist")

        # Apply the function to every cell of the 'filename' column.
        # Put it in a new "dilst" column.
        gdf["dlist"] = gdf['filename'].apply(dlist_func)

        # Save the output file as a GeoPackage.
        if save_to_file_if_not_already_there:
            gdf.to_file(self.filename, layer=self.default_layer_name, driver="GPKG")
            if verbose:
                print(self.filename, "written with {0} data tile outlines.".format(len(gdf.index)))

        return gdf


def create_and_parse_args():
    parser = argparse.ArgumentParser(description="Create a GeoDataFrame and GeoPackage from a directory of geotiffs.")
    parser.add_argument("directory_name", type=str, help="Name of the directory to collect raster data files.")
    parser.add_argument("geopackage_file", type=str, help="Geopackage file to output (.gpkg).")
    parser.add_argument("-file_filter_regex", "-regex", type=str, default=r"\.tif\Z", help="A regular expression to select appropriate datafiles. Default to files ending in \".tif\"")
    parser.add_argument("--no_recurse", "-nr", action="store_true", default=False, help="Avoid recursion into sub-directories. (Default: False, go into all the sub-directories.)")
    parser.add_argument("--quiet", "-q", action="store_true", default=False, help="Suppress output.")

    return parser.parse_args()

if __name__ == "__main__":
    # import sys
    # sys.exit(0)

    ET1 = ETOPO_Geopackage(1)
    # gdf = ET1.get_gdf(crm_only_if_1s=True)
    # fname = os.path.splitext(ET1.filename)[0] + "_CRM_only.gpkg"
    # gdf.to_file(fname, driver="GPKG")
    # print(fname, "written.")
    # ET1.create_dataset_geopackage()
    # ET1.add_dlist_paths_to_gdf()
    # import sys
    # sys.exit(0)

    # args = create_and_parse_args()
    # dataset_object = DatasetGeopackage(args.geopackage_file,
    #                                    base_dir=args.directory_name)

    # dataset_object.create_dataset_geopackage(dir_or_list_of_files = args.directory_name,
    #                                          geopackage_to_write = args.geopackage_file,
    #                                          recurse_directory= not args.no_recurse,
    #                                          file_filter_regex=args.file_filter_regex,
    #                                          verbose = not args.quiet
    #                                          )
