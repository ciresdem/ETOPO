# -*- coding: utf-8 -*-

"""Source code for the BlueTopo ETOPO source dataset class."""

import os
from osgeo import gdal
import re
import shutil
import pyproj

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset
# import utils.progress_bar

utm_zone_regex = r"(?<=NAD83 / UTM zone )(\d{2})[NS]"
# These cover all the UTM zones covered by the BlueTopo tiles.
epsg_lookups  = {"14N" : 26914,
                 "15N" : 26915,
                 "16N" : 26916,
                 # "17N" : 26917,
                 "18N" : 26918,
                 "19N" : 26919,
                 }

# Suppress gdal warnings in this file.
gdal.SetConfigOption('CPL_LOG', "/dev/null")

class source_dataset_BlueTopo(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "BlueTopo_config.ini" )):
        """Initialize the BlueTopo source dataset object."""

        super(source_dataset_BlueTopo, self).__init__("BlueTopo", configfile)

    def get_file_epsg(self, filename, utm_zone_name=False):
        """Given a BlueTopo tile, return the EPSG code for it."""
        # If we're using a "converted" file but can't find it, look for the original.
        if not os.path.exists(filename) and filename.find("/converted/") >= 1:
            # In some of these files, we tried (unsuccessfully) to convert it.
            filename = filename.replace("/converted/", "/").replace("_egm2008.tif", ".tif")
        assert os.path.exists(filename)
        ds = gdal.Open(filename, gdal.GA_ReadOnly)
        wkt = ds.GetProjection()
        utmz = re.search(utm_zone_regex, wkt).group()
        if utm_zone_name:
            return utmz
        else:
            return epsg_lookups[utmz]

    def split_files_into_epsg_folders(self, use_symlinks=True, verbose=True):
        """Since the BlueTopo is in five different UTM projections, I can't
        really handle it in one layer. Here, we copy the files (or create symlinks)
        into 5 different folders so that I can create 5 different datasets. Should do the job."""
        gdf = self.get_geodataframe(verbose=verbose)
        basedir = self.config.source_datafiles_directory
        for i,row in gdf.iterrows():
            fname = row.filename

            # We want the original tiles, not converted ones.
            if fname.find("/converted/") >= 1:
                fname = fname.replace("/converted/", "/").replace("_egm2008.tif",".tif")
            assert os.path.exists(fname)

            utm_zone = self.get_file_epsg(fname, utm_zone_name = True)

            dest_dir = os.path.join(basedir, utm_zone)
            if not os.path.exists(dest_dir):
                os.mkdir(dest_dir)
            dest_fname = os.path.join(dest_dir, os.path.split(fname)[1])
            if os.path.exists(dest_fname):
                continue
            if use_symlinks:
                os.symlink(fname, dest_fname)
                if verbose:
                    print(dest_fname, "link created.")
            else:
                shutil.copyfile(fname, dest_fname)
                if verbose:
                    print(dest_fame, "created.")
        return

    def list_utm_zones(self, as_epsg=False, verbose=False):
        gdf = self.get_geodataframe(verbose=verbose)
        filenames = gdf.filename.tolist()
        set_of_utm_zones = set()
        for i,fname in enumerate(filenames):
            if not os.path.exists(fname) and fname.find("/converted/") >= 1:
                # In some of these files, we tried (unsuccessfully) to convert it.
                fname = fname.replace("/converted/", "/").replace("_egm2008.tif",".tif")
            assert os.path.exists(fname)

            # Some of these files have weird warnings when we read them. Ignore the warnings.
            ds = gdal.Open(fname, gdal.GA_ReadOnly)
            wkt = ds.GetProjection()
            utmz = re.search(utm_zone_regex, wkt)
            if utmz is None:
                if verbose:
                    print("CANNOT FIND UTM ZONE in wkt of {0}:".format(os.path.split(fname)))
                    print(wkt)
                continue

            set_of_utm_zones.add(utmz.group())

        list_of_utm_zones = sorted(list(set_of_utm_zones))

        if verbose:
            for utmz in list_of_utm_zones:
                print(utmz)

        if as_epsg:
            return [espg_lookups[utmz] for utmz in list_of_utm_zones]
        else:
            return list_of_utm_zones

    def get_crs(self, as_epsg=True):
        """Note: this only works if using one of the BlueTopo sub-classes with a UTM Zone on it, like 'BlueTopo_14N'."""
        utmz = self.dataset_name[-3:]
        if as_epsg:
            return epsg_lookups[utmz]
        else:
            return pyproj.crs.CRS.from_epsg(epsg_lookups[utmz])

    def override_gdf_projection(self, verbose=True):
        """If a geodataframe was just created, it probably has a funny projection, as these BlueTopo tiles do.
        Override it with the projection we want here."""
        crs = self.get_crs(as_epsg=False)
        gpkg_object = self.get_geopkg_object(verbose=verbose)
        gdf = gpkg_object.get_gdf(verbose=verbose)

        if gdf.crs is not None and gdf.crs.equals(crs):
            return

        gdf.set_crs(crs, allow_override=True)
        gpkg_fname = gpkg_object.get_gdf_filename()
        gdf.to_file(gpkg_fname, layer=gpkg_object.default_layer_name, driver="GPKG")
        if verbose:
            print(os.path.split(gpkg_fname)[1], "written in EPSG:{0}".format(self.get_crs(as_epsg=True)), "with {0} tiles.".format(len(gdf)))
        return

if "__main__" == __name__:
    bt = source_dataset_BlueTopo()
    bt.split_files_into_epsg_folders()