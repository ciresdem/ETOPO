# -*- coding: utf-8 -*-

"""Source code for the BlueTopo ETOPO source dataset class."""

import os
from osgeo import gdal
import re
import shutil
import pyproj
import subprocess

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

    def generate_tile_datalist_entries(self, polygon, polygon_crs=None, resolution_s = None, verbose=True, weight=None):
        """Given a polygon (ipmortant, in WGS84/EPSG:4326 coords), return a list
        of all tile entries that would appear in a CUDEM datalist. If no source
        tiles overlap the polygon, return an empty list [].

        This is an overloaded function of etopo_source_dataset::generate_tile_datalist_entries.
        This one assigns a slightly different weight depending upon whether it's a BlueTopo_US5, _US4, or _US3 tile.
        Since the higher numbers (i.e. US5) are higher resolution, we want those to be slightly differnt, so rather than
        7, it'll be 7.3, 7.4, and 7.5, respectively.
        I will need to post-process to make sure the 7.x is changed back to 7 in the end.

        Each datalist entry is a 3-value string, as such:
        [path/filename] [format] [weight]
        In this case, format will always be 200 for raster. Weight will the weights
        returned by self.get_dataset_ranking_score().

        If polygon_crs can be a pyproj.crs.CRS object, or an ESPG number.
        If poygon_crs is None, we will use the CRS
        from the source dataset geopackage.

        If weight is None, use the weight returned by self.get_dataset_ranking_score.
        Else use the weight provided in "weight" for all entries.
        """
        if polygon_crs is None:
            polygon_crs = self.get_crs(as_epsg=False)

        if weight is None:
            weight = self.get_dataset_ranking_score()

        # Do a command-line "waffles -h" call to see datalist options. The datalist
        # type for raster files is "200"
        DTYPE_CODE = 200

        list_of_overlapping_files = self.retrieve_list_of_datafiles_within_polygon(polygon,
                                                                                   polygon_crs,
                                                                                   resolution_s = resolution_s,
                                                                                   verbose=verbose)

        return ["{0} {1} {2}".format(fname, DTYPE_CODE, get_tile_size_from_fname(fname, weight)) \
                for fname in list_of_overlapping_files]

    def reproject_tiles(self, suffix="_epsg4326", overwrite = False, verbose=True):
        """Project all the tiles into WGS84/latlon coordinates.

        The fucked-up NAD83 / UTM zone XXN is fucking with waffles. Converting them is the easiest answer now.
        After we do this, we can change the config.ini to look for the new "_epsg4326" tiles. Also, we can disable and
        delete the 5 BlueTopo_14N to _19N datasets, because they'll all be projected into the same coordinate system,
        which will be a lot easier.
        """
        # 1 arc-second is ~30.8 m at the equator.
        # The 0.3 tiles (level 3) are 16 m, 0.4 are 8 m, 0.5 are 4 m
        res_lookup_dict = {0.3: 16 / (30.8 * 60 * 60),
                           0.4: 8  / (30.8 * 60 * 60),
                           0.5: 4  / (30.8 * 60 * 60)}
        tilenames = self.retrieve_all_datafiles_list(verbose=verbose)
        for i,fname in enumerate(tilenames):
            dest_fname = os.path.splitext(fname)[0] + suffix + ".tif"
            if os.path.exists(dest_fname):
                if overwrite or gdal.Open(dest_fname, gdal.GA_ReadOnly) is None:
                    os.remove(dest_fname)
                else:
                    print("{0}/{1} {2} already written.".format(i+1, len(tilenames), os.path.split(dest_fname)[1]))
                    continue

            resolution = res_lookup_dict[get_tile_size_from_fname(fname, 0)]
            gdal_cmd = ["gdalwarp",
                        "-t_srs", "EPSG:4326",
                        "-dstnodata", "0.0",
                        "-tr", str(resolution), str(resolution),
                        "-r", "bilinear",
                        "-of", "GTiff",
                        "-co", "COMPRESS=DEFLATE",
                        "-co", "PREDICTOR=2",
                        "-co", "ZLEVEL=5",
                        fname, dest_fname]
            process = subprocess.run(gdal_cmd, capture_output = True, text=True)
            if verbose:
                print("{0}/{1} {2} ".format(i+1, len(tilenames), os.path.split(dest_fname)[1]), end="")
            if process.returncode == 0:
                if verbose:
                    print("written.")
            elif verbose:
                    print("FAILED")
                    print(" ".join(gdal_cmd), "\n")
                    print(process.stdout)

        return
def get_tile_size_from_fname(fname, orig_weight):
    """looking at the filename, get the file size from it (3,4,5), and add it as a decimal to the weight.

    The one-digit number immediately after _US in the filename is the tile size indicator.
    A filename of BlueTopo_US4TX1JG_20211020_egm2008.tiff with an original weight of 7 will return 7.4.

    This will put the bigger numbers (smaller tile sizes) at a greater weight than smaller numbers (larger tiles)."""
    # Look for the number just after "BlueTopo_US" and just before "LLNLL" wher L is a capital letter and N is a number.
    try:
        tile_size = int(re.search("(?<=BlueTopo_US)\d(?=[A-Z]{2}\w[A-Z]{2})", os.path.split(fname)[1]).group())
    except AttributeError as e:
        print("ERROR IN TILE:", fname)
        raise e
    assert 1 <= tile_size <= 5
    new_weight = orig_weight + (tile_size / 10.0)
    return new_weight

if "__main__" == __name__:
    # print(get_tile_size_from_fname("BlueTopo_US4LA1EN_20211018_egm2008.tiff", 0))
    bt = source_dataset_BlueTopo()
    gdf = bt.get_geodataframe()
    print(gdf.filename)
    # bt.reproject_tiles()
    # bt.split_files_into_epsg_folders()