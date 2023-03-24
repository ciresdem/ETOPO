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
import datasets.BlueTopo.download_BlueTopo_tiles
import etopo.convert_vdatum as convert_vdatum
import utils.traverse_directory
import utils.parallel_funcs
# import utils.progress_bar

# Look for the 3-letter string prefixed by "NAD83 / UTM zone" and consisting of 2 digit characters (\d{2}) following by either "N" or "S" ([NS])
utm_zone_regex = r"(?<=NAD83 / UTM zone )(\d{2})[NS]"
# These cover all the UTM zones covered by the BlueTopo tiles.
epsg_lookups  = {"10N" : 26910,
                 "11N" : 26911,
                 "12N" : 26912,
                 "13N" : 26913,
                 "14N" : 26914,
                 "15N" : 26915,
                 "16N" : 26916,
                 "17N" : 26917,
                 "18N" : 26918,
                 "19N" : 26919,
                 "20N" : 26920,
                 "21N" : 26921
                 }

# Suppress gdal warnings in this file.
gdal.SetConfigOption('CPL_LOG', "/dev/null")

class source_dataset_BlueTopo(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "BlueTopo_config.ini" )):
        """Initialize the BlueTopo source dataset object."""

        super(source_dataset_BlueTopo, self).__init__("BlueTopo", configfile)

        self.downloader = None


    def get_downloader_obj(self):
        if self.downloader is None:
            self.downloader = download_BlueTopo_tiles.BlueTopo_Downloader()

        return self.downloader


    def get_file_epsg(self, filename, utm_zone_name=False):
        """Given a BlueTopo tile, return the EPSG code for it."""
        # If we're using a "converted" file but can't find it, look for the original.
        if not os.path.exists(filename) and filename.find("/converted/") >= 1:
            # In some of these files, we tried (unsuccessfully) to convert it.
            filename = filename.replace("/converted/", "/").replace("_egm2008.tif", ".tif")
        assert os.path.exists(filename)
        ds = gdal.Open(filename, gdal.GA_ReadOnly)
        wkt = ds.GetProjection()
        try:
            utmz = re.search(utm_zone_regex, wkt).group()
        except:
            # If we hit here, we aren't actally using a tile with a real UTM zone. Return None.
            return None
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

    def reproject_tiles(self,
                        infile_regex = r"BlueTopo_\w{8}_\d{8}\.tiff",
                        suffix="_epsg4326",
                        suffix_vertical="_egm2008",
                        n_subprocs = 15,
                        overwrite = False,
                        verbose=True):
        """Project all the tiles into WGS84/latlon coordinates.

        The fucked-up NAD83 / UTM zone XXN is fucking with waffles. Converting them is the easiest answer now.
        After we do this, we can change the config.ini to look for the new "_epsg4326" tiles. Also, we can disable and
        delete the 5 BlueTopo_14N to _19N datasets, because they'll all be projected into the same coordinate system,
        which will be a lot easier.
        """

        # tilenames = self.retrieve_all_datafiles_list(verbose=verbose)
        input_tilenames = utils.traverse_directory.list_files(self.config._abspath(self.config.source_datafiles_directory),
                                                              regex_match = infile_regex)
        if verbose:
            print(len(input_tilenames), "input BlueTopo tiles.")

        horz_trans_fnames = [os.path.join(os.path.dirname(fname), "converted", os.path.splitext(os.path.basename(fname))[0] + suffix + ".tif") for fname in input_tilenames]
        output_fnames = [os.path.splitext(dfn)[0] + suffix_vertical + ".tif" for dfn in horz_trans_fnames]

        self_args = [self] * len(input_tilenames)

        args = list(zip(self_args, input_tilenames, horz_trans_fnames, output_fnames))
        kwargs = {"overwrite": overwrite,
                  "verbose": False}

        tempdirs = [os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "scratch_data", "temp{0:04d}".format(i))) for i in range(len(input_tilenames))]

        utils.parallel_funcs.process_parallel(source_dataset_BlueTopo.transform_single_tile,
                                              args,
                                              kwargs_list = kwargs,
                                              outfiles = output_fnames,
                                              temp_working_dirs = tempdirs,
                                              max_nprocs=n_subprocs
                                              )

        # for i, fname in enumerate(input_tilenames):
        #     dest_fname = os.path.join(os.path.dirname(fname), "converted", os.path.splitext(os.path.basename(fname))[0] + suffix + ".tif")
        #     # Now, do the vertical reprojection too.
        #     dest_fname_2 = os.path.splitext(dest_fname)[0] + suffix_vertical + ".tif"
        #
        #
        #     if verbose:
        #         print("{0}/{1} {2} ".format(i+1, len(input_tilenames), os.path.split(dest_fname_2)[1]), end="")
        #     if process.returncode == 0 and os.path.exists(dest_fname_2):
        #         if verbose:
        #             print("written.")
        #     elif verbose:
        #             print("FAILED")
        #             print(" ".join(gdal_cmd), "\n")
        #             print(process.stdout)

        return


    def transform_single_tile(self, input_name, horiz_transform_name_TEMP, output_name, overwrite=False, verbose=True):
        # 1 arc-second is ~30.8 m at the equator.
        # The 0.3 tiles (level 3) are 16 m, 0.4 are 8 m, 0.5 are 4 m
        res_lookup_dict = {0.3: 16 / (30.8 * 60 * 60),
                           0.4: 8 / (30.8 * 60 * 60),
                           0.5: 4 / (30.8 * 60 * 60),
                           0.6: 2 / (30.8 * 60 * 60)}

        # Some of these files aren't using a proper UTM zone (nor are they in NAVD88). Ignore them for now.
        if self.get_file_epsg(input_name) is None:
            return

        if os.path.exists(output_name):
            if overwrite or gdal.Open(output_name, gdal.GA_ReadOnly) is None:
                os.remove(output_name)
            else:
                return

        # First, transform horizontally.
        resolution = res_lookup_dict[get_tile_size_from_fname(input_name, 0)]
        gdal_cmd = ["gdalwarp",
                    "-t_srs", "EPSG:4326",
                    "-dstnodata", "0.0",
                    "-tr", str(resolution), str(resolution),
                    "-r", "bilinear",
                    "-of", "GTiff",
                    "-co", "COMPRESS=DEFLATE",
                    "-co", "PREDICTOR=2",
                    "-co", "ZLEVEL=5",
                    input_name, horiz_transform_name_TEMP]
        subprocess.run(gdal_cmd, capture_output = not verbose, text=True)

        if not os.path.exists(horiz_transform_name_TEMP):
            return

        # Then, transform that file vertically.
        convert_vdatum.convert_vdatum(horiz_transform_name_TEMP,
                                      output_name,
                                      input_vertical_datum="navd88",
                                      output_vertical_datum="egm2008",
                                      cwd=None,
                                      verbose=verbose)

        if os.path.exists(horiz_transform_name_TEMP):
            os.remove(horiz_transform_name_TEMP)

        return

def get_tile_size_from_fname(fname, orig_weight):
    """looking at the filename, get the file size from it (3,4,5), and add it as a decimal to the weight.

    The one-digit number immediately after _US in the filename is the tile size indicator.
    A filename of BlueTopo_US4TX1JG_20211020_egm2008.tiff with an original weight of 7 will return 7.4.

    This will put the bigger numbers (smaller tile sizes) at a greater weight than smaller numbers (larger tiles)."""
    # Look for the number just after "BlueTopo_US" and just before "LLNLL" wher L is a capital letter and N is a number.
    try:
        # This was the naming convention used in the previous version (summer 2022). It has since been replaced by a new naming convention.
        # tile_size = int(re.search("(?<=BlueTopo_US)\d(?=[A-Z]{2}\w[A-Z]{2})", os.path.split(fname)[1]).group())
        # This uses the new naming convetion (spring 2023)
        # BC = 16m, BF = 8m, BH = 4m
        # size numbers are 3,4, and 5, respectively.
        tile_size = {"C": 3, "F": 4, "H": 5, "J": 6}[re.search(r"(?<=BlueTopo_B)\w(?=\w{6}_)", os.path.split(fname)[1]).group()]
    except AttributeError as e:
        print("ERROR IN TILE:", fname)
        raise e
    assert 1 <= tile_size <= 6
    new_weight = orig_weight + (tile_size / 10.0)
    return new_weight

def get_updated_BlueTopo_tiles():
    """The BlueTopo tiles are regularly updated in their Amazon schema. Go get the latest GPKG download to the Tile_Scheme directory.

    Search through then new Scheme, grab any new tiles that we don't already have downloaded. Download those tiles.
    Convert the tiles in to WGS84/EPSG2008 horiz/vert datums.
    Re-build the BlueTopo geopackage using all the tiles (including the new ones).
    """
    # 1. Look on the website, get the name of the latest Tile_Scheme GPKG.
    # 2. Compare to the current GPKG in there. If the same, don't download. If different, download the new one.
    # 3. Parse through the GPKG, look for all tile URLs in there. If we don't yet have the tile locally, add it to the list.
    # 4. Use the dataset_downloader class to download all the new tiles we need.
    # 5. For all new tiles, create symlinks in the "14N" thru "19N" folders depending on their horizontal projections.
    # 6. Convert the tiles from NAVD88 --> EGM2008 vertical projection.
    # 7. Convert the tiles from UTM --> WGS84 horizontal projection.


if "__main__" == __name__:
    # print(get_tile_size_from_fname("BlueTopo_US4LA1EN_20211018_egm2008.tiff", 0))
    bt = source_dataset_BlueTopo()
    # bt.transform_single_tile("/home/mmacferrin/Research/DATA/DEMs/BlueTopo/data/BlueTopo/BlueTopo_BC26H26C_20220926.tiff",
    #                       "//home/mmacferrin/Research/DATA/DEMs/BlueTopo/data/BlueTopo/converted/BlueTopo_BC26H26C_20220926_epsg4326.tif",
    #                       "/home/mmacferrin/Research/DATA/DEMs/BlueTopo/data/BlueTopo/converted/BlueTopo_BC26H26C_20220926_epsg4326_egm2008.tif",
    #                       overwrite=True)
    #
    # foobar
    # print(
    # bt.get_file_epsg("/home/mmacferrin/Research/DATA/DEMs/BlueTopo/data/BlueTopo/BlueTopo_BC26H26C_20220926.tiff")
    # )
    # foobar

    # bt.reproject_tiles()

    gdf = bt.get_geodataframe()
    # for fname in gdf.filename:
    #     print(fname)
    #
    # print(bt.list_utm_zones())
    # print(bt.list_utm_zones(as_epsg=True))
    # bt.reproject_tiles()
    # bt.split_files_into_epsg_folders()