# -*- coding: utf-8 -*-

"""Source code for the BlueTopo ETOPO source dataset class."""

import os
from osgeo import gdal
import numpy
import math
import re
import shutil
import pyproj
import subprocess
import pandas
import geopandas
import shapely.ops
import fiona
import xml.etree.ElementTree as ET

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
utm_zone_regex = r"(?<= / UTM zone )(\d{2})[NS]"
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
        self.tile_scheme_gdf = None


    def get_downloader_obj(self):
        if self.downloader is None:
            self.downloader = datasets.BlueTopo.download_BlueTopo_tiles.BlueTopo_Downloader()

        return self.downloader


    def get_epsg_from_tile_scheme(self, filename, utm_zone_name=False, verbose=True):
        """Get the UTM zone of the tile from the file name.

        Generally only used if the projection information is not contained within the file itself."""
        gdf = self.get_tile_scheme_gdf(verbose=verbose)
        tile_id = re.search(r"(?<=BlueTopo_)\w{8}(?=_\d{8}\.tiff\Z)", filename).group()
        utm_zone_list = gdf[gdf.tile == tile_id].UTM.tolist()
        assert len(utm_zone_list) == 1
        utm_zone_int = utm_zone_list[0]
        utm_zone_str = str(utm_zone_int) + "N"
        if utm_zone_name:
            return utm_zone_str
        else:
            return epsg_lookups[utm_zone_str]

    def get_tile_scheme_gdf(self, verbose=True):
        if self.tile_scheme_gdf is None:
            self.tile_scheme_gdf = self.get_downloader_obj().read_latest_tile_scheme_gpkg(verbose=verbose)

        return self.tile_scheme_gdf

    def get_file_epsg(self, filename, utm_zone_name=False, derive_from_filename_if_missing=True, verbose=True):
        """Given a BlueTopo tile, return the EPSG code for it.

        Some BlueTopo tiles are missing their projection. If that's the case, derive it from the file name if
        'derive_from_filename_if_missing' is True."""
        # If we're using a "converted" file but can't find it, look for the original.
        # if not os.path.exists(filename) and filename.find("/converted/") >= 1:
        #     # In some of these files, we tried (unsuccessfully) to convert it.
        #     filename = filename.replace("/converted/", "/").replace("_egm2008.tif", ".tif")
        # assert os.path.exists(filename)
        if not os.path.exists(filename):
            raise FileNotFoundError(filename, "not found.")
        ds = gdal.Open(filename, gdal.GA_ReadOnly)
        wkt = ds.GetProjection()
        if wkt == '':
            if derive_from_filename_if_missing:
                return self.get_epsg_from_tile_scheme(filename, utm_zone_name=utm_zone_name, verbose=verbose)
            else:
                return None
        try:
            utmz = re.search(utm_zone_regex, wkt).group()
        except AttributeError:
            # If we hit here, we aren't actally using a tile with a real UTM zone. Return None.
            return None
        if utm_zone_name:
            return utmz
        else:
            return epsg_lookups[utmz]

    # def split_files_into_epsg_folders(self, use_symlinks=True, verbose=True):
    #     """Since the BlueTopo is in five different UTM projections, I can't
    #     really handle it in one layer. Here, we copy the files (or create symlinks)
    #     into 5 different folders so that I can create 5 different datasets. Should do the job."""
    #     gdf = self.get_geodataframe(verbose=verbose)
    #     basedir = self.config.source_datafiles_directory
    #     for i,row in gdf.iterrows():
    #         fname = row.filename
    #
    #         # We want the original tiles, not converted ones.
    #         if fname.find("/converted/") >= 1:
    #             fname = fname.replace("/converted/", "/").replace("_egm2008.tif",".tif")
    #         assert os.path.exists(fname)
    #
    #         utm_zone = self.get_file_epsg(fname, utm_zone_name = True)
    #
    #         dest_dir = os.path.join(basedir, utm_zone)
    #         if not os.path.exists(dest_dir):
    #             os.mkdir(dest_dir)
    #         dest_fname = os.path.join(dest_dir, os.path.split(fname)[1])
    #         if os.path.exists(dest_fname):
    #             continue
    #         if use_symlinks:
    #             os.symlink(fname, dest_fname)
    #             if verbose:
    #                 print(dest_fname, "link created.")
    #         else:
    #             shutil.copyfile(fname, dest_fname)
    #             if verbose:
    #                 print(dest_fame, "created.")
    #     return

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

    # def get_crs(self, as_epsg=True):
    #     """Note: this only works if using one of the BlueTopo sub-classes with a UTM Zone on it, like 'BlueTopo_14N'."""
    #     utmz = self.dataset_name[-3:]
    #     if as_epsg:
    #         return epsg_lookups[utmz]
    #     else:
    #         return pyproj.crs.CRS.from_epsg(epsg_lookups[utmz])

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

        return ["{0} {1} {2}".format(fname, DTYPE_CODE, self.get_tile_size_from_fname(fname, weight)) \
                for fname in list_of_overlapping_files]

    def preprocess_tiles(self,
                         infile_regex = r"BlueTopo_\w{8}_\d{8}\.tiff\Z",
                         suffix_mask = "_masked",
                         suffix="_epsg4326",
                         suffix_vertical="_egm2008",
                         n_subprocs=15,
                         overwrite=False,
                         reset_geodataframe=True,
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

        mask_temp_fnames = [os.path.join(os.path.dirname(fname), "converted", os.path.splitext(os.path.basename(fname))[0] + suffix_mask + ".tif") for fname in input_tilenames]
        horz_trans_fnames = [(os.path.splitext(fname)[0] + suffix + ".tif") for fname in mask_temp_fnames]
        output_fnames = [os.path.splitext(dfn)[0] + suffix_vertical + ".tif" for dfn in horz_trans_fnames]

        self_args = [self] * len(input_tilenames)

        args = list(zip(self_args, input_tilenames, mask_temp_fnames, horz_trans_fnames, output_fnames))
        kwargs = {"overwrite": overwrite,
                  "verbose": (verbose and (n_subprocs == 1))}

        tempdirs = [os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "scratch_data", "temp{0:04d}".format(i))) for i in range(len(input_tilenames))]

        # FOR DEBUGGING. TODO: Delete later.
        # tag_dict = {} # A dictionary for keeping track of existing 3-digit tags. The tags for UTM zone appear to be the middle 3 letters,
        # # __25X___, look for those three.
        # for fname in input_tilenames:
        #     utm_zone_name = self.get_file_epsg(fname, utm_zone_name=True)
        #     # if utm_zone_name is None:
        #     #     tag_dict[tag3] = utm_zone_name
        #     tag3 = re.search(r"(?<=BlueTopo_B\w)\w{3}(?=\w{3}_\d{8}\.tiff\Z)", fname).group()
        #     if tag3 in tag_dict.keys() and tag_dict[tag3] == utm_zone_name:
        #         continue
        #     tag_dict[tag3] = utm_zone_name
        #     # print(tag3 + ":", utm_zone_name)
        #
        # for tag3 in sorted(list(tag_dict.keys())):
        #     print(tag3 + ":", tag_dict[tag3])
        #
        #     # print(os.path.basename(fname), self.get_file_epsg(fname, utm_zone_name=False), self.get_file_epsg(fname, utm_zone_name=True))
        # return
        ########

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

        if reset_geodataframe:
            self.reset_geopackage(verbose=verbose)

        return


    def transform_single_tile(self, input_name, mask_name_TEMP, horiz_transform_name_TEMP, output_name, overwrite=False, verbose=True):
        # 1 arc-second is ~30.8 m at the equator. This will slightly over-sample at higher latitudes but that's okay.
        # The 0.3 tiles (level 3) are 16 m, 0.4 are 8 m, 0.5 are 4 m
        res_lookup_dict = {0.03: 16 / (30.8 * 60 * 60),
                           0.04: 8 / (30.8 * 60 * 60),
                           0.05: 4 / (30.8 * 60 * 60),
                           0.06: 2 / (30.8 * 60 * 60)}

        # Some of these files aren't using a proper UTM zone (nor are they in NAVD88). Ignore them for now.
        if self.get_file_epsg(input_name) is None:
            return

        if os.path.exists(output_name):
            if overwrite or gdal.Open(output_name, gdal.GA_ReadOnly) is None:
                os.remove(output_name)
            else:
                return

        if os.path.exists(mask_name_TEMP):
            os.remove(mask_name_TEMP)

        # First, maks out bad (interpolated, tinned) values.
        self.remove_interpolated_values_from_tile(input_name, mask_name_TEMP, also_remove_gmrt=True, verbose=verbose)

        # Second, mask out single isolated values from the tiles.
        self.filter_out_single_points(filename=mask_name_TEMP, min_contiguous_points=3, verbose=verbose)

        if os.path.exists(horiz_transform_name_TEMP):
            os.remove(horiz_transform_name_TEMP)

        # First, transform horizontally.
        resolution = res_lookup_dict[self.get_tile_size_from_fname(input_name, 0)]
        gdal_cmd = ["gdalwarp",
                    "-s_srs", "EPSG:{0}".format(self.get_file_epsg(input_name, utm_zone_name=False, derive_from_filename_if_missing=True, verbose=False)),
                    "-t_srs", "EPSG:4326",
                    "-dstnodata", "0.0",
                    "-tr", str(resolution), str(resolution),
                    "-r", "bilinear",
                    "-of", "GTiff",
                    "-co", "COMPRESS=DEFLATE",
                    "-co", "PREDICTOR=2",
                    "-co", "ZLEVEL=5",
                    mask_name_TEMP, horiz_transform_name_TEMP]
        subprocess.run(gdal_cmd, capture_output = not verbose, text=True)

        if not os.path.exists(horiz_transform_name_TEMP):
            return

        # A few of the tiles do not have good horizontal datum information, and also use MSL heights rather than NAVD88.
        # In those cases, just copy the file over and don't bother trying to convert the vertical datum.
        ds = gdal.Open(input_name, gdal.GA_ReadOnly)
        projstr = ds.GetProjection()
        if projstr.lower().find("+ msl height") > -1 or (projstr == ""):
            shutil.move(horiz_transform_name_TEMP, output_name)

        else:
            assert projstr.lower().find("+ navd88") > -1
            # Then, transform that file vertically.
            convert_vdatum.convert_vdatum(horiz_transform_name_TEMP,
                                          output_name,
                                          input_vertical_datum="navd88",
                                          output_vertical_datum="egm2008",
                                          cwd=None,
                                          verbose=verbose)

        # Remove temporary tiles.
        if os.path.exists(mask_name_TEMP):
            os.remove(mask_name_TEMP)
        if os.path.exists(horiz_transform_name_TEMP):
            os.remove(horiz_transform_name_TEMP)

        return

    def remove_redundant_BlueTopo_tiles(self, reset_gpkg_at_end=True):
        """As new tiles get updated, some of them are just updated versions of old tiles, with the time tile_ID code.
        Look through all existing tiles... if they have the same tile_ID code, get rid of any older versions."""
        f_dir = self.config._abspath(self.config.source_datafiles_directory)
        f_regex = self.config.datafiles_regex_orig
        fnames = sorted([fn for fn in os.listdir(f_dir) if (re.search(f_regex, fn) is not None)])
        
        tile_id_dict = dict()
        for fn in fnames:
            tile_id = re.search(r'(?<=BlueTopo_B)\w{7}(?=_\d{8}\.tiff)', fn).group()
            datenum = int(re.search(r'(?<=BlueTopo_B\w{7}_)\d{8}(?=\.tiff)', fn).group())

            # If no tile has yet been seen with this tile_ID, add it to the dictionary with its current date.
            if tile_id not in tile_id_dict.keys():
                tile_id_dict[tile_id] = datenum
            # If another tile has been seen and this date is less than the existing one, then delete this tile.
            elif tile_id_dict[tile_id] > datenum:
                new_tile_name = "BlueTopo_B{0}_{1}.tiff".format(tile_id, tile_id_dict[tile_id])
                xml_name = os.path.join(f_dir, fn + ".aux.xml")
                print("Remove old tile", fn, ("(and xml) " if os.path.exists(xml_name) else "") + "for new", new_tile_name)
                os.remove(os.path.join(f_dir, fn))
                if os.path.exists(xml_name):
                    os.remove(xml_name)
            # If the new tile is newer than the existing one, get rid of the existing ID and update with the new one.
            elif tile_id_dict[tile_id] < datenum:
                old_tile_name = "BlueTopo_B{0}_{1}.tiff".format(tile_id, tile_id_dict[tile_id])
                old_xml_name = os.path.join(f_dir, old_tile_name + ".aux.xml")
                print("Remove old tile", old_tile_name, ("(and xml) " if os.path.exists(old_xml_name) else "") + "for new", fn)
                os.remove(os.path.join(f_dir, old_tile_name))
                if os.path.exists(old_xml_name):
                    os.remove(old_xml_name)
                # Then update the dictionary with the new tile.
                tile_id_dict[tile_id] = datenum
            else:
                raise Exception("SHOULDN'T GET HERE.")

        return

    @staticmethod
    def get_raster_table_from_bluetopo_xml(tilename,
                                           relative_search_dir=".",
                                           file_ext=".aux.xml",
                                           warn_if_nonexistent=True):
        """Read the .aux.xml file associated with each BlueTopo tile, and return the 'GDALRasterAttributeTable' object,
        specifiying all the source_ids in that BlueTopo tile, as a pandas dataframe."""

        # First, find the XML file. Should in the "relative_search_dir" directory, relative to the source tile.
        search_dir = os.path.abspath(os.path.join(os.path.dirname(tilename), relative_search_dir))
        # The XML file should be the exact same as the filename, just with ".aux.xml" at the end
        xml_file = os.path.join(search_dir, os.path.basename(tilename) + file_ext)
        if not os.path.exists(xml_file):
            if warn_if_nonexistent:
                raise UserWarning("XML file", xml_file, "does not exist. Returning None.")
            return None

        # Second, read the XML as an etree, and navigate to the 3rd band, second field in there, which gives us the raster attribute table for Band 3.
        root = ET.parse(xml_file).getroot()
        # The third band (#2) is the raster ID band.
        # The second element in that child (#1) is the GDALRasterAttributeTable we're looking for. It has a bunch of
        # FileldDefn tags, and Row tags. First loop through the FieldDefn tags.

        # Each FieldDefn has as "Type" specified as 0,1,2. Far as I can tell, those are int, float, str. Use these
        # to convert data accordingly.
        dtypes_dict = {'0': int,
                       '1': float,
                       '2': str}

        fieldnames = []
        datarows = []
        field_dtypes = []
        got_gdal_table = False
        for child1 in root[2]:
            if child1.tag != "GDALRasterAttributeTable":
                continue

            for child2 in child1:
                if child2.tag == "FieldDefn":
                    assert child2[0].tag == "Name"
                    fieldnames.append( child2[0].text )

                    assert child2[1].tag == "Type"
                    field_dtypes.append( dtypes_dict[child2[1].text] )

                elif child2.tag == "Row":
                    assert len(child2[:]) == len(fieldnames) == len(field_dtypes)
                    # Each row should have the same number of children as all the fieldnames we just read in.
                    datarows.append([dtype(subchild.text) for (subchild, dtype) in zip(child2[:], field_dtypes)])

                else:
                    raise ValueError("Uknown tag {0} in XML: {1}".format(child2.tag, os.path.basename(xml_file)))

            got_gdal_table = True

        if not got_gdal_table:
            raise ValueError("Did not find 'GDALRasterAttributeTable' in XML:", os.path.basename(xml_file))

        # Get the xml FieldDefn's to define the table columsn, and Row's to define the data. Put into a DataFrame.
        # The data is in rows. Put it into columns.
        data_dict = {}
        for i, colname in enumerate(fieldnames):
            data_dict[colname] = numpy.array([row[i] for row in datarows])

        # Return the dataframe.
        df = pandas.DataFrame(data_dict)
        # print(df)
        # print(df.columns)
        return df

    @staticmethod
    def remove_interpolated_values_from_tile(tilename_in,
                                             tilename_out,
                                             also_remove_gmrt=True,
                                             overwrite=True,
                                             verbose=True):
        """Remove all data in which the source dataset is "interpolated" from a BlueTopo tile. Save to tilename_out.

        tilename_in: An input (original) BlueTopo tile, with 3 bands: Elevation, Uncertainty, Contributor

        Remove all data values for which "Contributor" is 0. These are "interpolated" values and have a lot of shitty
        triangulated tinning in them.

        also_remove_gmrt: We are finding artifacts in BlueTopo data where it is drawing its own source data from GMRT.
        Since we already have our own GMRT data from the GMRT layer, we don't need this data anyway and can get rid of it.

        We can read the GMRT fields from the .aux.xml files provided with each BlueTopo tile. Use those to filter out
        any data that are associated with a GMRT source.
        """
        # Get rid of the existing destination tile if it already exists.
        if overwrite and (tilename_in != tilename_out) and os.path.exists(tilename_out):
            os.remove(tilename_out)
            # Also remove any gdal .aux.xml stuff from this file too.
            xml_file = os.path.splitext(tilename_out)[0] + ".aux.xml"
            if os.path.exists(xml_file):
                os.remove(xml_file)

        # First, just copy the Band 1 (Elevation) of the dataset into a new raster.
        gdal_cmd = ["gdal_translate",
                    "-b", "1",
                    "-co", "COMPRESS=DEFLATE",
                    "-co", "PREDICTOR=2",
                    "-co", "ZLEVEL=5",
                    tilename_in, tilename_out]
        subprocess.run(gdal_cmd, capture_output = not verbose, text=True)

        # Now, get the Contributor band of the input layer.
        ds_src = gdal.Open(tilename_in, gdal.GA_ReadOnly)
        b3_src = ds_src.GetRasterBand(3)
        # b3_ndv = b3_src.GetNoDataValue() # Do I need the NDV here?
        b3_array = b3_src.ReadAsArray()

        # Destroy the source objects
        b3_src = None
        ds_src = None

        # Create a mask for all zero values.
        b3_zero_mask = (b3_array == 0)

        if also_remove_gmrt:
            # Read the .aux.xml file associated with each input tile to find the source_id's present in this BlueTopo tile.
            bt_sid_table = source_dataset_BlueTopo.get_raster_table_from_bluetopo_xml(tilename_in)

            # Find all the source_ids where the "source_institution" contains "(GMRT)".
            sid_table_gmrt = bt_sid_table[bt_sid_table["source_institution"].str.contains("(GMRT)", regex=False, na=False)]
            gmrt_sids = sid_table_gmrt["value"].tolist()

            # Mask out the BlueTopo data points that have any of these numbers in them.
            for gmrt_sid_number in gmrt_sids:
                b3_zero_mask = b3_zero_mask | (b3_array == gmrt_sid_number)

        if verbose:
            print("{0:,} pixels filtered out in {1} ({2:0.1f}%)".format(numpy.count_nonzero(b3_zero_mask),
                                                                os.path.basename(tilename_in),
                                                                numpy.count_nonzero(b3_zero_mask) * 100. / b3_zero_mask.size))

        # If there *are* no interpolated values, then our job is done here. Move along.
        if numpy.count_nonzero(b3_zero_mask) == 0:
            return

        # Open the output dataset (which at this point is just a copy of the input), and mask out the unwanted data.
        ds_dst = gdal.Open(tilename_out, gdal.GA_Update)
        b1 = ds_dst.GetRasterBand(1)
        b1_array = b1.ReadAsArray()
        b1_ndv = b1.GetNoDataValue()

        # Sanity check on array sizes.
        assert b3_zero_mask.shape == b1_array.shape

        # Mask out the interpolated values, put NDVs there.
        b1_array[b3_zero_mask] = b1_ndv
        b1.WriteArray(b1_array)
        # Re-compute the statistics of the band
        b1.GetStatistics(0, 1)
        # Write the band back out to disk.
        b1.FlushCache()

        # Destroy the destination objects (save the file to disk).
        b1 = None
        ds_dst = None

        if verbose:
            print(os.path.basename(tilename_out), "written")

        return

    @staticmethod
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
            # BC = 16m, BF = 8m, BH = 4m, BJ = 2m
            # size numbers are 3,4,5, and 6, respectively.
            tile_size = {"C": 3, "F": 4, "H": 5, "J": 6}[re.search(r"(?<=BlueTopo_B)\w(?=\w{6}_)", os.path.split(fname)[1]).group()]
        except AttributeError as e:
            print("ERROR IN TILE:", fname)
            raise e
        assert 1 <= tile_size <= 6
        # Divide by 100 instead of 10, so that the weights don't get rounded "up" when
        # converting to source_ids for values >= 5.
        new_weight = orig_weight + (tile_size / 100.0)
        return new_weight

    def filter_out_single_points(self, filename=None, min_contiguous_points=3, verbose=True):
        """The BlueTopo tiles, after filtering out tinning interpolations, have a lot of "singular" points, such as lead-
        line measurements, that create artifacts when laid over other layers. Here, we will create masks of all points that are connected by
        no more than 'min_contiguous_points' points (default, 3 points or less) that are otherwise surrounded by empty space.

        We will save those masks as separate 'isolated_points_mask.tif' files, and use them to filter out isolated points from the existing tiles.
        (They can later be used to add them back in if I later desire.)

        Note, this is a simple algorithm that only look at the immediate 4-way neighborhood around it, and queries
        adjacent pixels to see how many *they're* connected to. It will likely provide wrong answers if min_contiguous_pixels is set
        to anything larger than 3. It will work for 3, 2, or 1 adjacent pixel. Since we are only using it for 1 or 2, this works.
        """
        if filename is None:
            tilenames = self.get_geodataframe().filename.to_list()
        # print(len(tilenames), "BlueTopo tiles.")

        else:
            tilenames = [filename]

        for i, tilename in enumerate(tilenames):
            if verbose:
                print(("{0}/{1} ".format(i + 1, len(tilenames)) if (len(tilenames) > 1) else "")
                      + os.path.basename(tilename), end=", ")
            ds = gdal.Open(tilename, gdal.GA_Update)
            band = ds.GetRasterBand(1)
            band_data = band.ReadAsArray()
            ndv = band.GetNoDataValue()
            data_valid_mask = (~numpy.isnan(band_data)) if math.isnan(ndv) else (band_data != ndv)

            if verbose:
                print("{0:0.2f}% valid data".format(numpy.count_nonzero(data_valid_mask) * 100 / band_data.size))
            # print("   ", data.shape, "ndv:", ndv)
            # print([d[1000:1020] for d in numpy.where(data_valid_mask)])

            # Get the name of the mask path
            if os.path.basename(tilename).find("_masked_epsg4326_egm2008") > -1:
                output_mask_path = os.path.join(os.path.dirname(tilename),
                                                os.path.basename(tilename).replace("_masked_epsg4326_egm2008",
                                                                                   "_isolated_pixel_mask"))
            else:
                base, ext = os.path.splitext(tilename)
                output_mask_path = os.path.join(base + "_isolated_pixel_mask" + ext)


            if os.path.exists(output_mask_path):
                if verbose:
                    print("   ", "Reading", os.path.basename(output_mask_path))
                mask_raster = gdal.Open(output_mask_path, gdal.GA_ReadOnly)
                mask_band = mask_raster.GetRasterBand(1)
                mask_isolated = mask_band.ReadAsArray()
                mask_raster = None

            # If the raster doesn't yet exist, create it.
            else:
                mask = numpy.zeros_like(band_data, dtype=numpy.uint8)

                # This masking does the same task as looping all the i,j pixels, but it's vectorized and a fuckload faster.
                # Add up valid pixels below (ignoring last row)
                mask[:-1, :] = mask[:-1, :] + data_valid_mask[1:, :]
                # Add up valid pixels above (ignoring first row)
                mask[1:, :] = mask[1:, :] + data_valid_mask[:-1, :]
                # Add up valid pixels to the left (ignoring first column)
                mask[:, 1:] = mask[:, 1:] + data_valid_mask[:, :-1]
                # Add up valid pixels to the right (ignoring last column)
                mask[:, :-1] = mask[:, :-1] + data_valid_mask[:, 1:]
                # Add in the pixels themselves (if they are valid) and zero out ndv pixels.
                mask = mask + data_valid_mask
                mask[data_valid_mask == 0] = 0

                # Okay, now the second step, query all neighboring pixels, and add in the number of *their* neighbors minus
                # 2 (to not count ourselves and said neighboring pixel). This will give valid results up to 3, but may over-count
                # for values >3, but we don't care, it'll still work.
                mask_2 = mask.copy()
                mask_2cap = numpy.clip(mask, 0, 2) # Capping it at 2 allows us to subract 2 for each mask cell that contained a 2 or greater.
                # Add up sum from valid pixels below (ignoring last row)
                mask_2[:-1, :] = mask_2[:-1, :] + mask[1:, :] - mask_2cap[1:, :]
                # Add up valid pixels above (ignoring first row)
                mask_2[1:, :] = mask_2[1:, :] + mask[:-1, :] - mask_2cap[:-1, :]
                # Add up valid pixels to the left (ignoring first column)
                mask_2[:, 1:] = mask_2[:, 1:] + mask[:, :-1] - mask_2cap[:, :-1]
                # Add up valid pixels to the right (ignoring last column)
                mask_2[:, :-1] = mask_2[:, :-1] + mask[:, 1:] - mask_2cap[:, 1:]
                # Mask out all nodata pixels
                mask_2[data_valid_mask == 0] = 0

                mask_isolated = (mask_2 <= min_contiguous_points) & (mask_2 > 0)

                # # Okay, this seems to work.
                # slice_y = slice(139, 139+12)
                # slice_x = slice(6200, 6200+12)
                # # Print a selection of this, to test.
                # print(data_valid_mask[slice_y, slice_x])
                # print(mask_2[slice_y, slice_x])
                # print(mask_isolated[slice_y, slice_x])

                # Create an output raster with the same geotransform and projection as the input raster
                driver = gdal.GetDriverByName("GTiff")
                mask_raster = driver.Create(output_mask_path, ds.RasterXSize, ds.RasterYSize, 1,
                                              gdal.GDT_Byte, options=["COMPRESS=LZW"])
                mask_raster.SetGeoTransform(ds.GetGeoTransform())
                mask_raster.SetProjection(ds.GetProjection())

                # Write the mask array to the output raster band
                output_band = mask_raster.GetRasterBand(1)
                output_band.WriteArray(mask_isolated)

                # Set nodata value for the output raster band
                output_band.SetNoDataValue(0)
                output_band.GetStatistics(0, 1)

                # Close the input and output rasters
                mask_raster = None

                if verbose:
                    print("   ", os.path.basename(output_mask_path), "created.")

            if verbose:
                print("    {0:,} isolated pixels.".format(numpy.count_nonzero(mask_isolated)))

            # Now, filter out isolated pixels.
            if numpy.count_nonzero(band_data.flatten()[mask_isolated.flatten()] != ndv) == 0:
                if verbose:
                    print("    Isolated cells have already been filtered out. No changes made.")
            else:
                band_data[mask_isolated] = ndv
                band.WriteArray(band_data)
                band.GetStatistics(0, 1)
                if verbose:
                    print("   ", os.path.basename(tilename), "written back to disk.")

            band = None
            ds = None

            # break

        return

    def identify_bt_tiffs_without_xmls(self,
                                       infile_regex = r"BlueTopo_\w{8}_\d{8}\.tiff\Z",):
        """Some of the BlueTopo tiffs do NOT have associated XMLs. Find them and list them out."""
        input_tilenames = utils.traverse_directory.list_files(
            self.config._abspath(self.config.source_datafiles_directory),
            regex_match=infile_regex)

        xml_names = [fname + ".aux.xml" for fname in input_tilenames]
        for fname, xmlname in zip(input_tilenames, xml_names):
            if os.path.exists(xmlname):
                print(os.path.basename(fname), ".aux.xml exists.")
            else:
                print(os.path.basename(fname), os.path.basename(xmlname), "DOESN'T EXIST.")

    def remove_bad_polygons(self, verbose=True):
        """Some of the bad values aren't great to mask out with boxes, polygons work better.
        Read the bad_polygons directory, find all .gpkg files there, read the polygons within.
        For each polygon, find which BlueTopo tiles it intersects. Rasterize the polygon to the extent of
        those tiles, and put in the "rasterized" subdir. Then, mask out those rasters for each corresponding BlueTopo
        tile."""
        bad_polygons_dir = self.config._abspath(self.config.bluetopo_bad_polygons_dir)

        assert os.path.exists(bad_polygons_dir)
        bad_polygons_list_of_gpkgs = [os.path.join(bad_polygons_dir, fn) for fn in os.listdir(bad_polygons_dir)
                                      if (os.path.splitext(fn)[1].lower() == ".gpkg")]
        geopkg_obj = self.get_geopkg_object()
        gdf = self.get_geodataframe()

        tilenames = gdf.filename.to_list()

        rasterized_poly_dir = self.config._abspath(self.config.bluetopo_bad_polygons_rasterized_dir)
        # Clear out the rasterized poly dir completely.
        files_in_raster_dir = os.listdir(rasterized_poly_dir)

        if len(files_in_raster_dir) > 0:
            if verbose:
                print("Removing", len(files_in_raster_dir), "files from \"rasterized\" dir.")
            for fn in files_in_raster_dir:
                os.remove(os.path.join(rasterized_poly_dir, fn))

        rasters_created = []

        for poly_gpkg in bad_polygons_list_of_gpkgs:
            poly_gdf = geopandas.read_file(poly_gpkg)
            layer_names = fiona.listlayers(poly_gpkg)
            assert len(layer_names) == 1
            layername = layer_names[0]

            # Merge all the polygons in this file into one multi-polygon.
            united_polygon = shapely.ops.unary_union(poly_gdf.geometry.tolist())
            intersecting_tilenames = geopkg_obj.subset_by_polygon(united_polygon, poly_gdf.crs, verbose=verbose)

            if verbose:
                print(os.path.basename(poly_gpkg), "read with", len(poly_gdf), "features, intersecting", len(intersecting_tilenames), "tiles.")

            for itilename in intersecting_tilenames.filename.tolist():
                if verbose:
                    print("\t", os.path.basename(itilename))

                # For each tile this polygon interacts with, rasterize it to the extent of that tile.
                poly_raster_name = os.path.join(rasterized_poly_dir, os.path.splitext(os.path.basename(itilename))[0] + "_POLY_0.tif")
                # Since a tilename could overlap more than one polygon, make sure they have unique names.
                # Start with _POLY_0.tif, iterate to _POLY_1.tif, etc, until you fine one that's unclaimed.
                poly_raster_i = 0
                while os.path.exists(poly_raster_name):
                    poly_raster_i += 1
                    poly_raster_name = poly_raster_name.replace("_POLY_{0}.tif".format(poly_raster_i - 1),
                                                                "_POLY_{0}.tif".format(poly_raster_i))

                # Get necessary info from the BlueTopo tile.
                bt_tile_ds = gdal.Open(itilename, gdal.GA_ReadOnly)
                bt_width, bt_height = bt_tile_ds.RasterXSize, bt_tile_ds.RasterYSize
                bt_xmin, bt_xres, _, bt_ymax, _, bt_yres = bt_tile_ds.GetGeoTransform()
                bt_xmax = bt_xmin + (bt_xres * bt_width)
                bt_ymin = bt_ymax + (bt_yres * bt_height)
                assert (bt_xmax > bt_xmin) & (bt_ymax > bt_ymin)
                bt_proj = "EPSG:{0}".format(pyproj.CRS.from_string((bt_tile_ds.GetProjection())).to_epsg())

                print("Will create", poly_raster_name)

                gdal_rasterize_cmd = ["gdal_rasterize",
                                      "-l", layername,
                                      "-burn", "1",
                                      "-of", "GTiff",
                                      "-a_srs", bt_proj,
                                      "-a_nodata", "0",
                                      "-init", "0",
                                      "-co", "COMPRESS=LZW",
                                      "-co", "PREDICTOR=2",
                                      "-te", repr(bt_xmin), repr(bt_ymin), repr(bt_xmax), repr(bt_ymax),
                                      "-tr", repr(bt_xres), repr(abs(bt_yres)),
                                      "-ot", "Byte",
                                      poly_gpkg, poly_raster_name]

                if verbose:
                    print(" ".join(gdal_rasterize_cmd))
                subprocess.run(gdal_rasterize_cmd, capture_output=not verbose)

                if os.path.exists(poly_raster_name):
                    rasters_created.append(poly_raster_name)
                else:
                    raise UserWarning("WARNING: Raster", os.path.basename(poly_raster_name), "not created.")

                bt_tile_ds = None

        # Now, loop through and mask out the polygonized rasters from the BlueTopo tiles.
        for i, mask_raster in enumerate(rasters_created):
            # Extract the TILEID_YYYYMMDD combo from the mask raster. This will match it up with the appropriate tile.
            tile_id_str = re.search(r"BlueTopo_[A-Z0-9]{8}_\d{8}", mask_raster).group()
            gdf_subset = gdf.loc[gdf.filename.str.contains(tile_id_str, case=True)]
            assert len(gdf_subset) == 1
            tile_fname = gdf_subset.filename.tolist()[0]
            if verbose:
                print("{0}/{1}".format(i+1, len(rasters_created)), os.path.basename(mask_raster))

            tile_ds = gdal.Open(tile_fname, gdal.GA_Update)
            tile_band = tile_ds.GetRasterBand(1)
            tile_array = tile_band.ReadAsArray()
            tile_ndv = tile_band.GetNoDataValue()

            mask_ds = gdal.Open(mask_raster, gdal.GA_ReadOnly)
            mask_array = mask_ds.GetRasterBand(1).ReadAsArray().astype(bool)

            # Check to see if there are any valid data values in that mask (this will be False if it's already
            # been done, just move along).
            masked_values = tile_array[mask_array]
            if numpy.all(masked_values == tile_ndv):
                if verbose:
                    print("    No valid data in tile to mask out within polygon. Moving on.")
                continue

            # Mask out the values from the mask, with NDV
            tile_array[mask_array] = tile_ndv
            tile_band.WriteArray(tile_array)
            tile_ds.FlushCache()
            # Recopmute statistics.
            tile_band.GetStatistics(0, 1)

            # Save the output back to disk (again), after stats have been written.
            tile_ds.FlushCache()
            tile_band = None
            tile_ds = None

            if verbose:
                print("    Masked out {0:,} cells from within polygon.".format(numpy.count_nonzero(masked_values != tile_ndv)))

        return

        # OLD CODE FOR GMRT - TODO: DELETE WHEN DONE.
        for mask_out_tif in bad_polygons_list_of_tifs:
            # Get the location ID from the mask filename.
            try:
                tile_id = re.search(r"[NSns](\d{2})[EWew](\d{3})", mask_out_tif).group().upper()
            except AttributeError:
                print("Could not retrieve tile_id from mask file", os.path.basename(mask_out_tif), file=sys.stderr)
                continue

            # Find the right tile that goes with this.
            matching_tilenames = [fn for fn in tilenames if re.search(tile_id, os.path.basename(fn)) != None]
            # There should be only 1 tile with that ID
            if len(matching_tilenames) != 1:
                print("   ", len(matching_tilenames), "tiles matched with Tile ID", tile_id + ":",
                      [os.path.basename(fn) for fn in matching_tilenames])
                print("    We're like The Highlander here: There can be only one. Moving on.")
                continue

            tilename = matching_tilenames[0]

            print(os.path.basename(mask_out_tif))
            # Get the mask array from the geotiff.
            mask_ds = gdal.Open(mask_out_tif, gdal.GA_ReadOnly)
            mask_array = mask_ds.GetRasterBand(1).ReadAsArray().astype(bool)

            if not numpy.any(mask_array):
                print("    No valid data in mask_array. Moving on.")
                continue
            mask_ds = None

            # Read the data from the tile, in update mode.
            tile_ds = gdal.Open(tilename, gdal.GA_Update)
            tile_band = tile_ds.GetRasterBand(1)
            tile_array = tile_band.ReadAsArray()
            tile_ndv = tile_band.GetNoDataValue()

            # Check to see if there are any valid data values in that mask (this will be False if it's already
            # been done, just move along).
            masked_values = tile_array[mask_array]
            if numpy.all(masked_values == tile_ndv):
                print("    No valid data in tile to mask out within polygon. Moving on.")
                continue

            # Mask out the values from the mask, with NDV
            tile_array[mask_array] = tile_ndv
            tile_band.WriteArray(tile_array)
            tile_ds.FlushCache()
            # Recopmute statistics.
            tile_band.GetStatistics(0, 1)

            # Save the output back to disk (again), after stats have been written.
            tile_ds.FlushCache()
            tile_band = None
            tile_ds = None

            print("    Masked out", numpy.count_nonzero(masked_values != tile_ndv), "cells from within polygon.")


# def get_updated_BlueTopo_tiles():
#     """The BlueTopo tiles are regularly updated in their Amazon schema. Go get the latest GPKG download to the Tile_Scheme directory.
#
#     Search through then new Scheme, grab any new tiles that we don't already have downloaded. Download those tiles.
#     Convert the tiles in to WGS84/EPSG2008 horiz/vert datums.
#     Re-build the BlueTopo geopackage using all the tiles (including the new ones).
#     """
#     # 1. Look on the website, get the name of the latest Tile_Scheme GPKG.
#     # 2. Compare to the current GPKG in there. If the same, don't download. If different, download the new one.
#     # 3. Parse through the GPKG, look for all tile URLs in there. If we don't yet have the tile locally, add it to the list.
#     # 4. Use the dataset_downloader class to download all the new tiles we need.
#     # 5. For all new tiles, create symlinks in the "14N" thru "19N" folders depending on their horizontal projections.
#     # 6. Convert the tiles from NAVD88 --> EGM2008 vertical projection.
#     # 7. Convert the tiles from UTM --> WGS84 horizontal projection.


if "__main__" == __name__:
    # print(bt.get_tile_size_from_fname("BlueTopo_US4LA1EN_20211018_egm2008.tiff", 0))
    bt = source_dataset_BlueTopo()
    bt.reset_geopackage()
    # bt.remove_bad_polygons()
    # bt.get_geodataframe()
    # bt.identify_bt_tiffs_without_xmls()

    # bt.remove_redundant_BlueTopo_tiles()

    # bt.preprocess_tiles(n_subprocs=18, reset_geodataframe=True)

    ########################
    # TESTING WITH ONE TILE.
    # bt.remove_interpolated_values_from_tile("/home/mmacferrin/Research/DATA/DEMs/BlueTopo/data/BlueTopo/BlueTopo_BC26526R_20221122.tiff",
    #                                         "/home/mmacferrin/Research/DATA/DEMs/BlueTopo/data/BlueTopo/converted/BlueTopo_BC26526R_20221122_masked_TEST5.tif",
    #                                         also_remove_gmrt=True, verbose=True)
    #
    # shutil.copy("/home/mmacferrin/Research/DATA/DEMs/BlueTopo/data/BlueTopo/converted/BlueTopo_BC26526R_20221122_masked_TEST5.tif",
    #             "/home/mmacferrin/Research/DATA/DEMs/BlueTopo/data/BlueTopo/converted/BlueTopo_BC26526R_20221122_masked_TEST6.tif")
    #
    # # Second, mask out single isolated values from the tiles.
    # bt.filter_out_single_points(filename="/home/mmacferrin/Research/DATA/DEMs/BlueTopo/data/BlueTopo/converted/BlueTopo_BC26526R_20221122_masked_TEST6.tif",
    #                             min_contiguous_points=3, verbose=True)
    ########################

    # bt_sid_table = bt.get_raster_table_from_bluetopo_xml('/home/mmacferrin/Research/DATA/DEMs/BlueTopo/data/BlueTopo/BlueTopo_BC25L26L_20221130.tiff')
    # sid_table_gmrt = bt_sid_table[bt_sid_table["source_institution"].str.contains("(GMRT)", regex=False, na=False)]
    # print(sid_table_gmrt)

    # bt.filter_out_single_points()



    # bt.remove_interpolated_values_from_tile("/home/mmacferrin/Research/DATA/DEMs/BlueTopo/data/BlueTopo/BlueTopo_BH4PQ58H_20230327.tiff",
    #                                         "/home/mmacferrin/Research/DATA/DEMs/BlueTopo/data/BlueTopo/converted/BlueTopo_BH4PQ58H_20230327_masked.tif",
    #                                         verbose=True)

    # bt.transform_single_tile("/home/mmacferrin/Research/DATA/DEMs/BlueTopo/data/BlueTopo/BlueTopo_BH4PQ58H_20230327.tiff",
    #                          "/home/mmacferrin/Research/DATA/DEMs/BlueTopo/data/BlueTopo/converted/BlueTopo_BH4PQ58H_20230327_masked.tif",
    #                          "//home/mmacferrin/Research/DATA/DEMs/BlueTopo/data/BlueTopo/converted/BlueTopo_BH4PQ58H_20230327_masked_epsg4326.tif",
    #                       "/home/mmacferrin/Research/DATA/DEMs/BlueTopo/data/BlueTopo/converted/BlueTopo_BH4PQ58H_20230327_masked_epsg4326_egm2008.tif",
    #                       overwrite=True)
    #
    # foobar
    # print(
    # bt.get_file_epsg("/home/mmacferrin/Research/DATA/DEMs/BlueTopo/data/BlueTopo/BlueTopo_BC26H26C_20220926.tiff")
    # )
    # foobar

    # gdf = bt.get_geodataframe()
    # for fname in gdf.filename:
    #     print(fname)
    #
    # print(bt.list_utm_zones())
    # print(bt.list_utm_zones(as_epsg=True))
    # bt.reproject_tiles()
    # bt.split_files_into_epsg_folders()