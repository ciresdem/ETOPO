# -*- coding: utf-8 -*-

"""Source code for the USGS_TNM ETOPO source dataset class."""

import os
import re
import numpy
import subprocess
from osgeo import gdal
import pandas
import geopandas
import shutil
import time

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset
import utils.traverse_directory
import etopo.convert_vdatum

class source_dataset_USGS_TNM(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "USGS_TNM_config.ini" )):
        """Initialize the USGS_TNM source dataset object."""

        super(source_dataset_USGS_TNM, self).__init__("USGS_TNM", configfile)

        # These are USGS tiles that sit within the bounds of the CRMs, but are solely off-shore (or only contain small islands)
        # but are outside the bounds of CUDEM tiles which would presumably supercede them. We are eliminating these tiles
        # in order to get rid of the bathy artifacts  but still have the land tiles sit in the correct order over the
        # bathy tiles.
        self.tiles_to_remove = ["n26w098",
                                "n28w097",
                                "n29w096",
                                "n30w089",
                                "n30w086",
                                "n30w085",
                                "n27w083",
                                "n26w082",
                                "n25w083",
                                "n25w082",
                                "n25w081",
                                "n30w081",
                                "n32w081",
                                "n33w080",
                                "n34w079",
                                "n34w078",
                                "n39w075",
                                "n44w069",
                                "n45w067",
                                ]

    def eliminate_duplicates_in_links_file(self, links_file_orig="/home/mmacferrin/Research/DATA/DEMs/USGS_TNM/data/tnm_links.txt", output_suffix="_sorted"):
        f = open(links_file_orig, 'r')
        lines = [line.strip() for line in f.readlines()]
        f.close()

        print(len(lines), "input links.")

        lines_sorted = sorted(list(set(lines)))

        print(len(lines_sorted), "unique output links.")
        output_txt = "\n".join(lines_sorted)
        base, ext = os.path.splitext(links_file_orig)
        fname_out = base + output_suffix + ext
        f_out = open(fname_out, 'w')
        f_out.write(output_txt)
        f_out.close()
        print(fname_out, "written.")


    def generate_tile_datalist_entries(self, polygon, polygon_crs=None, resolution_s = None, verbose=True, weight=None):
        """Given a polygon (ipmortant, in WGS84/EPSG:4326 coords), return a list
        of all tile entries that would appear in a CUDEM datalist. If no source
        tiles overlap the polygon, return an empty list [].

        This is an overloaded function of etopo_source_dataset::generate_tile_datalist_entries.
        This one assigns a slightly different weight depending upon whether it's a 1s or 1_3s USGS Tile. 1_3 takes precedent.

        Each datalist entry is a 3-value string, as such:
        [path/filename] [format] [weight]
        In this case, format will always be 200 for raster. Weight will the weights
        returned by self.get_dataset_ranking_score().

        If polygon_crs can be a pyproj.crs.CRS object, or an ESPG number.
        If poygon_crs is None, we will use the CRS from the source dataset geopackage.

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


    @staticmethod
    def get_tile_size_from_fname(fname, orig_weight):
        """looking at the filename, get the file size from it (1 or 1/3), and add it as a decimal to the weight,
        as 0.1 or 0.13. This will put the 1/3 arc-second tiles slightly higher-ranked than the 1-second tiles.

        This will put the bigger numbers (smaller tile sizes) at a greater weight than smaller numbers (larger tiles)."""
        # Look for the number just after "BlueTopo_US" and just before "LLNLL" wher L is a capital letter and N is a number.
        try:
            # This was the naming convention used in the previous version (summer 2022). It has since been replaced by a new naming convention.
            # tile_size = int(re.search("(?<=BlueTopo_US)\d(?=[A-Z]{2}\w[A-Z]{2})", os.path.split(fname)[1]).group())
            # This uses the new naming convetion (spring 2023)
            # BC = 16m, BF = 8m, BH = 4m
            # size numbers are 3,4, and 5, respectively.
            tile_size = {"1": 0.1, "13": 0.13}[re.search(r"(?<=USGS_)\d{1,2}(?=_(\w{7})_(\d{8}))", os.path.basename(fname)).group()]
        except AttributeError as e:
            print("ERROR IN TILE:", fname)
            raise e
        assert 0.1 <= tile_size <= 0.13
        new_weight = orig_weight + tile_size
        return new_weight


    def omit_all_but_latest_tiles(self, omitted_subdir = "omitted"):
        """Remove all the but the latest of each geographically unique tile.

        The USGS dataset contains many duplicate tiles at 1 and 1/3 arc-second that were generated at different dates.
        The tiles all have a format of 'USGS_1[3]_nYYwXXX_DDDDDDDD.tif', with DDDDDDDD being a YYYYMMDD date string.
        For each unique nYYwXXX tile, find and keep the tile that has the latest DDDDDDDD string. Delete the rest.
        If both a 1s and 1/3s tile exist, just keep the 1/3s."""

        dirname = self.config._abspath(self.config.source_datafiles_directory)
        dirname_omitted = os.path.join(dirname, omitted_subdir)
        regex_str = self.config.datafiles_regex
        fnames = utils.traverse_directory.list_files(dirname, regex_str, depth=0)
        loc_tags = sorted(list(set([re.search(r"[ns]\d{2}[ew]\d{3}", os.path.basename(fn)).group() for fn in fnames])))

        files_removed = 0
        files_kept = 0
        for loc_tag in loc_tags:
            matching_fnames = [fn for fn in fnames if re.search(loc_tag, os.path.basename(fn)) is not None]
            # print(loc_tag, len(matching_fnames))
            assert len(matching_fnames) >= 1

            # If we've flagged to get rid of this tile ID, then get rid of all matching files.
            if loc_tag in self.tiles_to_remove:
                for fname in matching_fnames:
                    print("Omitting", os.path.basename(fname), "to get rid of offshore errors.")
                    # os.remove(fname)
                    shutil.move(fname, dirname_omitted)
                    files_removed += 1
                continue

            # If there's only one file matching, just move on.
            if len(matching_fnames) == 1:
                files_kept += 1
                continue

            date_tags = [re.search("(?<=_)\d{8}(?=[_\.])", os.path.basename(fn)).group() for fn in matching_fnames]
            max_date_tag = max(date_tags)
            tiles_that_match_date_tags = [fn for fn in matching_fnames if re.search(max_date_tag, os.path.basename(fn)) != None]
            if len(tiles_that_match_date_tags) == 1:
                max_file_to_keep = tiles_that_match_date_tags[0]
            elif len(tiles_that_match_date_tags) > 1:
                # Look for tiles that have the "USGS_13" marker. There should be only one.
                tiles_with_usgs13 = [fn for fn in tiles_that_match_date_tags if os.path.basename(fn).find("USGS_13_") > -1]
                assert len(tiles_with_usgs13) == 1
                max_file_to_keep = tiles_with_usgs13[0]
            else:
                # Shouldn't get here
                assert False

            print(loc_tag, max_date_tag)

            for fn in matching_fnames:
                if fn == max_file_to_keep:
                    print("Keeping", os.path.basename(max_file_to_keep))
                    files_kept += 1
                    continue
                else:
                    print("Omitting", os.path.basename(fn), "in lieu of", os.path.basename(max_file_to_keep))
                    # os.remove(fn)
                    shutil.move(fn, dirname_omitted)
                    files_removed += 1

            # for i in range(len(date_tags)):
            #     print(loc_tag, date_tags[i])
            #
            #     if i==max_i:
            #         files_kept += 1
            #         continue
            #
            #     else:
            #         print("Removing", os.path.basename(matching_fnames[i]), "in lieu of", os.path.basename(matching_fnames[max_i]))
            #         os.remove(matching_fnames[i])
            #         files_removed += 1

        print("Kept", files_kept, ",", "Removed", files_removed)
        print("Done.")
        return


    def convert_tiles(self,
                      infile_regex = r"USGS_1(3*)_(\w{7})_(\d{8}).tif\Z",
                      suffix_projected = "_epsg4326",
                      suffix_vertical = "_egm2008",
                      n_subprocs = 15,
                      overwrite = False,
                      verbose = True,
                      subproc_verbose = False):
        src_path = self.config._abspath(self.config.source_datafiles_directory)
        input_tilenames = utils.traverse_directory.list_files(src_path,
                                                              regex_match=infile_regex)
        if verbose:
            print(len(input_tilenames), "input USGS tiles.")

        # mask_temp_fnames = [os.path.join(os.path.dirname(fname), "converted", os.path.splitext(os.path.basename(fname))[0] + suffix_mask + ".tif") for fname in input_tilenames]
        horz_trans_fnames = [os.path.join(os.path.dirname(fname), "converted",
                                          (os.path.splitext(os.path.basename(fname))[0] + suffix_projected + ".tif")) for fname in
                             input_tilenames]
        output_fnames = [os.path.splitext(dfn)[0] + suffix_vertical + ".tif" for dfn in horz_trans_fnames]

        self_args = [self] * len(input_tilenames)

        args = list(zip(self_args, input_tilenames, horz_trans_fnames, output_fnames))
        kwargs = {"overwrite": overwrite,
                  "verbose": subproc_verbose}

        tempdirs = [os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "scratch_data", "temp{0:04d}".format(i))) for i in
                    range(len(input_tilenames))]

        utils.parallel_funcs.process_parallel(source_dataset_USGS_TNM.transform_single_tile,
                                              args,
                                              kwargs_list=kwargs,
                                              outfiles=output_fnames,
                                              temp_working_dirs=tempdirs,
                                              max_nprocs=n_subprocs,
                                              overwrite_outfiles=overwrite
                                              )

        return

    @staticmethod
    def get_tile_resolution_from_fname(fname):
        ds = gdal.Open(fname, gdal.GA_ReadOnly)
        gt = ds.GetGeoTransform()
        ds = None
        # Rather than testing if the x-res and y-res are exactly idential, instead test if they're within a tiny epsilon of each other.
        eps = 1e-9
        try:
            assert (abs(gt[5]) - eps) <= abs(gt[1]) <= (abs(gt[5]) + eps)
        except AssertionError as e:
            print(gt)
            raise e
        return abs(gt[1])
        # size_str = re.search("(?<=USGS_)\d+(?=_(\w{7})_(\d{8})\.tif\Z)", fname).group()
        # if size_str == "1":
        #     return 1 / (60. * 60.)
        # elif size_str == "13":
        #     return 1 / (60. * 60. * 3)

    def transform_single_tile(self, input_name, horiz_transform_name_TEMP, output_name, overwrite=False, verbose=True):
        if os.path.exists(output_name):
            if overwrite or gdal.Open(output_name, gdal.GA_ReadOnly) is None:
                os.remove(output_name)
            else:
                return

        # print(input_name, horiz_transform_name_TEMP)
        # return

        # if os.path.exists(mask_name_TEMP):
        #     os.remove(mask_name_TEMP)

        # # First, maks out bad (interpolated, tinned) values.
        # self.remove_interpolated_values_from_tile(input_name, mask_name_TEMP, verbose=verbose)

        if os.path.exists(horiz_transform_name_TEMP):
            os.remove(horiz_transform_name_TEMP)

        # First, transform horizontally.
        resolution = self.get_tile_resolution_from_fname(input_name)
        gdal_cmd = ["gdalwarp",
                    "-s_srs", "EPSG:4269",
                    "-t_srs", "EPSG:4326",
                    "-dstnodata", "0.0",
                    "-tr", str(resolution), str(resolution),
                    "-r", "bilinear",
                    "-of", "GTiff",
                    "-co", "COMPRESS=DEFLATE",
                    "-co", "PREDICTOR=2",
                    "-co", "ZLEVEL=5",
                    input_name, horiz_transform_name_TEMP]
        subprocess.run(gdal_cmd, capture_output=not verbose, text=True)

        if not os.path.exists(horiz_transform_name_TEMP):
            return

        # Then, transform that file vertically.
        etopo.convert_vdatum.convert_vdatum(horiz_transform_name_TEMP,
                                            output_name,
                                            input_vertical_datum="navd88",
                                            output_vertical_datum="egm2008",
                                            cwd=None,
                                            verbose=verbose)

        # Remove temporary tiles.
        # if os.path.exists(mask_name_TEMP):
        #     os.remove(mask_name_TEMP)
        if os.path.exists(horiz_transform_name_TEMP):
            os.remove(horiz_transform_name_TEMP)

        return

    @staticmethod
    def extract_country_outline(countries_shpfile, ADM0_A3="USA"):
        """From the Natural Earth admin (countries) shapefile, extract the polygon for the United States"""
        base, ext = os.path.splitext(countries_shpfile)
        shp_out = base + "_" + ADM0_A3 + ext

        if os.path.exists(shp_out):
            return shp_out

        countries_gdf = geopandas.read_file(countries_shpfile)

        gdf_filtered = countries_gdf[countries_gdf.ADM0_A3 == ADM0_A3]
        print(gdf_filtered)
        assert len(gdf_filtered) == 1

        gdf_filtered.to_file(shp_out)
        print(shp_out, "written.")
        return shp_out

    def get_filename_from_tile_id(self, tile_id, include_geometry=False):
        gdf = self.get_geodataframe()
        filenames = gdf.filename.to_list()
        matching_files = [fn for fn in filenames if re.search(tile_id, os.path.basename(fn)) != None]
        if len(matching_files) == 0:
            if include_geometry:
                return None, None
            else:
                return None
        elif len(matching_files) > 1:
            return matching_files
        else:
            if include_geometry:
                mf = matching_files[0]
                polygon = gdf[gdf.filename == mf].geometry.to_list()[0]
                return mf, polygon
            else:
                return matching_files[0]

    def clip_country_polygon_to_tile_id(self, country_shp, tile_id, tile_polygon, overwrite=False):
        """Create a new shapefile, of the country outline, clipped to the tile bounds of a USGS_TNM tile."""
        assert os.path.exists(country_shp)
        base, ext = os.path.splitext(country_shp)
        shp_clipped_out = base + "_" + tile_id + ext

        if os.path.exists(shp_clipped_out):
            if overwrite:
                os.remove(shp_clipped_out)
            else:
                return shp_clipped_out

        gdf = geopandas.read_file(country_shp)
        gdf_clipped = geopandas.clip(gdf, tile_polygon, keep_geom_type=True)

        gdf_clipped.to_file(shp_clipped_out)
        if os.path.exists(shp_clipped_out):
            print("   ", os.path.basename(shp_clipped_out), "written.")

            return shp_clipped_out
        else:
            return None


    def crop_files_to_us_border(self, overwrite=False, regenerate_gpkg=True):
        files_to_omit_csv = self.config._abspath(self.config.tiles_to_crop_or_omit_csv)
        countries_shpfile = self.config._abspath(self.config.country_outlines_shapefile)

        files_omit_df = pandas.read_csv(files_to_omit_csv, index_col=False)
        # countries_gdf = geopandas.read_file(countries_shpfile)

        # tnm_gdf = self.get_geodataframe()

        print(files_to_omit_csv, os.path.exists(files_to_omit_csv))
        print(files_omit_df)

        # print(countries_shpfile, os.path.exists(countries_shpfile))
        # print(countries_gdf)
        # print(list(countries_gdf.columns))

        # print(tnm_gdf)

        # Extract the "United States of America" multi-polygon from the countries_gdf
        usa_polygon_shp = self.extract_country_outline(countries_shpfile)

        # For each row of the files_omit_df:
        for i,(_, files_omit_row) in enumerate(files_omit_df.iterrows()):
            tile_id, omit_or_mask = files_omit_row.tile_id, files_omit_row.omit_or_mask
            # print(tile_id, omit_or_mask)
            # 1. Find the matching USGS_TNM tile filename from the tnm_gdf (using the tile_id)
            tile_fname, tile_polygon = self.get_filename_from_tile_id(tile_id, include_geometry=True)
            assert (type(tile_fname) == str) or (tile_fname is None)
            print("{0}/{1}".format(i+1, len(files_omit_df)),
                  tile_id if tile_fname is None else os.path.basename(tile_fname),
                  f"({omit_or_mask})", end=" ")
            # 2. If "omit", delete the tile.
            if tile_fname is None:
                print("already gone.")
                continue
            if omit_or_mask == "omit":
                if os.path.exists(tile_fname):
                    os.remove(tile_fname)
                    print("removed.")
                else:
                    print("already gone.")
            elif omit_or_mask == "mask":
                # 3. If "mask", create an "_ORIGINAL" version of the file (just move the file to this).
                base, ext = os.path.splitext(tile_fname)
                tile_fname_original = base + "_ORIGINAL" + ext
                # If the "_ORIGINAL" version of the file exists, just leave it there, don't overwrite it. In that case,
                # just remove the previous modified one to re-create it.
                if os.path.exists(tile_fname_original):
                    if overwrite:
                        os.remove(tile_fname)
                    else:
                        print("already exists.")
                        continue
                else:
                    os.rename(tile_fname, tile_fname_original)
                print("\n   ", os.path.basename(tile_fname_original), "created.")

                # 4.    Then, run gdal masking on the tile to mask out any non-us data.
                kwargs = {"creationOptions": ["COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=5"],
                          "format": "GTiff"}
                # This is taking FOR>>>EVER! Let's see if we can speed it up by using a subset of the shapefile, clipped to the bounidng bot of the tile.
                usa_polygon_shp_tile_clipped = self.clip_country_polygon_to_tile_id(usa_polygon_shp, tile_id, tile_polygon)

                out_ds = gdal.Warp(tile_fname,
                                   tile_fname_original,
                                   cutlineDSName = usa_polygon_shp_tile_clipped,
                                   cropToCutline=True,
                                   dstNodata=-9999.0,
                                   **kwargs)
                out_ds = None
                if os.path.exists(tile_fname):
                    print("   ", os.path.basename(tile_fname), "written.")

            else:
                raise ValueError("Unhandled value for column 'omit_or_mask':", omit_or_mask)

            # return # Just for testing. Remove this at the end.

        gpkg_fname = self.geopackage_filename
        print(gpkg_fname, os.path.exists(gpkg_fname))

        if os.path.exists(gpkg_fname):
            if regenerate_gpkg:
                os.remove(gpkg_fname)
                time.sleep(0.1)

        if not os.path.exists(gpkg_fname):
            # This will automatically re-create the geopackage.
            self.get_geopkg_object().create_dataset_geopackage()

        return

if __name__ == "__main__":
    tnm = source_dataset_USGS_TNM()

    # Just to refresh the dataframe, after files may have been omitted.
    # gdf = tnm.get_geodataframe()

    # tnm.omit_all_but_latest_tiles()
    # tnm.eliminate_duplicates_in_links_file()
    # tnm.convert_tiles(n_subprocs=15, overwrite=False)
    tnm.crop_files_to_us_border()
