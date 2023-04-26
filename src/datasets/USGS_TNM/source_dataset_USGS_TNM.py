# -*- coding: utf-8 -*-

"""Source code for the USGS_TNM ETOPO source dataset class."""

import os
import re
import numpy
import subprocess
from osgeo import gdal

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


    def remove_all_but_latest_tiles(self):
        """Remove all the but the latest of each geographically unique tile.

        The USGS dataset contains many duplicate tiles at 1/3 arc-second that were generated at different dates.
        The tiles all have a format of 'USGS_13_nYYwXXX_DDDDDDDD.tif', with DDDDDDDD being a YYYYMMDD date string.
        For each unique nYYwXXX tile, find and keep the tile that has the latest DDDDDDDD string. Delete the rest."""

        dirname = self.config._abspath(self.config.source_datafiles_directory)
        regex_str = self.config.datafiles_regex
        fnames = utils.traverse_directory.list_files(dirname, regex_str)
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
                    print("Removing", os.path.basename(fname), "to get rid of offshore errors.")
                    os.remove(fname)
                    files_removed += 1
                continue

            # If there's only one file matching, just move on.
            if len(matching_fnames) == 1:
                files_kept += 1
                continue

            date_tags = [re.search("(?<=_)\d{8}(?=\.tif\Z)", os.path.basename(fn)).group() for fn in matching_fnames]
            max_i = numpy.argmax(date_tags)
            for i in range(len(date_tags)):
                print(loc_tag)

                if i==max_i:
                    files_kept += 1
                    continue

                else:
                    print("Removing", os.path.basename(matching_fnames[i]), "in lieu of", os.path.basename(matching_fnames[max_i]))
                    os.remove(matching_fnames[i])
                    files_removed += 1

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
                                              max_nprocs=n_subprocs
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

if __name__ == "__main__":
    tnm = source_dataset_USGS_TNM()
    gdf = tnm.get_geodataframe()
    # tnm.convert_tiles(n_subprocs=15)
    # tnm.eliminate_duplicates_in_links_file()
    # tnm.remove_all_but_latest_tiles()
