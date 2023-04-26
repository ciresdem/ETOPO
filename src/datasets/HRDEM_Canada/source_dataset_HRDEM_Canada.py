# -*- coding: utf-8 -*-

"""Source code for the HRDEM_Canada ETOPO source dataset class."""

import os
from osgeo import gdal
import re
import subprocess

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset
import utils.traverse_directory
import etopo.convert_vdatum

utm_zone_regex = r"(?<= / UTM zone )(\d{2})[NS]"
# These cover all the UTM zones covered by the BlueTopo tiles.
epsg_lookups  = {"19N" : 2960,
                 "20N" : 2961,
                 }

# Suppress gdal warnings in this file.
gdal.SetConfigOption('CPL_LOG', "/dev/null")


class source_dataset_HRDEM_Canada(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "HRDEM_Canada_config.ini" )):
        """Initialize the HRDEM_Canada source dataset object."""

        super(source_dataset_HRDEM_Canada, self).__init__("HRDEM_Canada", configfile)


    def get_epsg_from_tile_name(self, filename, return_utm_zone_name=False, verbose=True):
        """Get the UTM zone of the tile from the file name.

        Generally only used if the projection information is not contained within the file itself."""
        utm_zone_str = re.search(r"(?<=dtm_[12]m_utm)\d{2}(?=_[ew]_(\d{2}))", filename).group() + "N"
        if return_utm_zone_name:
            return utm_zone_str
        else:
            return epsg_lookups[utm_zone_str]


    def get_file_epsg(self, filename, return_utm_zone_name=False, derive_from_filename_if_missing=True, verbose=True):
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
                return self.get_epsg_from_tile_name(filename, return_utm_zone_name=utm_zone_name, verbose=verbose)
            else:
                return None
        try:
            utmz = re.search(utm_zone_regex, wkt).group()
        except AttributeError:
            # If we hit here, we aren't actally using a tile with a real UTM zone. Return None.
            return None
        if return_utm_zone_name:
            return utmz
        else:
            return epsg_lookups[utmz]

    def reproject_tiles(self,
                        infile_regex = r"dtm_[12]m_utm(\d{2})_[ew]_(\d{2})_(\d{2,3}).tif\Z",
                        suffix="_epsg4326",
                        suffix_vertical="_egm2008",
                        n_subprocs = 15,
                        overwrite = False,
                        verbose=True,
                        subproc_verbose=False):
        """Project all the tiles into WGS84/latlon coordinates.

        The fucked-up NAD83 / UTM zone XXN is fucking with waffles. Converting them is the easiest answer now.
        After we do this, we can change the config.ini to look for the new "_epsg4326" tiles. Also, we can disable and
        delete the 5 BlueTopo_14N to _19N datasets, because they'll all be projected into the same coordinate system,
        which will be a lot easier.
        """

        # tilenames = self.retrieve_all_datafiles_list(verbose=verbose)
        src_path = self.config._abspath(self.config.source_datafiles_directory)
        input_tilenames = utils.traverse_directory.list_files(src_path,
                                                              regex_match = infile_regex)
        if verbose:
            print(len(input_tilenames), "input HRDEM tiles.")

        # mask_temp_fnames = [os.path.join(os.path.dirname(fname), "converted", os.path.splitext(os.path.basename(fname))[0] + suffix_mask + ".tif") for fname in input_tilenames]
        horz_trans_fnames = [os.path.join(os.path.dirname(fname), "converted", (os.path.splitext(os.path.basename(fname))[0] + suffix + ".tif")) for fname in input_tilenames]
        output_fnames = [os.path.splitext(dfn)[0] + suffix_vertical + ".tif" for dfn in horz_trans_fnames]

        self_args = [self] * len(input_tilenames)

        args = list(zip(self_args, input_tilenames, horz_trans_fnames, output_fnames))
        kwargs = {"overwrite": overwrite,
                  "verbose": subproc_verbose}

        tempdirs = [os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "scratch_data", "temp{0:04d}".format(i))) for i in range(len(input_tilenames))]


        utils.parallel_funcs.process_parallel(source_dataset_HRDEM_Canada.transform_single_tile,
                                              args,
                                              kwargs_list = kwargs,
                                              outfiles = output_fnames,
                                              temp_working_dirs = tempdirs,
                                              overwrite_outfiles = overwrite,
                                              max_nprocs=n_subprocs
                                              )

        return

    @staticmethod
    def get_tile_size_from_fname(fname):
        return int(re.search("(?<=dtm_)\d+(?=m_utm)", fname).group())

    def transform_single_tile(self, input_name, horiz_transform_name_TEMP, output_name, overwrite=False, verbose=True):
        # 1 arc-second is ~30.8 m at the equator.
        # The 0.3 tiles (level 3) are 16 m, 0.4 are 8 m, 0.5 are 4 m
        res_lookup_dict = {16: 16 / (30.8 * 60 * 60),
                           8 : 8 / (30.8 * 60 * 60),
                           4 : 4 / (30.8 * 60 * 60),
                           2 : 2 / (30.8 * 60 * 60),
                           1 : 1 / (30.8 * 60 * 60)}

        # Some of these files aren't using a proper UTM zone (nor are they in NAVD88). Ignore them for now.
        if self.get_file_epsg(input_name, verbose=verbose) is None:
            return

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
        resolution = res_lookup_dict[self.get_tile_size_from_fname(input_name)]
        gdal_cmd = ["gdalwarp",
                    "-s_srs", "EPSG:{0}".format(self.get_file_epsg(input_name, return_utm_zone_name=False, derive_from_filename_if_missing=True, verbose=False)),
                    "-t_srs", "EPSG:4326",
                    "-tr", str(resolution), str(resolution),
                    "-r", "bilinear",
                    "-of", "GTiff",
                    "-co", "COMPRESS=DEFLATE",
                    "-co", "PREDICTOR=2",
                    "-co", "ZLEVEL=5",
                    input_name, horiz_transform_name_TEMP]
        if verbose:
            print(" ".join(gdal_cmd))
        subprocess.run(gdal_cmd, capture_output = not verbose, text=True)

        if not os.path.exists(horiz_transform_name_TEMP):
            return

        # Then, transform that file vertically.
        etopo.convert_vdatum.convert_vdatum(horiz_transform_name_TEMP,
                                            output_name,
                                            input_vertical_datum="cgvd2013",
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
    gdf = source_dataset_HRDEM_Canada().get_geodataframe()
    # source_dataset_HRDEM_Canada().reproject_tiles(n_subprocs=13, verbose=True, overwrite=True, subproc_verbose=False)