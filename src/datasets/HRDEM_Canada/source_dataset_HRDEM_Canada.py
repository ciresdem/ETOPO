# -*- coding: utf-8 -*-

"""Source code for the HRDEM_Canada ETOPO source dataset class."""

import os
from osgeo import gdal
import re
import subprocess
import numpy

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset
import utils.traverse_directory
import etopo.convert_vdatum
import etopo.coastline_mask
import utils.configfile
import utils.parallel_funcs

etopo_config = utils.configfile.config()

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
        return int(re.search(r"(?<=dtm_)\d+(?=m_utm)", fname).group())

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
                    "-s_srs", "EPSG:{0}".format(self.get_file_epsg(input_name,
                                                                   return_utm_zone_name=False,
                                                                   derive_from_filename_if_missing=True,
                                                                   verbose=False)),
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


    def downsample_tiles_and_mask_ocean(self, resolution_s=1, overwrite=False, n_procs=1, verbose=True):
        """Downsample the tiles to whatever resampled resolution we need them at (presumably, 1s or 15s), and
        mask out the ocean using the Copernicus land-mask data.

        Best if done at 1s or coarser resolution, as the Copernicus land-masks are at 1s.

        Put in a folder named 1s_masked."""
        input_tiles = self.retrieve_all_datafiles_list(verbose=verbose)

        resampled_tiles = [os.path.abspath(os.path.join(os.path.dirname(fn), "..", "..", "{0}s_masked".format(resolution_s),
                           os.path.splitext(os.path.basename(fn))[0] + "_{0}s".format(resolution_s) + os.path.splitext(fn)[1])) for fn in input_tiles]
        coastline_mask_tiles = [os.path.splitext(fn)[0][:-3] + "_coastline_mask" + os.path.splitext(fn)[1] for fn in resampled_tiles]
        output_tiles = [os.path.splitext(fn)[0] + "_masked" + os.path.splitext(fn)[1] for fn in resampled_tiles]

        tempdirs = [os.path.join(etopo_config._abspath(etopo_config.etopo_cudem_cache_directory), "temp{0}".format(i))
                    for i in range(len(input_tiles))]

        args_lists = list(zip(input_tiles, coastline_mask_tiles, output_tiles))
        kwargs = {"resolution_s" : 1,
                  "overwrite": overwrite,
                  "verbose": True if ((n_procs == 1) and verbose) else False}

        utils.parallel_funcs.process_parallel(self.downsample_and_mask_ocean_single_tile,
                                              args_lists,
                                              kwargs_list=kwargs,
                                              outfiles=output_tiles,
                                              temp_working_dirs=tempdirs,
                                              overwrite_outfiles=overwrite,
                                              max_nprocs=n_procs,
                                              verbose=verbose
                                              )
        # self.downsample_and_mask_ocean_single_tile(*args_lists[0], **kwargs)

        # for i in range(len(input_tiles)):
        #     print(input_tiles[i], "\n    ", resampled_tiles[i], "\n    ", coastline_mask_tiles[i], "\n    ", output_tiles[i], "\n")

    @staticmethod
    def downsample_and_mask_ocean_single_tile(input_file, coastline_mask_file, output_file, resolution_s=1, overwrite=False, verbose=True):
        """See description for self.downsample_tiles_and_mask_ocean(), above."""
        # 1. Resample input_file to resolution. Be sure to align it with the closest whole-cell-boundary to that resolution.
        # 2. Create coastline mask file of that resampled file.
        # 3. Create copy of resampled tile, and edit to mask by the coastline mask.

        if os.path.exists(output_file):
            if overwrite:
                if verbose:
                    print("Removing previous", os.path.basename(output_file))
                os.remove(output_file)
            else:
                return

        # Define new bounds
        resolution_deg = resolution_s / 3600.
        # Open input tile.
        ds = gdal.Open(input_file, gdal.GA_ReadOnly)
        band = ds.GetRasterBand(1)
        # Get geotransform of input tile.
        gt = ds.GetGeoTransform()
        # Get NDV of input tile.
        ndv = band.GetNoDataValue()
        # Get array of input tile. Fine all
        array = band.ReadAsArray()

        band = None
        ds = None

        # Find the min_x, min_y, max_x, max_y coordinates of the actual data.
        yi_good, xi_good = numpy.where(array != ndv)
        min_xi = numpy.min(xi_good)
        max_xi = numpy.max(xi_good)
        min_yi = numpy.min(yi_good)
        max_yi = numpy.max(yi_good)
        # Remember, latitudes iterate downward, so the min_lat is the maximum index of good data, and vice-versa.
        assert gt[5] < 0
        min_lat = gt[3] + (gt[5] * (max_yi + 1))
        max_lat = gt[3] + (gt[5] * min_yi)
        min_lon = gt[0] + (gt[1] * min_xi)
        max_lon = gt[0] + (gt[1] * (max_xi + 1))
        # print(input_file)
        # print(gt)
        # print(numpy.count_nonzero(array != ndv) * 100. / array.size)
        # print(numpy.where(array != ndv))
        # print(min_lon, max_lon, min_lat, max_lat)

        # Align those boundaries in 1s coordinate envelope. This will be our raster.
        min_lat_1se = min_lat - (min_lat % resolution_deg)
        max_lat_1se = max_lat + (resolution_deg - (max_lat % resolution_deg))
        min_lon_1se = min_lon - (min_lon % resolution_deg)
        max_lon_1se = max_lon + (resolution_deg - (max_lon % resolution_deg))
        # print(min_lon_1se, max_lon_1se, min_lat_1se, max_lat_1se)

        # Resample to ESPG lat/lon, at 1s, using "average" mask, which gives good values but bad outlines.
        gdal_warp_avg_cmd = ["gdalwarp",
                             "-te", repr(min_lon_1se), repr(min_lat_1se), repr(max_lon_1se), repr(max_lat_1se),
                             "-tr", repr(resolution_deg), repr(resolution_deg),
                             "-tap",
                             "-co", "compress=lzw",
                             "-t_srs", "EPSG:4326",
                             "-r", "average",
                             input_file, output_file]

        if verbose:
            print(" ".join(gdal_warp_avg_cmd))
        subprocess.run(gdal_warp_avg_cmd, capture_output=not verbose)

        # Modify the above command to resample to nearest neighbor, which gives better outlines.
        # We will use the nearest-neighbor mask to get the correct extent of the output file.
        gdal_warp_near_cmd = gdal_warp_avg_cmd.copy()
        assert gdal_warp_near_cmd[-3] == "average"
        gdal_warp_near_cmd[-3] = "near"
        near_fname_temp = os.path.splitext(output_file)[0] + "_near_TEMP.tif"
        gdal_warp_near_cmd[-1] = near_fname_temp

        if os.path.exists(near_fname_temp):
            os.remove(near_fname_temp)

        if verbose:
            print(" ".join(gdal_warp_near_cmd))
        subprocess.run(gdal_warp_near_cmd, capture_output=not verbose)

        # Calcualte the coastline mask.
        if os.path.exists(coastline_mask_file):
            os.remove(coastline_mask_file)

        etopo.coastline_mask.create_coastline_mask(output_file,
                                                   mask_out_lakes=False,
                                                   include_gmrt=False,
                                                   mask_out_buildings=False,
                                                   mask_out_urban=False,
                                                   mask_out_nhd=False,
                                                   use_osm_planet=False,
                                                   output_file=coastline_mask_file,
                                                   run_in_tempdir=False,
                                                   horizontal_datum_only=True,
                                                   verbose=verbose)

        if not os.path.exists(coastline_mask_file):
            raise FileNotFoundError(coastline_mask_file)

        # Now, re-open the output tile to edit it according to the coastline mask tile.
        ds = gdal.Open(output_file, gdal.GA_Update)
        band = ds.GetRasterBand(1)
        ndv = band.GetNoDataValue()
        array = band.ReadAsArray()

        ds_coast = gdal.Open(coastline_mask_file, gdal.GA_ReadOnly)
        band_coast = ds_coast.GetRasterBand(1)
        array_coast = band_coast.ReadAsArray()
        del band_coast
        del ds_coast

        ds_near = gdal.Open(near_fname_temp, gdal.GA_ReadOnly)
        band_near = ds_near.GetRasterBand(1)
        array_near = band_near.ReadAsArray()
        ndv_near = band_near.GetNoDataValue()
        del band_near
        del ds_near

        assert array.shape == array_coast.shape
        # Mask out any ocean cells with ndv.
        array[array_coast != 1] = ndv
        # Mask out any values not in the nearest-neighbor extent.
        array[array_near == ndv_near] = ndv

        band.WriteArray(array)
        band.GetStatistics(0, 1)
        ds.FlushCache()
        del band
        del ds

        os.remove(coastline_mask_file)
        os.remove(near_fname_temp)

        return


if __name__ == "__main__":
    hrdem = source_dataset_HRDEM_Canada()
    gdf = hrdem.get_geodataframe()
    # hrdem.reproject_tiles(n_subprocs=1, verbose=True, overwrite=True, subproc_verbose=True)
    # hrdem.downsample_tiles_and_mask_ocean(overwrite=False, n_procs=15, verbose=True)