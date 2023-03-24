# -*- coding: utf-8 -*-

"""Source code for the FABDEM ETOPO source dataset class."""

import os
import re
import subprocess
from osgeo import gdal

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset
import datasets.CopernicusDEM.source_dataset_CopernicusDEM as copernicus
import utils.trim_ndv_mask
import utils.parallel_funcs
import datasets.dataset_geopackage
import utils.configfile
import numpy
etopo_config = utils.configfile.config()

class source_dataset_FABDEM(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "FABDEM_config.ini" )):
        """Initialize the FABDEM source dataset object."""

        etopo_source_dataset.ETOPO_source_dataset.__init__(self, "FABDEM", configfile)

    # def find_matching_copernicus_1s_tile(cop_files_list, fab_tilename):
    @staticmethod
    def find_matching_copernicus_1s_tile(cop_basedir, fab_tilename):
        fab_basename = os.path.basename(fab_tilename)
        fab_n_tag = fab_basename[0:3]
        fab_e_tag = fab_basename[3:7]
        assert (fab_n_tag[0] in ('N','S')) and (fab_e_tag[0] in ('E','W'))

        cop_filename = os.path.join(cop_basedir,
                                    "Copernicus_DSM_COG_10_" + fab_n_tag + "_00_" + fab_e_tag + "_00_DEM_1s.tif")
        if os.path.exists(cop_filename):
            return cop_filename

        else:
            # There should be a matching CopernicusDEM tile for each & every FABDEM tile. We've hit an error if we get here.
            raise ValueError("Did not find a matching Copernicus tile for FABDEM tile " + fab_basename)

        # for cop_fn in cop_files_list:
        #     if re.search('10_' + fab_n_tag + '_00_' + fab_e_tag + '_00_DEM', os.path.basename(cop_fn)) is not None:
        #         return cop_fn


    def trim_all_tiles_to_COP_bounds(self, nprocs=15, overwrite=False, verbose=True):
        """FABDEM has some strange edge artifacts that create a slightly-exaggerated shoreline compared
        to the CopernicusDEM tiles it's based upon. Fix that.

        Compare each FABDEM tile to its subsequent CopernicusDEM tile (1s-resampled to the same grid), and trim it.
        Put the outfile in the trimmed directory."""
        files = self.retrieve_all_datafiles_list(resolution_s=1)

        trimmed_dir = os.path.join(os.path.dirname(os.path.dirname(files[0])), "trimmed")
        assert os.path.exists(trimmed_dir)
        outfiles = [os.path.join(trimmed_dir, os.path.splitext(os.path.basename(fn))[0] + "_trimmed.tif") for fn in files]

        copernicus_object = copernicus.source_dataset_CopernicusDEM()
        copernicus_dir = copernicus_object.config._abspath(copernicus_object.config.soruce_datafiles_regridded_1s_directory)
        # cop_files = [os.path.join(copernicus_dir, fn) for fn in os.listdir(copernicus_dir) if re.search("_DEM_1s\.tif\Z", fn) is not None]
        if verbose:
            print("Finding matching Copernicus tiles...")
        cop_files_to_use = [self.find_matching_copernicus_1s_tile(copernicus_dir, fn) for fn in files]

        target_func = utils.trim_ndv_mask.trim_gtif
        args = [(fn_cop, fn_orig, fn_out) for (fn_orig, fn_cop, fn_out) in zip(files, cop_files_to_use, outfiles)]
        kwargs = {"out_ndv": -9999,
                  "verbose": False,
                  "overwrite" : overwrite}

        if verbose:
            print("Kicking off processing...")

        utils.parallel_funcs.process_parallel(target_func,
                                              args,
                                              kwargs_list=kwargs,
                                              outfiles=outfiles,
                                              max_nprocs = nprocs,
                                              verbose=verbose)
        return

    def reassign_NDV_0_to_neg9999(self, resolution_s=1, overwrite=True, verbose=True):
        filenames = self.retrieve_all_datafiles_list(resolution_s=resolution_s, verbose=verbose)

        for i,fn in enumerate(filenames):
            fn_new = fn.replace("/orig/", "/reset_ndv/")
            # Get rid of the subdirectory it was in before.
            fn_new = fn_new.replace(re.search(r"(?<=/reset_ndv/)[\w-]+/(?=[\w-]+\.tif\Z)", fn_new).group(), "")

            ds = gdal.Open(fn, gdal.GA_ReadOnly)

            if os.path.exists(fn_new):
                if overwrite:
                    os.remove(fn_new)
                else:
                    if verbose:
                        print("{0}/{1}".format(i + 1, len(filenames)),
                              os.path.basename(fn), "already exists.")
                        continue

            ds.GetDriver().CreateCopy(fn_new, ds, strict=0)
            ds = None
            ds = gdal.Open(fn_new, gdal.GA_Update)


            band = ds.GetRasterBand(1)
            array = band.ReadAsArray()
            ndv = band.GetNoDataValue()

            # print(fn)
            # print("NDV", band.GetNoDataValue())
            print("{0}/{1}".format(i+1,len(filenames)),
                  os.path.basename(fn),
                  "setting",
                  numpy.count_nonzero(array==0),
                  "{0:0.1f}%".format(100*numpy.count_nonzero(array==0)/array.size),
                  "to NDV of",
                  ndv)

            array[array == 0] = ndv
            band.WriteArray(array)
            band = None
            ds = None

    def create_trimmed_15s_tiles(self, tile_id = None, overwrite=False, nprocs=18, verbose=True, verbose_gdal=False):
        """Create 15s tiles, that are trimmed to a realistic coastline without coastal-creep.

        Create a 15s average, a 15s nearest, then trim the 15s-average to the 15s-nearest mask. Save that to
        a 15s-average_trimmed file. DOES THIS WORK AT 15s ETOPO GRID-LINES?"""
        trimmed_1s_dir = self.config._abspath(self.config.source_datafiles_directory_1s_trimmed)
        trimmed_1s_files = sorted([os.path.join(trimmed_1s_dir, fn) for fn in os.listdir(trimmed_1s_dir) if re.search(r"_FABDEM_V1-0_trimmed.tif\Z", fn) is not None])

        near_dir_15 = self.config._abspath(self.config.source_datafiles_directory_15s_nearest)
        avg_dir_15 = self.config._abspath(self.config.source_datafiles_directory_15s_average)
        avg_src_grid_dir_15 = self.config._abspath(self.config.source_datafiles_directory_15s_average_source_grid)
        avg_trm_dir_15 = self.config._abspath(self.config.source_datafiles_directory_15s_average_trimmed)

        near_15_fnames = [os.path.join(near_dir_15, os.path.basename(fn).replace("trimmed.tif","15s_near.tif")) for fn in trimmed_1s_files]
        avg_15s_fnames = [os.path.join(avg_dir_15, os.path.basename(fn).replace("trimmed.tif","15s_avg.tif")) for fn in trimmed_1s_files]
        avg_src_grid_15s_fnames = [os.path.join(avg_src_grid_dir_15, os.path.basename(fn).replace("trimmed.tif","15s_avg_src_grid.tif")) for fn in trimmed_1s_files]
        avg_trm_15s_fnames = [os.path.join(avg_trm_dir_15, os.path.basename(fn).replace("trimmed.tif","15s_avg_trimmed.tif")) for fn in trimmed_1s_files]

        # Allow us to do just one tile_id for testing, if we choose.
        if tile_id:
            trimmed_1s_files = [fn for fn in trimmed_1s_files if fn.find(tile_id) > -1]
            near_15_fnames = [fn for fn in near_15_fnames if fn.find(tile_id) > -1]
            avg_15s_fnames = [fn for fn in avg_15s_fnames if fn.find(tile_id) > -1]
            avg_src_grid_15s_fnames = [fn for fn in avg_src_grid_15s_fnames if fn.find(tile_id) > -1]
            avg_trm_15s_fnames = [fn for fn in avg_trm_15s_fnames if fn.find(tile_id) > -1]

        args = list(zip(trimmed_1s_files, near_15_fnames, avg_src_grid_15s_fnames, avg_15s_fnames, avg_trm_15s_fnames))
        kwargs = {"verbose": False if (len(trimmed_1s_files) > 1) else verbose,
                  "verbose_gdal": False if (len(trimmed_1s_files) > 1) else verbose_gdal,
                  "overwrite": overwrite}

        utils.parallel_funcs.process_parallel(self.create_single_trimmed_15s_tile,
                                              args,
                                              kwargs_list=kwargs,
                                              outfiles = avg_trm_15s_fnames,
                                              overwrite_outfiles = overwrite,
                                              max_nprocs = nprocs,
                                              verbose=verbose)
        return


    @staticmethod
    def create_single_trimmed_15s_tile(tilename_1s,
                                       tilename_15s_near,
                                       tilename_15s_average_src_grid,
                                       tilename_15s_average_etopo_grid,
                                       tilename_15s_average_trimmed,
                                       overwrite=False,
                                       verbose=True,
                                       verbose_gdal=False):
        """From a single FABDEM tile, create a resampled and trimmed 15s version of it.

        Try it on the ETOPO grid and see how we do."""
        tile_1s_basename = os.path.basename(tilename_1s)
        n_tag = tile_1s_basename[0:3]
        etopo_ymin = (-1 if (n_tag[0] == "S") else 1) * int(n_tag[1:])
        e_tag = tile_1s_basename[3:7]
        etopo_xmin = (-1 if (e_tag[0] == "W") else 1) * int(e_tag[1:])
        etopo_ymax = etopo_ymin + 1
        etopo_xmax = etopo_xmin + 1
        size = 3600/15

        # First check if we already have the output file. If so (and we're not overwriting), just move along.
        if os.path.exists(tilename_15s_average_trimmed) and not overwrite:
            return

        # 1. Create the nearest-neighbor tile. This works fine at 15s.
        if os.path.exists(tilename_15s_near):
            os.remove(tilename_15s_near)

        if not os.path.exists(tilename_15s_near):
            gdal_warp_near_cmd = ["gdalwarp",
                                  "-te", repr(etopo_xmin), repr(etopo_ymin), repr(etopo_xmax), repr(etopo_ymax),
                                  "-ts", str(int(size)), str(int(size)),
                                  "-r", "near",
                                  "-co", "COMPRESS=DEFLATE",
                                  "-co", "PREDICTOR=2",
                                  tilename_1s,
                                  tilename_15s_near]
            if verbose_gdal:
                print(" ".join(gdal_warp_near_cmd))
            try:
                subprocess.run(gdal_warp_near_cmd, capture_output=not verbose_gdal)
            except (Exception, KeyboardInterrupt) as e:
                if os.path.exists(tilename_15s_near):
                    os.remove(tilename_15s_near)
                raise e

        # 2. Create a TEMP average file at the FABDEM grid boundaries.
        # We'll resample this in a bit to the ETOPO grid boundaries.

        if os.path.exists(tilename_15s_average_src_grid):
            os.remove(tilename_15s_average_src_grid)

        if not os.path.exists(tilename_15s_average_src_grid):
            ds_orig = gdal.Open(tilename_1s, gdal.GA_ReadOnly)
            src_xmin, src_xres, _, src_ymax, _, src_yres = ds_orig.GetGeoTransform()
            src_xsize, src_ysize = ds_orig.RasterXSize, ds_orig.RasterYSize
            src_xmax = src_xmin + (src_xres * src_xsize)
            src_ymin = src_ymax + (src_yres * src_ysize)
            gdal_warp_avg_src_grid_cmd = ["gdalwarp",
                                          "-te", repr(src_xmin), repr(src_ymin), repr(src_xmax), repr(src_ymax),
                                          "-ts", str(int(size)), str(int(size)),
                                          "-r", "average",
                                          "-co", "COMPRESS=DEFLATE",
                                          "-co", "PREDICTOR=2",
                                          tilename_1s,
                                          tilename_15s_average_src_grid]
            if verbose_gdal:
                print(" ".join(gdal_warp_avg_src_grid_cmd))
            try:
                subprocess.run(gdal_warp_avg_src_grid_cmd, capture_output=not verbose_gdal)
            except (Exception, KeyboardInterrupt) as e:
                if os.path.exists(tilename_15s_average_src_grid):
                    os.remove(tilename_15s_average_src_grid)
                raise e

        # 3. Then, we'll fucking resample our half-pixel-offset grid using nearest neighbor to the ETOPO grid.
        # This is so weird but it works. And no, the 1/2-second offset in a 15s grid won't matter.
        if os.path.exists(tilename_15s_average_etopo_grid):
            os.remove(tilename_15s_average_etopo_grid)

        if not os.path.exists(tilename_15s_average_etopo_grid):
            gdal_warp_avg_src_grid_cmd = ["gdalwarp",
                                          "-te", repr(etopo_xmin), repr(etopo_ymin), repr(etopo_xmax), repr(etopo_ymax),
                                          "-ts", str(int(size)), str(int(size)),
                                          "-r", "near",
                                          "-co", "COMPRESS=DEFLATE",
                                          "-co", "PREDICTOR=2",
                                          tilename_15s_average_src_grid,
                                          tilename_15s_average_etopo_grid]
            if verbose_gdal:
                print(" ".join(gdal_warp_avg_src_grid_cmd))
            try:
                subprocess.run(gdal_warp_avg_src_grid_cmd, capture_output=not verbose_gdal)
            except (Exception, KeyboardInterrupt) as e:
                if os.path.exists(tilename_15s_average_etopo_grid):
                    os.remove(tilename_15s_average_etopo_grid)
                raise e

        if os.path.exists(tilename_15s_average_trimmed):
            if overwrite:
                os.remove(tilename_15s_average_trimmed)

        if not os.path.exists(tilename_15s_average_trimmed):
            gdal_warp_avg_trm_cmd = ["python", os.path.join(etopo_config.project_base_directory,
                                                            "src", "utils",
                                                            "trim_ndv_mask.py"),
                                     tilename_15s_near,
                                     tilename_15s_average_etopo_grid,
                                     "-O", tilename_15s_average_trimmed,
                                     "--quiet"]
            if verbose_gdal:
                print(" ".join(gdal_warp_avg_trm_cmd))
            try:
                subprocess.run(gdal_warp_avg_trm_cmd, capture_output=not verbose_gdal)
            except (Exception, KeyboardInterrupt) as e:
                if os.path.exists(tilename_15s_average_trimmed):
                    os.remove(tilename_15s_average_trimmed)
                raise e


        if verbose:
            print(os.path.basename(tilename_15s_average_trimmed), end=" ")
            if os.path.exists(tilename_15s_average_trimmed):
                print("written.")
            else:
                print("NOT written.")

        return


if __name__ == "__main__":
    fab = source_dataset_FABDEM()
    fab.reassign_NDV_0_to_neg9999()
    # fab.trim_all_tiles_to_COP_bounds(nprocs=20)
    # fab.create_trimmed_15s_tiles(overwrite=False, nprocs=18, verbose=True, verbose_gdal=True)