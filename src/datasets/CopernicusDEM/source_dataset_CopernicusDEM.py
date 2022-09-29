# -*- coding: utf-8 -*-

"""Source code for the CopernicusDEM ETOPO source dataset class."""

import os
from osgeo import gdal
import subprocess
import multiprocessing
import re

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset
import utils.parallel_funcs
import utils.configfile
etopo_config = utils.configfile.config()

class source_dataset_CopernicusDEM(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "CopernicusDEM_config.ini" )):
        """Initialize the CopernicusDEM source dataset object."""

        etopo_source_dataset.ETOPO_source_dataset.__init__(self, "CopernicusDEM", configfile)


    def regrid_and_reassign_ndv_parallel(self, nprocs=10, overwrite=False, verbose=True):
        """Do the stuff in regrid_and_reassign_ndv(), but do it for all the tiles, in parallel."""
        files = self.retrieve_all_datafiles_list(verbose=verbose)
        output_dir = os.path.dirname(files[0]) + "_regridded_1s"

        output_fns = [os.path.join(output_dir, os.path.basename(fn).replace(".tif", "_1s.tif")) for fn in files]
        target_func = source_dataset_CopernicusDEM.regrid_and_reassign_ndv

        kw_args = [{"tilename": fn,
                    "tilename_out": fn_out,
                    "overwrite": overwrite,
                    "verbose": False,
                    "verbose_gdal": False} for (fn, fn_out) in zip(files, output_fns)]

        args_list = [[self]] * len(files)

        utils.parallel_funcs.process_parallel(target_func,
                                              args_list,
                                              kwargs_list = kw_args,
                                              outfiles = output_fns,
                                              overwrite_outfiles = overwrite,
                                              max_nprocs = nprocs,
                                              verbose=verbose)

        return
    def regrid_and_reassign_ndv(self,
                                tilename=None,
                                tilename_out=None,
                                overwrite=False,
                                verbose=True,
                                verbose_gdal=False):
        """Re-grid to full 1s coverage and re-assign NDV from 0 to -9999.

        This makes it play nicer with all the other functions
        and for masking out FABDEM extraneous pixels.

        Put in the COP30_hh_regridded_1s folder"""
        if tilename is None:
            files = self.retrieve_all_datafiles_list()
            output_dir = os.path.dirname(files[0]) + "_regridded_1s"
        else:
            files = [tilename]
            output_dir = os.path.dirname(files[0])

        # tempdir = os.path.join(output_dir, "temp")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        new_ndv = -9999

        for i,fn in enumerate(files):
            # First, regrid the dataset to 1s into the new file.
            if tilename_out is None:
                output_fn = os.path.join(output_dir, os.path.basename(fn).replace(".tif", "_1s.tif"))
            else:
                output_fn = tilename_out

            if os.path.exists(output_fn):
                if overwrite:
                    os.remove(output_fn)
                else:
                    if verbose:
                        print("{0}/{1}".format(i+1, len(files)), os.path.basename(output_fn), "already exists.")
                    continue

            ds = gdal.Open(fn, gdal.GA_ReadOnly)
            xleft, xres, _, ytop, _, yres = ds.GetGeoTransform()
            old_ndv = ds.GetRasterBand(1).GetNoDataValue()
            assert ((xres > 0) and (yres < 0))
            # In this particular case, the size of the x-resolution is 1-10s, while the y-resolution is always 1s
            assert abs(yres) <= abs(xres)

            gdal_translate_cmd = ["gdal_translate",
                                  "-r", "near",
                                  "-a_srs", "EPSG:4326",
                                  "-tr", repr(abs(yres)), repr(abs(yres)),
                                  fn, output_fn]
            if verbose_gdal:
                print(" ".join(gdal_translate_cmd))
            subprocess.run(gdal_translate_cmd, capture_output=not verbose_gdal)

            if old_ndv is None:
                old_ndv = 0.0

            # Second, in place, substitute all 0.0 values to -9999 values.
            # Use the same filename to just overwrite it in place.
            gdal_findreplace_cmd = ["gdal_findreplace.py",
                                    "-s_value", repr(old_ndv),
                                    "-t_value", repr(new_ndv),
                                    output_fn, output_fn]
            if verbose_gdal:
                print(" ".join(gdal_findreplace_cmd))
            subprocess.run(gdal_findreplace_cmd, capture_output=not verbose_gdal)

            # Third, set the ndv to be equal to this new value.
            gdal_edit_cmd = ["gdal_edit.py",
                             "-a_nodata", repr(new_ndv),
                             output_fn]
            if verbose_gdal:
                print(" ".join(gdal_edit_cmd))
            subprocess.run(gdal_edit_cmd, capture_output=not verbose_gdal)

            if verbose:
                print("{0}/{1}".format(i + 1, len(files)), os.path.basename(output_fn), end=" ")
                if os.path.exists(output_fn):
                    print("written.")
                else:
                    print("NOT written.")


    def create_trimmed_15s_tiles(self, overwrite=False, nprocs=18, verbose=True):
        """Create 15s tiles, that are trimmed to a realistic coastline without coastal-creep.

        Create a 15s average, a 15s nearest, then trim the 15s-average to the 15s-nearest mask. Save that to
        a 15s-average_trimmed file. DOES THIS WORK AT 15s ETOPO GRID-LINES?"""
        trimmed_1s_dir = self.config._abspath(self.config.soruce_datafiles_regridded_1s_directory)
        trimmed_1s_files = sorted([os.path.join(trimmed_1s_dir, fn) for fn in os.listdir(trimmed_1s_dir) if re.search("_00_DEM_1s.tif\Z", fn) is not None])

        near_dir_15 = self.config._abspath(self.config.source_datafiles_directory_15s_nearest)
        avg_dir_15 = self.config._abspath(self.config.source_datafiles_directory_15s_average)
        avg_src_grid_dir_15 = self.config._abspath(self.config.source_datafiles_directory_15s_average_source_grid)
        avg_trm_dir_15 = self.config._abspath(self.config.source_datafiles_directory_15s_average_trimmed)

        near_15_fnames = [os.path.join(near_dir_15, os.path.basename(fn).replace("1s.tif","15s_near.tif")) for fn in trimmed_1s_files]
        avg_15s_fnames = [os.path.join(avg_dir_15, os.path.basename(fn).replace("1s.tif","15s_avg.tif")) for fn in trimmed_1s_files]
        avg_src_grid_15s_fnames = [os.path.join(avg_src_grid_dir_15, os.path.basename(fn).replace("trimmed.tif","15s_avg_src_grid.tif")) for fn in trimmed_1s_files]
        avg_trm_15s_fnames = [os.path.join(avg_trm_dir_15, os.path.basename(fn).replace("1s.tif","15s_avg_trimmed.tif")) for fn in trimmed_1s_files]

        args = list(zip(trimmed_1s_files, near_15_fnames, avg_src_grid_15s_fnames, avg_15s_fnames, avg_trm_15s_fnames))
        kwargs = {"verbose": False,
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
                                       verbose=True):
        """From a single FABDEM tile, create a resampled and trimmed 15s version of it.

        Try it on the ETOPO grid and see how we do."""
        tile_1s_basename = os.path.basename(tilename_1s)
        n_tag = re.search("(?<=_)[NS](\d{2})(?=_)", tile_1s_basename).group()
        etopo_ymin = (-1 if (n_tag[0] == "S") else 1) * int(n_tag[1:])
        e_tag = re.search("(?<=_)[EW](\d{3})(?=_)", tile_1s_basename).group()
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
                                  "-te", str(etopo_xmin), str(etopo_ymin), str(etopo_xmax), str(etopo_ymax),
                                  "-ts", str(size), str(size),
                                  "-r", "near",
                                  "-co", "COMPRESS=DEFLATE",
                                  "-co", "PREDICTOR=2",
                                  tilename_1s,
                                  tilename_15s_near]
            try:
                subprocess.run(gdal_warp_near_cmd, capture_output=True)
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
                                          "-te", str(src_xmin), str(src_ymin), str(src_xmax), str(src_ymax),
                                          "-ts", str(size), str(size),
                                          "-r", "average",
                                          "-co", "COMPRESS=DEFLATE",
                                          "-co", "PREDICTOR=2",
                                          tilename_1s,
                                          tilename_15s_average_src_grid]
            try:
                subprocess.run(gdal_warp_avg_src_grid_cmd, capture_output=True)
            except (Exception, KeyboardInterrupt) as e:
                if os.path.exists(tilename_15s_average_src_grid):
                    os.remove(tilename_15s_average_src_grid)
                raise e


        # 3. Then, we'll fucking resample our half-pixel-offset grid using nearest neighbor to the ETOPO grid.
        # This is so weird but it works. And no, the 1/2-second offset in a 15s grid won't matter.
        if os.path.exists(tilename_15s_average_etopo_grid):
            os.remove(tilename_15s_average_etopo_grid)

        if not os.path.exists(tilename_15s_average_etopo_grid):
            gdal_warp_avg_cmd = ["gdalwarp",
                                 "-te", str(etopo_xmin), str(etopo_ymin), str(etopo_xmax), str(etopo_ymax),
                                 "-ts", str(size), str(size),
                                 "-r", "near",
                                 "-co", "COMPRESS=DEFLATE",
                                 "-co", "PREDICTOR=2",
                                 tilename_15s_average_src_grid,
                                 tilename_15s_average_etopo_grid]
            try:
                subprocess.run(gdal_warp_avg_cmd, capture_output=True)
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
            try:
                subprocess.run(gdal_warp_avg_trm_cmd, capture_output=True)
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
    cop = source_dataset_CopernicusDEM()
    # cop.regrid_and_reassign_ndv(tilename="/home/mmacferrin/Research/DATA/ETOPO/scratch_data/stacks_test/Copernicus_DSM_COG_10_N76_00_W084_00_DEM.tif",
    #                             verbose=True,
    #                             verbose_gdal=True)
    # cop.regrid_and_reassign_ndv(verbose=True,
    #                             verbose_gdal=False)
    # cop.regrid_and_reassign_ndv_parallel(nprocs=15)
    cop.create_trimmed_15s_tiles(overwrite = True, nprocs=18, verbose=True)