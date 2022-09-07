# -*- coding: utf-8 -*-

"""Source code for the GEBCO ETOPO source dataset class."""

import os
import re
import time

import numpy
from osgeo import gdal

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset

class source_dataset_GEBCO(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "GEBCO_config.ini" )):
        """Initialize the GEBCO source dataset object."""

        etopo_source_dataset.ETOPO_source_dataset.__init__(self, "GEBCO", configfile)

    def get_tid_filename(self, filename, check_if_exists=True):
        """Given the tilename of a GEBCO tile, find the corresponding tid (type ID) filename that pairs with it."""
        tid_dirname = self.config._abspath(self.config.source_datafiles_tid_directory)
        filename_short = os.path.split(filename)[1]
        assert filename_short.find("gebco_2022_") == 0
        locstr_match = re.search("n(\d{1,2})\.0_s(\-?)(\d{1,2})\.0_w(\-?)(\d{1,3})\.0_e(\-?)(\d{1,3})\.0\.tif\Z", filename_short)
        # Check to make sure the match we just fit above appears right after "gebco_2022_" at the start of the filename.
        assert locstr_match != 0 and len("gebco_2022_") == locstr_match.span()[0]

        fname_out = os.path.join(tid_dirname, "gebco_2022_tid_" + locstr_match.group())
        if check_if_exists:
            assert os.path.exists(fname_out)
        return fname_out


    def mask_out_bad_values_single_tile(self, tilename, tid_badvalues=[40], append="_masked", verbose=True):
        """For a single tile, find all values where the TID has the tid_badvalue, and mask it out.

        Save the output file to a filename with the 'append' post-fix on it."""
        # Get the name of the TypeID filename associated with this tile.
        tid_fname = self.get_tid_filename(tilename)

        if type(tid_badvalues) in (int, float):
            tid_badvalues = [int(tid_badvalues)]

        # Derive the output file name.
        out_base, out_ext = os.path.splitext(tilename)
        out_fname = out_base + append + out_ext

        tile_dset = gdal.Open(tilename, gdal.GA_ReadOnly)
        tile_band = tile_dset.GetRasterBand(1)
        tile_array = tile_band.ReadAsArray()
        tile_ndv = tile_band.GetNoDataValue()

        tid_dset = gdal.Open(tid_fname, gdal.GA_ReadOnly)
        tid_array = tid_dset.GetRasterBand(1).ReadAsArray()
        # Generate a mask of bad values based on all the values given.
        bad_values_mask = numpy.zeros(tile_array.shape, dtype=bool)
        print("\t{0:,} values start off 'bad.'".format(numpy.count_nonzero(bad_values_mask)))
        for bad_tid in tid_badvalues:
            this_band_tid_mask = (tid_array == bad_tid)
            print("\t{0:,}".format(numpy.count_nonzero(this_band_tid_mask)), "of", ("{0:,}".format(bad_values_mask.size)), "==", bad_tid)
            bad_values_mask = bad_values_mask | this_band_tid_mask

        # Create a copy of the array. Set all bad tid values to NDV
        out_array = tile_array.copy()
        out_array[bad_values_mask] = tile_ndv

        if os.path.exists(out_fname):
            os.remove(out_fname)

        tile_out_ds = tile_dset.GetDriver().CreateCopy(out_fname, tile_dset)
        # First, write the new file to disk.
        tile_out_ds = None
        # Then, open it up with gdal.GA_Update, and change the array, recompute stats, and output.
        while not os.path.exists(out_fname):
            time.sleep(0.01)
        assert os.path.exists(out_fname)

        tile_out_ds = gdal.Open(out_fname, gdal.GA_Update)
        tile_out_band = tile_out_ds.GetRasterBand(1)
        tile_out_band.WriteArray(out_array)
        tile_out_band.SetNoDataValue(tile_ndv)
        tile_out_band.SetStatistics(*tile_out_band.GetStatistics(0,1))
        tile_out_ds.FlushCache()
        tile_out_ds = None

        if verbose:
            print(os.path.split(out_fname)[1], "written with {0:.2f}% NoData.".format(100. * numpy.count_nonzero(bad_values_mask) / out_array.size))

        return tid_fname

    def mask_out_bad_values_all_tiles(self, tid_badvalues=[40], append="_masked", verbose=True):
        """Go through all the tiles, mask out bad values. Tell us how much was masked out."""
        gdf = self.get_geodataframe(verbose=verbose)
        for i,row in gdf.iterrows():
            fname = row.filename
            self.mask_out_bad_values_single_tile(fname, tid_badvalues=tid_badvalues, append=append, verbose=verbose)
        return

if __name__ == "__main__":
    gebco = source_dataset_GEBCO()
    gebco.mask_out_bad_values_all_tiles()