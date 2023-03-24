# -*- coding: utf-8 -*-

"""Source code for the CUDEM ETOPO source dataset class."""

import os
import numpy
from osgeo import gdal
import geopandas
import random

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset
import etopo.convert_vdatum
import utils.traverse_directory
import utils.configfile

class source_dataset_CUDEM(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "CUDEM_config.ini" )):
        """Initialize the CUDEM source dataset object."""

        super(source_dataset_CUDEM, self).__init__("CUDEM", configfile)

    def measure_navd88_vs_egm2008_elevs(self):
        """Give a distribution of how much elevation was changed when we converted from NAVD88 to EGM2008.

        (Specifically for CONUS tiles in the southeast US, covering the CRMs)."""
        etopo_config = utils.configfile.config()
        crm_gdf = geopandas.read_file(etopo_config._abspath(etopo_config.crm_tiles_outline_shapefile)).geometry
        assert len(crm_gdf) == 1
        crm_polygon = crm_gdf.geometry[0]

        fnames = self.retrieve_list_of_datafiles_within_polygon(crm_polygon, crm_gdf.crs, resolution_s = 1, return_fnames_only=True)
        random.shuffle(fnames)

        assert numpy.all([fn.find("_egm2008_epsg4326.tif") > 0 for fn in fnames])
        for i,fname in enumerate(fnames):
            egm_2008_fname = fname.replace("_epsg4326", "")
            orig_fname = egm_2008_fname.replace("/converted/", "/").replace("_egm2008", "")
            try:
                assert os.path.exists(egm_2008_fname)
                assert os.path.exists(orig_fname)
            except AssertionError as e:
                print("Original file:", orig_fname)
                print("EGM2008 file:", egm_2008_fname)
                print("WGS84 file:", fname)
                print("One of these is not found.")
                raise e

            egm_ds = gdal.Open(egm_2008_fname, gdal.GA_ReadOnly)
            egm_band = egm_ds.GetRasterBand(1)
            egm_array = egm_band.ReadAsArray()

            navd_ds = gdal.Open(orig_fname, gdal.GA_ReadOnly)
            navd_band = navd_ds.GetRasterBand(1)
            navd_array = navd_band.ReadAsArray()

            navd_band = None
            navd_ds = None
            egm_band = None
            egm_ds = None

            diffs = egm_array - navd_array
            mind = numpy.amin(diffs)
            maxd = numpy.amax(diffs)
            meand = numpy.mean(diffs)
            stdd = numpy.std(diffs)

            print("{0}/{1}".format(i+1, len(fnames)), os.path.basename(orig_fname), meand, "+/-", stdd, "min", mind, "max", maxd)
        return

    def delete_empty_tiles(self, recreate_if_deleted=True, delete_if_any_nodata=True, check_start=0, check_end=None, verbose=True):
        """For some fucking reason, some of the CUDEM converted tiles don't have any valid data in them.

        Find those files and re-convert them.

        If delete_if_any_nodata, then delete if there's even a single NDV value in there.
        Otherwise, just delete if they're all NDV."""
        gdf = self.get_geodataframe()

        ndv_tiles_count = 0

        for i,row in enumerate(gdf.iterrows()):
            if (i+1) < check_start:
                continue
            elif check_end and ((i+1) > check_end):
                continue

            _, row = row

            print("{0}/{1} {2}".format(i + 1, len(gdf), row.filename[len(os.path.dirname(
                os.path.dirname(os.path.dirname(row.filename)))) + 1:]), end="")

            # The CRM 1s tiles (at least the one for Hawaii) do NOT need to be free to NDV.
            if row.filename.find("/CRMs_1s/") > -1:
                # The 2 crm tiles are fine. Move along.
                print(" is a CRM tile. It's fine.")
                continue

            # Check to see if the vertically-converted file (not just the projected one) also has NDVs. If so, delete it too.
            fname_egm2008 = row.filename.replace("_epsg4326", "")
            fname_original = fname_egm2008.replace("_egm2008", "").replace("/converted/", "/")

            if os.path.exists(row.filename):
                ds = gdal.Open(row.filename, gdal.GA_ReadOnly)
                band = ds.GetRasterBand(1)
                ndv = band.GetNoDataValue()
                array = band.ReadAsArray()
                if (numpy.any(array == ndv) and delete_if_any_nodata) or numpy.all(array == ndv):
                    print(" contains {0:0.2f}% NODATA.".format(numpy.count_nonzero(array==ndv) * 100. / array.size))
                    print(" "*3,"Removing", os.path.basename(row.filename))
                    os.remove(row.filename)
                    ndv_tiles_count += 1
                else:
                    print(" fine.")
                    continue

            if os.path.exists(fname_egm2008):
                ds_egm_2008 = gdal.Open(fname_egm2008, gdal.GA_ReadOnly)
                band_egm_2008 = ds_egm_2008.GetRasterBand(1)
                ndv_egm_2008 = band_egm_2008.GetNoDataValue()
                array_egm_2008 = band_egm_2008.ReadAsArray()
                if (numpy.any(array_egm_2008 == ndv_egm_2008) and delete_if_any_nodata) or numpy.all(array_egm_2008 == ndv):
                    print(" " * 3, "Removing", os.path.basename(fname_egm2008))
                    os.remove(fname_egm2008)

            if not recreate_if_deleted:
                break

            if not os.path.exists(fname_egm2008):
                assert os.path.exists(fname_original)
                starting_vdatum = get_cudem_original_vdatum_from_file_path(fname_original)
                print(" " * 3, "Re-converting", os.path.basename(fname_original))
                etopo.convert_vdatum.convert_vdatum(fname_original,
                                                    fname_egm2008,
                                                    input_vertical_datum=starting_vdatum,
                                                    output_vertical_datum="egm2008",
                                                    verbose=False)

                ndv_tiles_count += 1

            if not os.path.exists(row.filename):
                print(" " * 3, "Re-projecting", os.path.basename(fname_egm2008))
                self.reproject_tiles_from_nad83(range_start = i, range_stop=i+1, overwrite=False, verbose=False)

            out_ds = gdal.Open(row.filename, gdal.GA_ReadOnly)
            out_band = out_ds.GetRasterBand(1)
            out_ndv = out_band.GetNoDataValue()
            out_array = out_band.ReadAsArray()
            print(" " * 3, os.path.basename(row.filename), "has", "{0:0.2f}%".format(numpy.count_nonzero(out_array == out_ndv) * 100. / out_array.size), "nodata.")

        if verbose:
            print("{0} of {1} tiles found containinng NDVs.".format(ndv_tiles_count, len(gdf)))

    def check_if_all_tiles_are_converted(self):
        """Do a quick traverse through the directory to make sure each CUDEM tile has a _egm2008_epsg4326.tif equivalent."""
        basedir = self.config.source_datafiles_directory
        fnames_list = utils.traverse_directory.list_files(basedir, regex_match="ncei([\w\-\.]+)_v\d\.tif\Z", include_base_directory=True)

        for fname in fnames_list:
            fname_converted_1 = os.path.splitext(fname_converted)[0] + "_egm2008.tif"
            fname_converted_2 = os.path.join(os.path.dirname(fname_converted_1), "converted", os.path.basename(fname_converted_1))

            if not (os.path.exists(fname_converted_1) or os.path.exists(fname_converted_2)):
                print(fname, "does NOT have an accompanying _egm2008_epsg4326 file.")

        print("Done.")
        return

def get_cudem_original_vdatum_from_file_path(file_path):
    """Quick little uility for getting the original vetical datum from the folder or filename of the CUDEM tile."""
    vertical_datum_lookup = {"/AmericanSamoa/": "asvd02",
                             "/CNMI/": "nmvd03",
                             "/CONUS/": "navd88",
                             "/CONUS_Sandy/": "navd88",
                             "/Guam/": "guvd04",
                             "/Hawaii/": "msl",
                             "/Puerto_Rico/": "prvd02",
                             "/US_Virgin_Islands/": "prvd02",
                             "crm1_hawaii_": "msl",
                             "crm1_prvi_": "prvd02"}

    for lookup_key in vertical_datum_lookup.keys():
        if file_path.find(lookup_key) > -1:
            return vertical_datum_lookup[lookup_key]

    raise ValueError("Unhandled vertical datum lookup for " + file_path)


# If the Geopackage database doesn't exist (i.e. it's been deleted after some new files were created or added), this will create it.
if __name__ == "__main__":
    # gdf = source_dataset_CUDEM().get_geodataframe()
    cudem = source_dataset_CUDEM()
    cudem.convert_vdatum()
    # cudem.measure_navd88_vs_egm2008_elevs()
    # cudem.reproject_tiles_from_nad83(overwrite=False)
    # cudem.delete_empty_tiles(check_start=0, recreate_if_deleted=True)

    # cudem.check_if_all_tiles_are_converted()