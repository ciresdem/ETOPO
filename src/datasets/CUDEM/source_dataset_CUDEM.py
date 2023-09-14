# -*- coding: utf-8 -*-

"""Source code for the CUDEM ETOPO source dataset class."""

import os
import shutil

import numpy
from osgeo import gdal, ogr
import geopandas
import random
import subprocess
import time

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
        """Do a quick traverse through the directory to make sure each CUDEM tile has a _egm2008.tif equivalent."""
        basedir = self.config.source_datafiles_directory
        fnames_list = utils.traverse_directory.list_files(basedir,
                                                          regex_match=r"ncei([\w\-\.]+)v\d(_bathy)*\.tif\Z",
                                                          include_base_directory=True)

        # print(fnames_list)

        for fname in fnames_list:
            fname_converted_1 = os.path.splitext(fname)[0] + "_egm2008.tif"
            fname_converted_2 = os.path.join(os.path.dirname(fname_converted_1), "converted", os.path.basename(fname_converted_1))

            if not (os.path.exists(fname_converted_1) or os.path.exists(fname_converted_2)):
                print(fname, "does NOT have an accompanying _egm2008 file.")

        print("Done.")
        return

    def remove_river_from_N39W077(self, overwrite=False):
        """The tile N39W077, in a river at the end of Chesapeake Bay, creates an artifact in the CRM 1s tiles where
        the river just 'cuts off' at a corner as it leaves the CUDEM tile and enters the USGS TNM layers. The CUDEMs
        use a default bathy "flat" elevation in the river, which (after converting to EGM2008) give all of those elevations
        a slight-negative value between -1 and 0, upstream from a dam in the river. If we remove all those pixels upstream
        from the dam, then it will get rid of that artifact and replace it with "surface" values from USGS TNM, and just
        have a wall at the location of the dam, which is acceptable (because it is, actually, a physical wall there)."""
        river_polygon_shp = self.config._abspath(self.config.n39x75_w076x25_river_polygon_shp)
        print(river_polygon_shp, os.path.exists(river_polygon_shp))

        filenames = self.get_geodataframe().filename.to_list()
        input_raster_paths = [fn for fn in filenames if fn.find("n39x75_w076x25") > -1]
        # print(input_raster_path)
        assert len(input_raster_paths) == 1
        input_raster_path = input_raster_paths[0]

        # Open input raster
        input_raster = gdal.Open(input_raster_path, gdal.GA_ReadOnly)
        # input_band = input_raster.GetRasterBand(1)

        mask_raster_path = os.path.join(os.path.splitext(river_polygon_shp)[0] + "_polygon_mask.tif")

        if not os.path.exists(mask_raster_path) or overwrite:
            input_gt = input_raster.GetGeoTransform()
            xmin, xres, _, ymax, _, yres = input_gt
            xsize, ysize = input_raster.RasterXSize, input_raster.RasterYSize
            xmax = xmin + (xres * xsize)
            ymin = ymax + (yres * ysize)

            gdal_cmd = ['gdal_rasterize',
                        '-burn', '1',
                        '-a_nodata', '0',
                        '-te', repr(xmin), repr(ymin), repr(xmax), repr(ymax),
                        '-tr', repr(xres), repr(abs(yres)),
                        '-co', "COMPRESS=DEFLATE",
                        river_polygon_shp, mask_raster_path]

            print(" ".join(gdal_cmd))
            subprocess.run(gdal_cmd)
            # Read the shapefile
            # shapefile = ogr.Open(river_polygon_shp)
            # layer = shapefile.GetLayer()
            #
            # # Create output raster
            # output_driver = gdal.GetDriverByName('GTiff')
            # mask_raster = output_driver.Create(output_raster_path, input_raster.RasterXSize, input_raster.RasterYSize, 1,
            #                                      gdal.GDT_Byte, options=['COMPRESS=DEFLATE'])
            # mask_raster.SetProjection(input_raster.GetProjection())
            # mask_raster.SetGeoTransform(input_raster.GetGeoTransform())
            #
            # # Set output band
            # output_band = mask_raster.GetRasterBand(1)
            # output_band.WriteArray(nupmy.zeros([input_raster.RasterYSize, input_raster.RasterXSize], dtype=numpy.uint8))
            # output_band.SetNoDataValue(0)
            # output_band.FlushCache()
            #
            # # Rasterize polygon
            # gdal.RasterizeLayer(mask_raster, [1], layer, burn_values=[1])
            # mask_raster.FlushCache()

            # Trim to extent
            # gdal.Warp(output_raster_path, output_raster, format='GTiff', cutlineDSName=river_polygon_shp, cropToCutline=True,
            #           options=['COMPRESS=DEFLATE'])

            # Clean up
            # input_raster = None
            # mask_raster = None
            # shapefile = None

            print("Rasterization and trimming completed successfully.")

        temp_output_dem = os.path.splitext(input_raster_path)[0] + "_MASKED_TEMP.tif"
        orig_dem_file = os.path.splitext(input_raster_path)[0] + "_ORIGINAL.tif"
        if os.path.exists(orig_dem_file):
            print("Original already exists. Will not overwrite (to preserve original copy). Exiting.")
            return

        # Create a copy of the file into the temp destination.
        shutil.copyfile(input_raster_path, temp_output_dem)
        print(temp_output_dem, "created.")

        # Open and read the data, in update mode.
        dem_ds = gdal.Open(temp_output_dem, gdal.GA_Update)
        dem_band = dem_ds.GetRasterBand(1)
        dem_array = dem_band.ReadAsArray()
        # Mask out all values between -1 and 0 (in this case, the river values)
        dem_mask_btw_n1_0 = (dem_array >= -1) & (dem_array < 0)
        dem_ndv = dem_band.GetNoDataValue()

        # Open and read the polygon mask data
        mask_ds = gdal.Open(mask_raster_path, gdal.GA_ReadOnly)
        mask_array = mask_ds.GetRasterBand(1).ReadAsArray().astype(bool)
        mask_ds = None

        # Combine the polygon mask and the elevation mask to get our river cells to mask out.
        river_mask = dem_mask_btw_n1_0 & mask_array

        # Mask out the river cells, put it in the output band.
        dem_array[river_mask] = dem_ndv
        dem_band.WriteArray(dem_array)

        # Write the changes to disk. Pause for a fraction of a second to make sure they're written out.
        # (Gawd, gdal is wonky this way.)
        dem_ds.FlushCache()
        dem_ds = None
        dem_band = None
        time.sleep(0.1)

        print(os.path.basename(temp_output_dem), "written with masked values.")

        # Move the files around to put the masked version now in the original filename, and save the original file name.
        print(os.path.basename(input_raster_path), "->", os.path.basename(orig_dem_file))
        shutil.move(input_raster_path, orig_dem_file)
        print(os.path.basename(temp_output_dem), "->", os.path.basename(input_raster_path))
        shutil.move(temp_output_dem, input_raster_path)

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
    cudem.get_geodataframe()
    # cudem.convert_vdatum()

    # cudem.remove_river_from_N39W077(overwrite=False)
    # cudem.measure_navd88_vs_egm2008_elevs()
    # cudem.reproject_tiles_from_nad83(overwrite=False)
    # cudem.delete_empty_tiles(check_start=0, recreate_if_deleted=True)

    # cudem.check_if_all_tiles_are_converted()