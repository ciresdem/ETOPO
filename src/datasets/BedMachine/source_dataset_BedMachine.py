# -*- coding: utf-8 -*-

"""Source code for the BedMachine ETOPO source dataset class."""

import os
import subprocess
from osgeo import gdal
import numpy
import pyproj
import shapely
import matplotlib.pyplot as plt
import re

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir

import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset
import datasets.dataset_geopackage as dataset_geopackage
import utils.traverse_directory
import etopo.convert_vdatum

class source_dataset_BedMachine(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""

    def __init__(self,
                 configfile=os.path.join(THIS_DIR, "BedMachine_config.ini")):
        """Initialize the BedMachine source dataset object."""

        etopo_source_dataset.ETOPO_source_dataset.__init__(self, "BedMachine", configfile)

    def convert_and_reproject_netcdf_to_wgs84_egm2008_geotiffs(self,
                                                               resolution_s=15,
                                                               overwrite=False,
                                                               verbose=True):
        """For both Greenland and Antarctica, transform the horizontal and veritcal datums to WGS84+EGM2008.
        Convert it to the highest lat/lon resolution that the original data is in (reprojected to WGS84).

        After running this once, we can point the .config files to the converted GeoTIFFs rather than the netcdfs."""
        # Get the .nc files in the data directory.
        # If we sort it, we know we'll do Antarctica first.
        tilenames = sorted(
            utils.traverse_directory.list_files(self.config._abspath(self.config.source_datafiles_directory),
                                                regex_match=r"BedMachine([\w\-]+)\.nc\Z"))
        variable_names = ["bed", "dataid", "errbed", "geoid", "mask", "source", "surface", "thickness"]
        etopo_gpkg = dataset_geopackage.ETOPO_Geopackage(resolution=resolution_s)

        converted_files_list = []

        # First, loop through both .nc files and pull out all the layers. This is easy to do.
        for tname in tilenames:

            # First, convert the netcdf elevations to geotiff.
            for var in variable_names:

                # Antarctica doesn't have the "dataid" field. Skip it.
                if (tname.find("BedMachineAntarctica") >= 0) and (var == "dataid"):
                    continue

                gtif_name = os.path.splitext(tname)[0] + ".{0}.tif".format(var)
                if os.path.exists(gtif_name) and not overwrite:
                    continue
                cmd_flags = ["gdal_translate",
                             'NETCDF:"{0}":{1}'.format(tname, var),
                             gtif_name,
                             "-co", "COMPRESS=DEFLATE",  # Compression options.
                             "-co", "PREDICTOR=2"]
                if verbose:
                    print(" ".join(cmd_flags))
                subprocess.run(cmd_flags,
                               cwd=self.config.source_datafiles_directory,
                               capture_output=True)

                assert os.path.exists(gtif_name)
                if verbose:
                    print(gtif_name, "written.")

            # Second, look at the masks.
            mask_name = os.path.splitext(tname)[0] + ".mask.tif"
            mask_ds = gdal.Open(mask_name, gdal.GA_ReadOnly)
            mask_array = mask_ds.GetRasterBand(1).ReadAsArray()
            mask_ds = None  # Close the file & deallocate resources.
            # The mask values are: 0: ocean
            #                      1: ice-free land
            #                      2: grounded ice
            #                      3: floating ice
            #                      4: non-Greenland land (in Greenland only)

            for var in ['bed', 'surface']:
                gtif_name = os.path.splitext(tname)[0] + ".{0}.tif".format(var)
                gtif_masked_name = os.path.splitext(gtif_name)[0] + "_wgs84_masked.tif"
                gtif_geoid_masked_name = os.path.splitext(gtif_name)[0] + "_geoid_masked.tif"
                elev_ds = gdal.Open(gtif_name, gdal.GA_ReadOnly)
                assert elev_ds is not None

                if os.path.exists(gtif_masked_name) and os.path.exists(gtif_geoid_masked_name) and (not overwrite):
                    continue
                else:
                    # Create a copy of the source dataset, into the _wgs84_masked dataset. We'll modify the values below.
                    driver = elev_ds.GetDriver()
                    if os.path.exists(gtif_masked_name):
                        if overwrite:
                            os.remove(gtif_masked_name)
                    if not os.path.exists(gtif_masked_name):
                        dst_ds = driver.CreateCopy(gtif_masked_name, elev_ds)
                        dst_ds = None

                    if os.path.exists(gtif_geoid_masked_name):
                        if overwrite:
                            os.remove(gtif_geoid_masked_name)
                    if not os.path.exists(gtif_geoid_masked_name):
                        dst_ds_geoid = driver.CreateCopy(gtif_geoid_masked_name, elev_ds)
                        dst_ds_geoid = None

                elev_band = elev_ds.GetRasterBand(1)
                elev_ndv = elev_band.GetNoDataValue()
                elev_array = elev_band.ReadAsArray()
                assert elev_ndv is not None
                # print("NDV:", elev_ndv)

                geoid_fname = os.path.splitext(tname)[0] + ".geoid.tif"
                geoid_ds = gdal.Open(geoid_fname, gdal.GA_ReadOnly)
                geoid_array = geoid_ds.GetRasterBand(1).ReadAsArray()
                # Convert the elevation array to WGS84 elevations (from the funky EIGEN-6C4 geoid)
                # We'll convert to EGM2008 later.
                elev_array_wgs84 = elev_array + geoid_array
                elev_array_geoid = elev_array

                mask_invalid = None

                if tname.find("BedMachineGreenland") >= 0:
                    # Get only data that isn't nodata or not part of Greenland (omit Ellesmere Island)
                    mask_invalid = (mask_array == 4)
                if tname.find("Antarctica") >= 0:
                    # We aren't going to use a surface dataset for Antarctica, since Copernicus is higher-res there.
                    # Simply mask out everything as invalid for the surface dataset. Just use the bed.
                    if var == 'surface':
                        mask_invalid = numpy.ones(mask_array.shape, dtype=bool)
                    # Filter out ocean basin in Antarctica. GEBCO is better there.
                    #
                    else:
                        # For the bed, we just want to use 2: grounded ice and 3: floating ice
                        assert var == 'bed'
                        mask_invalid = (mask_array < 2)

                elev_array_wgs84[mask_invalid] = elev_ndv
                elev_array_geoid[mask_invalid] = elev_ndv

                # Put the masked values into the arrays, in both wgs84 and geoid elevs.
                masked_ds = gdal.Open(gtif_masked_name, gdal.GA_Update)
                masked_band = masked_ds.GetRasterBand(1)

                masked_band.WriteArray(elev_array_wgs84)
                # Write to disk.
                masked_band = None
                masked_ds = None

                masked_ds_geoid = gdal.Open(gtif_geoid_masked_name, gdal.GA_Update)
                masked_band_geoid = masked_ds_geoid.GetRasterBand(1)
                masked_band_geoid.WriteArray(elev_array_geoid)
                # Write to disk.
                masked_band_geoid = None
                masked_ds = None

                if verbose:
                    print(os.path.split(gtif_masked_name)[1], "written in WGS84 heights with {0:4.1f}% valid data.".format(
                        numpy.count_nonzero(~mask_invalid) / mask_invalid.size * 100))

            # Find the ETOPO tiles that overlap this polygon.
            # Let's convert this polygon to WGS84 to make this not break at all (and make fer' damned sure Antarctica is wrapped correctly).
            for var in ['bed', 'surface']:
                gtif_wgs84_name = os.path.splitext(tname)[0] + ".{0}_wgs84_masked.tif".format(var)
                gtif_geoid_name = os.path.splitext(tname)[0] + ".{0}_geoid_masked.tif".format(var)

                polygon, proj, xleft, ytop, xres, yres, xsize, ysize = \
                    etopo_gpkg.create_outline_geometry_of_geotiff(gtif_wgs84_name)
                # print(polygon.exterior.xy)
                # plt.plot(*polygon.exterior.xy)
                if tname.find("Greenland") >= 0:
                    proj = pyproj.crs.CRS.from_epsg(3413)
                else:
                    assert tname.find("Antarctica") >= 0
                    proj = pyproj.crs.CRS.from_epsg(3031)
                wgs84_crs = pyproj.crs.CRS.from_epsg(4326)

                projector = pyproj.Transformer.from_crs(proj,
                                                        wgs84_crs,
                                                        always_xy=True).transform
                polygon_interpolated = shapely.geometry.Polygon(
                    [polygon.boundary.interpolate(i, normalized=True) for i in numpy.linspace(0, 1, 401)])

                polygon_to_use = shapely.ops.transform(projector, polygon_interpolated)

                if tname.find("Antarctica") >= 0:
                    # For antarctica, we must wrap the projected polygon around the poles.
                    lons, lats = [numpy.array(a) for a in polygon_to_use.exterior.xy]
                    # print(type(lons), type(lats))
                    # All the latitudes should be negative. Sanity check.
                    assert numpy.all(lats < 0.0)
                    # Find the min and max longitudes, closest to -180 and +180
                    min_lon_i = numpy.argmin(lons)
                    max_lon_i = numpy.argmax(lons)
                    # This is going to the right (eastward) and wraps, so the max should be the very next number after the min.
                    assert min_lon_i == (max_lon_i + 1)
                    # print(min_lon_i, max_lon_i)
                    # Wrap the rectangle to the edge (180 e), down over the pole (-90 s), then back up the west edge (-180 w)
                    new_lons = numpy.concatenate((lons[:(max_lon_i + 1)],
                                                  [180., 180., -180., -180.],
                                                  lons[min_lon_i:]))
                    new_lats = numpy.concatenate((lats[:(max_lon_i + 1)],
                                                  numpy.array([lats[max_lon_i], -90, -90, lats[min_lon_i]]),
                                                  lats[min_lon_i:]))
                    # print(polygon_to_use)
                    polygon_to_use = shapely.geometry.Polygon(zip(new_lons, new_lats))
                    # print(polygon_to_use)

                # plt.plot(*polygon_to_use.exterior.xy)
                # print(polygon_to_use.exterior.xy)

                etopo_gdf_subset = etopo_gpkg.subset_by_polygon(polygon_to_use,
                                                                polygon_crs=wgs84_crs,
                                                                resolution_s=resolution_s,
                                                                verbose=verbose)

                # Reproject out to all the tiles.
                for i, row in etopo_gdf_subset.iterrows():
                    # gtif_name = gtif_wgs84_name
                    # print(row.xleft, row.xres, row.ytop, row.yres)
                    # print(str(proj.to_epsg()))
                    ybottom = round(row.ytop + (row.yres * row.ysize))
                    xright = round(row.xleft + (row.xres * row.xsize))
                    # gtif_fname_only = os.path.split(gtif_name)[1]

                    for gtif_name in (gtif_wgs84_name, gtif_geoid_name):

                        outfile_basename = "BedMachine" + \
                                           ("Greenland" if tname.find("Greenland") >= 0 else "Antarctica") + \
                                           "_{0}s_".format(resolution_s) + \
                                           ("wgs84_" if (gtif_name == gtif_wgs84_name) else "eigen6C4_") + \
                                           ("S" if (ybottom < 0) else "N") + \
                                           "{0:02d}".format(abs(ybottom)) + \
                                           ("W" if (row.xleft < 0) else "E") + \
                                           "{0:03d}".format(abs(int(row.xleft))) + \
                                           ".{0}.tif".format(var)

                        if verbose:
                            print(outfile_basename)
                        # continue

                        projected_name = os.path.join(os.path.dirname(gtif_name),
                                                      "{0}s".format(int(resolution_s)),
                                                      "projected",
                                                      outfile_basename)
                        if os.path.exists(projected_name) and overwrite:
                            os.remove(projected_name)

                        if not os.path.exists(projected_name):
                            # noinspection IncorrectFormatting
                            gdal_cmd = ["gdalwarp",
                                        "-s_srs", "EPSG:" + str(proj.to_epsg()),
                                        "-t_srs", "EPSG:" + str(wgs84_crs.to_epsg()),
                                        "-te", str(row.xleft),
                                        str(ybottom),
                                        str(xright),
                                        str(row.ytop),
                                        "-tr", str(row.xres),
                                        str(row.yres),
                                        "-r", "bilinear",
                                        gtif_name,
                                        projected_name,
                                        "-co", "COMPRESS=DEFLATE",  # Compression options.
                                        "-co", "PREDICTOR=2"
                                        ]
                            print(" ".join(gdal_cmd))
                            subprocess.run(gdal_cmd,
                                           cwd=self.config._abspath(self.config.source_datafiles_directory),
                                           capture_output=True)
                            print()
                        assert os.path.exists(projected_name)

                        base, ext = os.path.splitext(outfile_basename)
                        converted_fname = os.path.join(os.path.dirname(gtif_name),
                                                       "{0}s".format(int(resolution_s)),
                                                       "converted",
                                                       base + "_egm2008" + ext)

                        if os.path.exists(converted_fname) and overwrite:
                            os.remove(converted_fname)

                        if gtif_name == gtif_wgs84_name:
                            # Then, we'll convert it to EGM2008 elevations.
                            qualifier_text = "already exists."
                            if not os.path.exists(converted_fname) or overwrite:
                                etopo.convert_vdatum.convert_vdatum(projected_name,
                                                                    converted_fname,
                                                                    input_vertical_datum="wgs84",
                                                                    output_vertical_datum="egm2008",
                                                                    cwd=os.path.dirname(converted_fname),
                                                                    verbose=False)
                                qualifier_text = "written."
                            if os.path.exists(converted_fname):
                                # If this worked, but outta the loop, we don't need to use the geoid version.
                                converted_files_list.append(converted_fname)
                                if verbose:
                                    print(os.path.split(converted_fname)[1], qualifier_text)
                                break
                            else:
                                print("ERROR: Could not create", os.path.split(converted_fname)[1])
                        else:
                            # Instead, let's create it from the EIGEN-6C4 geoid, as a stand-in for now until we figure this out.
                            # The elevations will be very close to each other.
                            assert gtif_name == gtif_geoid_name
                            qualifier_text = "already exists."
                            # If we've gone to the geoid version, just copy it over, no need to convert the vdatum.
                            if not os.path.exists(converted_fname) or overwrite:
                                projected_ds = gdal.Open(projected_name, gdal.GA_ReadOnly)
                                driver = projected_ds.GetDriver()
                                converted_ds = driver.CreateCopy(converted_fname, projected_ds)
                                # Write them to disk, the weird GDAL-way (by de-referencing .
                                projected_ds = None
                                converted_ds = None
                                qualifier_text = "written."
                            if os.path.exists(converted_fname):
                                # If this worked, but outta the loop, we don't need to use the geoid version.
                                converted_files_list.append(converted_fname)
                                if verbose:
                                    print(os.path.split(converted_fname)[1], qualifier_text)
                                break
                            else:
                                print("ERROR: Could not create", os.path.split(converted_fname)[1])


                        # print("ERROR: Could not create", os.path.split(converted_fname)[1])
                        # print("DEBUG: Try again with verbose on.")
                        # etopo.convert_vdatum.convert_vdatum(projected_name,
                        #                                     converted_fname,
                        #                                     input_vertical_datum="wgs84",
                        #                                     output_vertical_datum="egm2008",
                        #                                     cwd=os.path.dirname(converted_fname),
                        #                                     verbose=True)


        # Go through the list of converted files and get rid of ones that have no good data. This happens from the
        # Masking above.
        self.clear_empty_tiles(converted_files_list, verbose=verbose)
        # for fname in converted_files_list:
        #     if not os.path.exists(fname):
        #         continue
        #     ds = gdal.Open(fname, gdal.GA_ReadOnly)
        #     band = ds.GetRasterBand(1)
        #     array = band.ReadAsArray()
        #     ndv = band.GetNoDataValue()
        #     # If everything is nodata, just get rid of the file.
        #     if numpy.all(array == ndv):
        #         if verbose:
        #             print("Removing", os.path.split(fname)[1], "w/ no valid data.")
        #         os.remove(fname)
        #         continue
        #
        #     del band
        #     del ds

        return

    def clear_empty_tiles(self, file_list, resolution_s = 15, verbose: bool = True):
        """Get rid of tiles that only have nodata values in them."""
        # FOR NOW, just handle a list of files. If we need to traverse a directory later, do it then.
        # if type(file_list_or_dirname) == str:
        #     if os.path.isdir(file_list_or_dirname):
        #         # Get the lists of tiles from BedMachine_Bed and BedMachine_Surface
        #         bm_bed_obj = etopo_source_dataset.get_source_dataset_object("BedMachine_Bed", verbose=verbose)
        #         bm_srf_obj = etopo_source_dataset.get_source_dataset_object("BedMachine_Surface", verbose=verbose)
        #         # First, get the dataframe from each one.
        #         bm_bed_tiles = bm_bed_obj.retrieve_all_datafiles_list(resoultion_s=resolution_s,
        #                                                               verbose=verbose)
        #         bm_srf_tiles = bm_srf_obj.retrieve_all_datafiles_list(resoultion_s=resolution_s,
        #                                                               verbose=verbose)
        for fname in file_list:
            ds = gdal.Open(fname)
            if ds is None:
                if verbose:
                    print("Could not open", fname)
                continue
            band = ds.GetRasterBand(1)
            array = band.ReadAsArray()
            ndv = band.GetNoDataValue()
            # If everything is nodata, just get rid of the file.
            if numpy.all(array == ndv):
                if verbose:
                    print("Removing", os.path.split(fname)[1], "w/ no valid data.")
                os.remove(fname)
                continue
            elif verbose:
                print(os.path.split(fname)[1], "contains valid data.")

            del band
            del ds


if __name__ == "__main__":
    bm = source_dataset_BedMachine()
    bm.convert_and_reproject_netcdf_to_wgs84_egm2008_geotiffs(resolution_s=15)
    # dirname = "/home/mmacferrin/Research/DATA/DEMs/BedMachine/15s/converted"
    # bm.clear_empty_tiles([os.path.join(dirname, fn) for fn in os.listdir(dirname) if re.search(r"_egm2008\.tif\Z", fn) is not None])
