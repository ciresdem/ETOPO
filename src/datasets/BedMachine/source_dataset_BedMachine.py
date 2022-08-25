# -*- coding: utf-8 -*-

"""Source code for the BedMachine ETOPO source dataset class."""

import os
import subprocess
from osgeo import gdal
import numpy
import pyproj
import shapely
import matplotlib.pyplot as plt

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
                               capture_output=not verbose)

                assert os.path.exists(gtif_name)

            # Second, filter out non-Greenland elevations (if in Greenland). -- OR JUST DATA FROM NON-Greenland land.
            # Also, filter out ocean elevations in Antarctica.
            mask_name = os.path.splitext(tname)[0] + ".mask.tif"
            mask_ds = gdal.Open(mask_name, gdal.GA_ReadOnly)
            mask_array = mask_ds.GetRasterBand(1).ReadAsArray()
            mask_ds = None  # Close the file & deallocate resources.
            # The mask values are: 0: ocean
            #                      1: ice-free land
            #                      2: grounded ice
            #                      3: floating ice
            #                      4: non-Greenland land (in Greenland only)

            # Mask out non-Greenland parts of the Greenland bedmachine.
            for var in ['bed', 'surface']:
                gtif_name = os.path.splitext(tname)[0] + ".{0}.tif".format(var)
                elev_ds = gdal.Open(gtif_name, gdal.GA_Update)
                assert elev_ds is not None
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
                elev_array = elev_array + geoid_array

                if tname.find("BedMachineGreenland") >= 0:
                    # Get only data that isn't nodata, and is part of Greenland (omit Ellesmere Island)
                    non_greenland_vals = (mask_array == 4)
                    if numpy.count_nonzero(non_greenland_vals) == 0:
                        continue
                    mask_invalid = non_greenland_vals  # | (elev_array == elev_ndv)
                if tname.find("Antarctica") >= 0:
                    # Filter out ocean basin in Antarctica. GEBCO is better there.
                    ocean_cells = (mask_array == 0)
                    if numpy.count_nonzero(ocean_cells):
                        continue
                    mask_invalid = ocean_cells  # | (elev_array == elev_ndv)

                elev_array[mask_invalid] = elev_ndv
                elev_band.WriteArray(elev_array)
                # Write to disk.
                elev_band = None
                elev_ds = None

                if verbose:
                    print(os.path.split(gtif_name)[1], "written with {0:4.1f}% valid data.".format(
                        numpy.count_nonzero(~mask_invalid) / mask_invalid.size * 100))

            # Find the ETOPO tiles that overlap this polygon.
            # Let's convert this polygon to WGS84 to make this not break at all (and make fer' damned sure Antarctica is wrapped correctly).
            for var in ['bed', 'surface']:
                gtif_name = os.path.splitext(tname)[0] + ".{0}.tif".format(var)

                polygon, proj, xleft, ytop, xres, yres, xsize, ysize = \
                    etopo_gpkg.create_outline_geometry_of_geotiff(gtif_name)
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

                # print(etopo_gdf_subset.columns)
                for i, row in etopo_gdf_subset.iterrows():
                    # print(row.xleft, row.xres, row.ytop, row.yres)
                    # print(str(proj.to_epsg()))
                    ybottom = round(row.ytop + (row.yres * row.ysize))
                    xright = round(row.xleft + (row.xres * row.xsize))
                    gtif_fname_only = os.path.split(gtif_name)[1]
                    outfile_basename = "BedMachine" + \
                                       ("Greenland" if tname.find("Greenland") >= 0 else "Antarctica") + \
                                       "_{0}s_".format(resolution_s) + \
                                       ("S" if (ybottom < 0) else "N") + \
                                       "{0:02d}".format(abs(ybottom)) + \
                                       ("W" if (row.xleft < 0) else "E") + \
                                       "{0:03d}".format(abs(int(row.xleft))) + \
                                       ".{0}.tif".format(var)

                    if verbose:
                        print(outfile_basename)
                    # continue

                    projected_name = os.path.join(os.path.dirname(gtif_name), "projected", outfile_basename)
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
                                       capture_output=not verbose)
                        print()
                    assert os.path.exists(projected_name)

                    base, ext = os.path.splitext(outfile_basename)
                    converted_fname = os.path.join(os.path.dirname(gtif_name), "converted", base + "_egm2008" + ext)

                    if os.path.exists(converted_fname) and overwrite:
                        os.remove(converted_fname)

                    if not os.path.exists(converted_fname):
                        # Then, we'll convert it to EGM2008 elevations.
                        etopo.convert_vdatum.convert_vdatum(projected_name,
                                                            converted_fname,
                                                            input_vertical_datum="wgs84",
                                                            output_vertical_datum="egm2008",
                                                            cwd=os.path.dirname(converted_fname),
                                                            verbose=verbose)

                    if not os.path.exists(converted_fname):
                        print("ERROR: Could not create", os.path.split(converted_fname)[1])
                    converted_files_list.append(converted_fname)

        # Go through the list of converted files and get rid of ones that have no good data. This happens from the
        # Masking above.
        for fname in converted_files_list:
            ds = gdal.Open(fname, gdal.GA_ReadOnly)
            band = ds.GetRasterBand(1)
            array = band.ReadAsArray()
            ndv = band.GetNoDataValue()
            # If everything is nodata, just get rid of the file.
            if numpy.all(array == ndv):
                if verbose:
                    print("Removing", os.path.split(fname)[1], "w/ no valid data.")
                os.remove(fname)
                continue

            del band
            del ds

        return


if __name__ == "__main__":
    bm = source_dataset_BedMachine()
    bm.convert_and_reproject_netcdf_to_wgs84_egm2008_geotiffs()
