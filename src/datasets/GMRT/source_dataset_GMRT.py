# -*- coding: utf-8 -*-

"""Source code for the GMRT ETOPO source dataset class."""

import os
import numpy
import subprocess
import re
from osgeo import gdal
import pandas

#####################
# Don't actually need the CUDEM package here, but it's an indicator whether CUDEM is installed on this machine.
# If not, toss an error (we need it).
try:
    import cudem
except ImportError as e:
    import sys
    print('CUDEM must be installed on this machine.', file=sys.stderr)
    raise e
#####################

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset
import datasets.dataset_geopackage as dataset_geopackage

class source_dataset_GMRT(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "GMRT_config.ini" )):
        """Initialize the GMRT source dataset object."""

        super(source_dataset_GMRT, self).__init__("GMRT", configfile)

    def create_tiles(self, resolution_s=15, crm_only_if_1s=True, also_compute_stats=True, overwrite=False, verbose=True):
        """Using the CUDEM 'fetches' module, create all the tiles needed for this dataset."""
        etopo_gpkg = dataset_geopackage.ETOPO_Geopackage(resolution = resolution_s)
        etopo_gdf = etopo_gpkg.get_gdf(crm_only_if_1s=crm_only_if_1s,
                                       resolution_s=resolution_s,
                                       verbose=verbose).copy()

        # Loop through all the ETOPO files and create an identical tile in this dataset.
        for i, row in etopo_gdf.iterrows():
            xleft = row.xleft
            xright = numpy.round(xleft + (row.xsize*row.xres))
            ytop = row.ytop
            ybottom = numpy.round(ytop + (row.ysize*row.yres))

            gmrt_tile_fname = os.path.join(self.config._abspath(self.config.source_datafiles_directory),
                                           "{0}s".format(resolution_s),
                                           self.config.datafiles_name_template.format(resolution_s,
                                                                                      "N" if ybottom >= 0 else "S",
                                                                                      abs(int(numpy.round(ybottom))),
                                                                                      "E" if xleft >= 0 else "W",
                                                                                      abs(int(numpy.round(xleft)))))

            # TEMP: for debugging only.
            # TODO: Remove later
            # if not ((xleft == 0 and ytop == 90) or (xleft == 0 and ybottom == -90)):
            #     continue

            tile_already_exists = False
            if os.path.exists(gmrt_tile_fname):
                if overwrite:
                    os.remove(gmrt_tile_fname)
                else:
                    tile_already_exists = True
                # else:
                    # print("{0}/{1} {2} skipped (already written).".format(i+1, len(etopo_gdf), os.path.split(gmrt_tile_fname)[1]))
                    # continue

            if ybottom == -90:
                ybottom = -89.999
            if ytop == 90:
                ytop = 89.999

            fetches_command = ["waffles", "-M", "stacks",
                               "-w",
                               # Note: we include a /-/0 elevation cutoff (only include values below 0) to exclude land from this dataset. We're only interested in ocean here.
                               "-R", "{0}/{1}/{2}/{3}/-/0".format(xleft,xright,ybottom,ytop),
                               "-E", "{0}s".format(resolution_s),
                               "--t_srs", "EPSG:4326",
                               "-D", etopo_gpkg.config.etopo_cudem_cache_directory,
                               "-k",
                               "-O", os.path.splitext(gmrt_tile_fname)[0],
                               "-f",
                               "-F", "GTiff",
                               "gmrt:layer=topo-mask:fmt=netcdf"]

            if not os.path.exists(gmrt_tile_fname):
                # p = subprocess.run(fetches_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

                print(" ".join(fetches_command))
                p = subprocess.run(fetches_command)

            xml_file = gmrt_tile_fname + ".aux.xml"
            xml_already_exists = False
            if os.path.exists(xml_file):
                if overwrite:
                    os.remove(xml_file)
                else:
                    xml_already_exists = True

            if also_compute_stats and os.path.exists(gmrt_tile_fname):
                gdal_cmd = ["gdalinfo", "-stats", gmrt_tile_fname]
                subprocess.run(gdal_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            if (not (os.path.exists(gmrt_tile_fname) or (p.returncode == 0))) and verbose:
                print("ERROR: '{0}' sent return code {1}:".format(" ".join(fetches_command), p.returncode))
                print(p.stdout)

            if verbose:
                print("{0}/{1} {2} {3}{4}".format(i+1, len(etopo_gdf),
                                                   os.path.split(gmrt_tile_fname)[1],
                                                   "" if os.path.exists(gmrt_tile_fname) else "NOT ",
                                                   "already exists" if tile_already_exists else "written"), end="")
                if also_compute_stats:
                    print(", {0} {1}{2}.".format(os.path.split(xml_file)[1],
                                                  "" if os.path.exists(xml_file) else "NOT ",
                                                  "already exists" if xml_already_exists else "written"))
                else:
                    print(".")
        return


    def delete_empty_tiles(self, resolution_s=15, verbose=True):
        """The above generation made a tile for every point on Earth. Many contain no data. Eliminate the ones with no data."""
        gmrt_dir = os.path.join(self.config._abspath(self.config.source_datafiles_directory),
                                "{0}s".format(resolution_s))
        tile_regex = self.config.datafiles_regex

        tilenames = sorted([os.path.join(gmrt_dir, fn) for fn in os.listdir(gmrt_dir) if re.search(tile_regex, fn) != None])
        # print(str(resolution_s) + "s", len(tilenames), "tiles.")

        num_removed = 0
        num_kept = 0
        for i,tile in enumerate(tilenames):
            print("{0}/{1} {2}".format(i+1, len(tilenames), os.path.split(tile)[1]), end="")
            ds = gdal.Open(tile, gdal.GA_ReadOnly)
            band = ds.GetRasterBand(1)
            array = band.ReadAsArray()
            ndv = band.GetNoDataValue()
            if numpy.all(array == ndv):
                band = None
                ds = None
                os.remove(tile)
                num_removed += 1
                if verbose:
                    print(" removed.")
            else:
                num_kept += 1
                band = None
                ds = None
                if verbose:
                    print(" kept w {0:0.2f}% good data.".format(100.0 * numpy.count_nonzero(array != ndv) / array.size))

        if verbose:
            print("{0} tiles kept, {1} tiles removed.".format(num_kept, num_removed))
        return

    def clean_bad_gmrt_values(self,
                              resolution_s = 15,
                              remove_gte0 = True,
                              delete_empty_tiles = True,
                              update_gpkg = True,
                              verbose=True):
        """Some GMRT areas have bad-data values, especially in the global 15s data. Remove them.

        Also, some areas, despite using the topo-layer mask, they include land values. If remove_gte0,
        remove all elevations values >= 0.

        If delete_empty_tiles, after we've cleaned the data, delete any tiles that are now empty (all no-data).
        """
        gdf = self.get_geodataframe(resolution_s = resolution_s, verbose=verbose)

        bad_tiles_csvname = self.config.gmrt_bad_values_csv.format(resolution_s)

        bad_tiles_df = None
        if os.path.exists(bad_tiles_csvname):
            bad_tiles_df = pandas.read_csv(bad_tiles_csvname, index_col=False)
            bad_tiles_id_list = bad_tiles_df.TileID.tolist()
            if verbose:
                print(os.path.split(bad_tiles_csvname)[1], "read with", len(bad_tiles_df), "entries.")
        elif verbose:
            print(os.path.split(bad_tiles_csvname)[1], "does not exist.")

        any_files_deleted = False
        for i,row in enumerate(gdf.iterrows()):
            _, row = row
            if not os.path.exists(row.filename):
                if verbose:
                    print("{0}/{1} {2} does not exist.".format(i+1, len(gdf), os.path.split(row.filename)[1]))
                continue
            ds = gdal.Open(row.filename, gdal.GA_Update)
            band = ds.GetRasterBand(1)
            array = band.ReadAsArray()
            ndv = band.GetNoDataValue()
            badvalues_mask = numpy.zeros(array.shape, dtype=bool)
            if bad_tiles_df is not None:
                tile_id = re.search("[NS](\d{2})[EW](\d{3})", os.path.split(row.filename)[1]).group()
                if tile_id in bad_tiles_id_list:
                    for badvalue in bad_tiles_df[bad_tiles_df.TileID == tile_id].badvalue:
                        badvalues_mask = badvalues_mask | (array == badvalue)

            if remove_gte0:
                badvalues_mask = badvalues_mask | (array >= 0.0)

            # If we've flagged any bad values, put NDV in and save it back to the grid.
            if numpy.any(badvalues_mask):
                array[badvalues_mask] = ndv

                # Now, check to see if the array is all NDVs or not.
                if delete_empty_tiles and numpy.all(array == ndv):
                    band = None
                    ds = None
                    os.remove(row.filename)
                    any_files_deleted = True
                    if verbose:
                        print("{0}/{1} {2} deleted with no more valid data.".format(i+1,
                                                                                    len(gdf),
                                                                                    os.path.split(row.filename)[1]))
                    continue

                band.WriteArray(array)
                band.GetStatistics(0,1)
                band = None
                ds = None
                if verbose:
                    print("{0}/{1} {2} re-written with {3:0.1f}% bad data filtered out.".format(
                           i+1,
                           len(gdf),
                           os.path.split(row.filename)[1],
                           100*numpy.count_nonzero(badvalues_mask)/array.size))

            else:
                if verbose:
                    print("{0}/{1} {2} unchanged.".format(i+1, len(gdf), os.path.split(row.filename)[1]))

        if any_files_deleted and update_gpkg:
            gdf = None
            gpkg_object = self.get_geopkg_object(verbose = verbose)
            gpkg_fname = gpkg_object.get_gdf_filename(resolution_s = resolution_s)
            os.remove(gpkg_fname)
            gpkg_object.gdf = None
            # The dataset_geopackage object will rebuild the gdf using the new files.
            gdf = self.get_geodataframe(resolution_s = resolution_s, verbose=verbose)


if __name__ == "__main__":
    # Create the tiles.
    gmrt = source_dataset_GMRT()
    # gmrt.create_tiles(resolution_s=1)
    # Get rid of any empty tiles.
    # gmrt.delete_empty_tiles(resolution_s=15)
    # gmrt.delete_empty_tiles(resolution_s=1)

    for res in (15,):
        gmrt.create_tiles(resolution_s=res, overwrite=False)
        gmrt.clean_bad_gmrt_values(resolution_s = res, verbose = True)
    gmrt.delete_empty_tiles(resolution_s=res)


    # gdf15 = gmrt.get_geodataframe(resolution_s = 15)
    # gmrt1 = source_dataset_GMRT()
    # gdf1 = gmrt1.get_geodataframe(resolution_s = 1)
    #
    # print (gdf15)
    # print(gdf1)