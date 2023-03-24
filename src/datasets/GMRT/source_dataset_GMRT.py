# -*- coding: utf-8 -*-

"""Source code for the GMRT ETOPO source dataset class."""

import os
import numpy
import subprocess
import re
from osgeo import gdal
import pandas
import shapely.geometry

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
import utils.traverse_directory

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
        for i, row in enumerate(etopo_gdf.iterrows()):
            _, row = row

            # Right now, just do these three tiles before meeting w/ Chris & Matt. Otherwise, skip.
            # TODO: Comment this out later.
            # tile = row.filename
            # if not ((tile.find("N15W090") > -1) or (tile.find("N30W090") > -1) or (tile.find("N45W180") > -1)):
            #     continue

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

            # THESE ARE TILES WHERE GMRT LEGIT HAS NO DATA. Don't bother with these.
            tile = os.path.basename(gmrt_tile_fname)
            if  (tile.find("N00E015") > -1) or \
                (tile.find("S15E015") > -1) or \
                (tile.find("N30E090") > -1) or \
                (tile.find("N45E105") > -1) or \
                (tile.find("N45W120") > -1) or \
                (tile.find("S90W135") > -1) or \
                (tile.find("S90W090") > -1) or \
                (tile.find("S90W015") > -1) or \
                (tile.find("S90E000") > -1) or \
                (tile.find("S90E015") > -1) or \
                (tile.find("S90E030") > -1) or \
                (tile.find("S90E045") > -1) or \
                (tile.find("S90E060") > -1) or \
                (tile.find("S90E075") > -1) or \
                (tile.find("S90E090") > -1) or \
                (tile.find("S90E105") > -1) or \
                (tile.find("S90E120") > -1) or \
                (tile.find("S90E135") > -1):
                if verbose:
                    print("{0}/{1} {2} contains no GMRT data and has been excluded.".format(i+1, len(etopo_gdf), os.path.basename(tile)))
                continue

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
                ybottom = -90 + (resolution_s / 3600)
            if ytop == 90:
                ytop = 90 - (resolution_s / 3600)

            stacks_command = ["waffles", "-M", "stacks",
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
                               "gmrt:layer=topo-mask:fmt=netcdf:res=max"]

            if not os.path.exists(gmrt_tile_fname):
                # p = subprocess.run(fetches_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

                print("\n" + " ".join(stacks_command))
                p = subprocess.run(stacks_command)

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
                tile_id = re.search(r"[NS](\d{2})[EW](\d{3})", os.path.split(row.filename)[1]).group()
                if tile_id in bad_tiles_id_list:
                    bad_tiles_df_subset = bad_tiles_df[bad_tiles_df.TileID == tile_id]

                    for (_,badvalues_row) in bad_tiles_df_subset.iterrows():
                        # If we've used the "badvalue" column, mask out anything of that value.
                        if not numpy.isnan(badvalues_row.badvalue):
                            badvalues_mask = badvalues_mask | (array == badvalues_row.badvalue)
                        # If we've instead used the "bad_row_min" and other similar columns to box-mask, mask out those.
                        else:
                            assert not (numpy.isnan(badvalues_row.bad_row_min) \
                                        or numpy.isnan(badvalues_row.bad_row_max) \
                                        or numpy.isnan(badvalues_row.bad_col_min) \
                                        or numpy.isnan(badvalues_row.bad_col_max))
                            badvalues_mask[int(badvalues_row.bad_row_min):int(badvalues_row.bad_row_max),
                                           int(badvalues_row.bad_col_min):int(badvalues_row.bad_col_max)] = True

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


    def translate_15s_bad_regions_to_1s(self, verbose=True):
        """We had come up with "GMRT_bad_tile_values_15s.csv" to eliminate regions in the GMRT tiles that had bad data.
        We new need to translate those same regions into the 1s tile coordinates.
        """
        gdf_15 = self.get_geodataframe(resolution_s=15, verbose=verbose)
        gdf_1 = self.get_geodataframe(resolution_s=1, verbose=verbose)
        # TODO: Finish here.

        bad_tiles_csvname_15s = self.config.gmrt_bad_values_csv.format(15)
        bad_tiles_csvname_1s = self.config.gmrt_bad_values_csv.format(1)

        df_bad_values_15s = pandas.read_csv(bad_tiles_csvname_15s)
        # Create a dtype-like dictionary object to define the columns of the new datatype.
        dt_dict = dict([(name, pandas.Series(dtype=df_bad_values_15s[name].dtype)) for name in df_bad_values_15s.columns])
        df_bad_values_1s = pandas.DataFrame(dt_dict)
        # list_of_new_records = []

        for i, row_15 in df_bad_values_15s.iterrows():
            tile_id = str(row_15.TileID)

            # Find the feature in the 15s gdf with this tile_ID
            tile_15s_row = gdf_15[gdf_15.filename.str.contains(tile_id)]
            tile_15s_geom = tile_15s_row.geometry.tolist()[0]

            if not numpy.isnan(row_15.badvalue) and numpy.isnan(row_15.bad_row_min):
                # In this case, we're giving it a bad value to mask out, not a bad bounding box.
                print(i, "Case 1", tile_id)

                # print(tilenames_1s)
                tilenames_1s = self.retrieve_list_of_datafiles_within_polygon(tile_15s_geom, gdf_15.crs, resolution_s=1,
                                                                              verbose=verbose)
                tile_ids_1s = [re.search(r"[NS]\d{2}[EW]\d{3}", os.path.basename(tn)).group() for tn in tilenames_1s]

                # Since there are currently no IDs overlapping the CRMs where we had used "badvalue" notation (the only
                # tile we used that one on was in Alaska, which does not intersect the CONUS-coastal CRMs), then this
                # produces an empty set there. I could flesh out this logic for completeness, but will skip it for now.
                # If we ever do produce 1s tiles over Alaska, and/or include other mask areas to omit certain values,
                # we will fix this then. But I'm skipping it for now.

            elif numpy.isnan(row_15.badvalue) and not numpy.isnan(row_15.bad_row_min):
                # In this case, we've specified a row/col bounding box to mask out.

                # None of these values should be blank here. We need them all.
                assert not (numpy.isnan(row_15.bad_row_min) or numpy.isnan(row_15.bad_row_max) or
                            numpy.isnan(row_15.bad_col_min) or numpy.isnan(row_15.bad_col_max))

                # Get the geotransform from the 15s raster file.
                gtf_15s = gdal.Open(tile_15s_row.filename.tolist()[0], gdal.GA_ReadOnly).GetGeoTransform()
                xmin, xstep, _, ymax, _, ystep = gtf_15s
                assert (xstep > 0) and (ystep < 0)

                latmax = ymax + (row_15.bad_row_min * ystep)
                latmin = ymax + (row_15.bad_row_max * ystep)
                lonmin = xmin + (row_15.bad_col_min * xstep)
                lonmax = xmin + (row_15.bad_col_max * xstep)
                assert (latmin < latmax) and (lonmin < lonmax)
                # Create a lat/lon geometry from this bounding 15s bounding box.
                bbox_15s = shapely.geometry.Polygon.from_bounds(lonmin, latmin, lonmax, latmax)

                # Find the subset of all the 1s tiles that insersect this 15s bounding-box.
                # Create a copy, since we'll be assigning to it in the next command.
                subset_1s = self.retrieve_list_of_datafiles_within_polygon(bbox_15s,
                                                                           gdf_15.crs,
                                                                           resolution_s=1,
                                                                           return_fnames_only=False,
                                                                           verbose=verbose).copy()

                # Get just the subset of each 1s polygon that insersects with the 15s polygon.
                subset_1s["intersect_1s"] = subset_1s.intersection(bbox_15s)

                print(i, "Case 2", tile_id, len(subset_1s), "1s records.")

                for j, row_1s in subset_1s.iterrows():
                    sub_minlon, sub_minlat, sub_maxlon, sub_maxlat = row_1s.intersect_1s.bounds
                    # Get the 1s geotransform from the file.
                    gtf_1s = gdal.Open(row_1s.filename, gdal.GA_ReadOnly).GetGeoTransform()
                    tile_minx, tile_xstep, _, tile_maxy, _, tile_ystep = gtf_1s
                    badvalue_data_1s = {'TileID': re.search(r"[NS]\d{2}[EW]\d{3}", os.path.basename(row_1s.filename)).group(),
                                        'badvalue': numpy.NaN,
                                        'bad_row_min': int((sub_maxlat - tile_maxy) / tile_ystep),
                                        'bad_row_max': int((sub_minlat - tile_maxy) / tile_ystep),
                                        'bad_col_min': int((sub_minlon - tile_minx) / tile_xstep),
                                        'bad_col_max': int((sub_maxlon - tile_minx) / tile_xstep),
                                        'comment': row_15.comment
                                        }

                    assert badvalue_data_1s['bad_row_min'] < badvalue_data_1s['bad_row_max']
                    assert badvalue_data_1s['bad_col_min'] < badvalue_data_1s['bad_col_max']
                    # print(badvalue_data_1s)

                    # Add this row to the dataframe_badvalues_1s
                    df_bad_values_1s.loc[len(df_bad_values_1s.index)] = badvalue_data_1s.values()
                    # list_of_new_records.append(pandas.Series(badvalue_data_1s))

            else:
                raise ValueError("GMRT Bad_value bad row instance. Must specify a 'badvalue' or a bounding-box, but not both:\n    " +
                                 str(row))

        # df_bad_values_1s = pandas.concat([df_bad_values_1s] + list_of_new_records)
        df_bad_values_1s.to_csv(bad_tiles_csvname_1s)
        if verbose:
            print(bad_tiles_csvname_1s, "written with", len(df_bad_values_1s), "records.")

def remove_xml_files(self, and_inf = True):
        """Remove all XML files from the data directory. If and_inf, remove the .inf files as well."""
        xml_files = utils.traverse_directory.list_files(self.config.source_datafiles_directory,
                                                        regex_match=r"\.((xml)|(inf))\Z" if and_inf else r"\.xml\Z",
                                                        include_base_directory = True)
        for xml in xml_files:
            os.remove(xml)
        print(len(xml_files), "files removed.")

if __name__ == "__main__":
    # Create the tiles.
    gmrt = source_dataset_GMRT()
    # gmrt.translate_15s_bad_regions_to_1s()
    gmrt.clean_bad_gmrt_values(resolution_s=1)
    # gmrt.create_tiles(resolution_s=1, overwrite=True)
    # Get rid of any empty tiles.
    # # gmrt.delete_empty_tiles(resolution_s=15)
    # gmrt.delete_empty_tiles(resolution_s=1)

    # for res in (15,1):
    # for res in (15,):
        # gmrt.create_tiles(resolution_s=res, overwrite=False)
        # gmrt.clean_bad_gmrt_values(resolution_s = res, verbose = True)
        # gmrt.delete_empty_tiles(resolution_s=res)
    # gmrt.delete_empty_tiles(resolution_s=res)

    # os.remove(gmrt.get_geopkg_object().get_gdf_filename(resolution_s=15))
    # print(os.path.basename(gmrt.get_geopkg_object().get_gdf_filename(resolution_s=15)), "removed.")
    # gdf15 = gmrt.get_geodataframe(resolution_s = 15)

    # gmrt.remove_xml_files(and_inf=True)
    # gmrt1 = source_dataset_GMRT()
    # gdf1 = gmrt1.get_geodataframe(resolution_s = 1)
    #
    # print (gdf15)
    # print(gdf1)