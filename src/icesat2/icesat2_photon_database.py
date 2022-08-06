# -*- coding: utf-8 -*-

"""Code to manage spatial databases of classified land/veg icesat-2 photons
for memory-efficient handling of ICESat-2 data."""

import os
import pyproj
import geopandas
import pandas
import re
import numpy
import shapely.geometry
import shutil
import tables
import time

#####################################
# Suppress the annoying pandas.FutureWarning warnings caused by library version conflicts.
# It doesn't affect my code and will be resolved in future releases of pandas.
# For now, just suppress the warning.
import warnings
warnings.filterwarnings("ignore", message=".*pandas.Int64Index is deprecated*")
#####################################

####################################3
# Include the base /src/ directory of thie project, to add all the other modules.
import import_parent_dir; import_parent_dir.import_src_dir_via_pythonpath()
####################################3
import icesat2.classify_icesat2_photons                    as classify_icesat2_photons
import icesat2.nsidc_download                              as nsidc_download
import datasets.CopernicusDEM.source_dataset_CopernicusDEM as Copernicus
# import datasets.dataset_geopackage                         as dataset_geopackage
import utils.configfile
import etopo.generate_empty_grids
import utils.progress_bar

class ICESat2_Database:
    """A database to manage ICESat-2 photon clouds for land-surface validations.
    The database is a set of a tiles, managed spatially by a GeoPackage object,
    with each tile being an HDF5 database of all land & canopy photons classified
    from ICESat-2 granules that overlap each tile."""

    def __init__(self, tile_resolution_deg = 0.25):
        """tile_resolutin_deg should be some even fraction of 1. I.e. 1, or 0.5, or 0.25, or 0.1, etc."""
        self.etopo_config = utils.configfile.config()
        self.gpkg_fname = self.etopo_config.icesat2_photon_geopackage
        self.tiles_directory = self.etopo_config.icesat2_photon_tiles_directory

        self.gdf = None # The actual geodataframe object.
        self.tile_resolution_deg = tile_resolution_deg
        self.crs = pyproj.CRS.from_epsg(4326)

    def get_gdf(self, verbose=True):
        """Return self.gpkg if exists, otherwise read self.gpkg_name, save it to self.gdf and return."""
        if self.gdf is None:
            if os.path.exists(self.gpkg_fname):
                print("Reading", os.path.split(self.gpkg_fname)[1])
                self.gdf = geopandas.read_file(self.gpkg_fname, mode='r')
            else:
                self.gdf = self.create_new_geopackage(verbose=verbose)

        return self.gdf

    def numrecords(self):
        return len( self.get_gdf(verbose=False) )

    def create_new_geopackage(self, populate_with_existing_tiles = True, verbose=True):
        """Create the geopackage that handles the photon database files of ICESat-2 data."""
        # Columns to have : filename, xmin, xmax, ymin, ymax, numphotons, numphotons_canopy, numphotons_ground, geometry
        # "numphotons", "numphotons_canopy", "numphotons_ground" are all set to zero at first. Are populated later as files are written.
        # "geometry" will be a square polygon bounded by xmin, xmax, ymin, ymax

        # Since we're interested in land-only, we will use the CopernicusDEM dataset
        # to determine where land tiles might be.
        copernicus_gdf = Copernicus.source_dataset_CopernicusDEM().get_geodataframe(verbose=verbose)
        copernicus_fnames = [os.path.split(fn)[1] for fn in copernicus_gdf["filename"].tolist()]
        copernicus_bboxes = [self.get_bbox_from_copernicus_filename(fn) for fn in copernicus_fnames]
        copernicus_bboxes.extend(etopo.generate_empty_grids.get_azerbaijan_1deg_bboxes())
        # Skip all bounding boxes with a y-min of -90, since ICESat-2 only goes down to -89.
        # Don't need to worry about north pole, since the northernmost land bbox tops out at 84*N
        copernicus_bboxes = [bbox for bbox in copernicus_bboxes if bbox[1] > -90]

        copernicus_bboxes = sorted(copernicus_bboxes)

        # Subtract the epsilon just to make sure we don't accidentally add an extra box due to a rounding error
        tiles_to_degree_ratio = int(1/(self.tile_resolution_deg - 0.0000000001))
        N = len(copernicus_bboxes) * (tiles_to_degree_ratio)**2

        tile_filenames = [None] * N
        tile_xmins = numpy.empty((N,), dtype=numpy.float32)
        tile_xmaxs = numpy.empty((N,), dtype=numpy.float32)
        tile_ymins = numpy.empty((N,), dtype=numpy.float32)
        tile_ymaxs = numpy.empty((N,), dtype=numpy.float32)
        tile_geometries = [None] * N
        # These fields are all initialized to zero. Will be filled in as files are created.
        tile_numphotons = numpy.zeros((N,), dtype=numpy.uint32)
        tile_numphotons_canopy = numpy.zeros((N,), dtype=numpy.uint32)
        tile_numphotons_ground = numpy.zeros((N,), dtype=numpy.uint32)
        tile_is_populated = numpy.zeros((N,), dtype=bool)
        # Loop through each copernicus bbox, get tile bboxes
        i_count = 0

        if verbose:
            print("Creating", self.gpkg_fname, "...")

        for cop_bbox in copernicus_bboxes:

            bbox_xrange = numpy.arange(cop_bbox[0], cop_bbox[2]-0.0000000001, self.tile_resolution_deg)
            bbox_yrange = numpy.arange(cop_bbox[1], cop_bbox[3]-0.0000000001, self.tile_resolution_deg)
            assert len(bbox_xrange) == tiles_to_degree_ratio
            assert len(bbox_yrange) == tiles_to_degree_ratio
            for tile_xmin in bbox_xrange:
                tile_xmax = tile_xmin + self.tile_resolution_deg
                for tile_ymin in bbox_yrange:
                    tile_ymax = tile_ymin + self.tile_resolution_deg
                    tile_fname = os.path.join(self.tiles_directory, "photon_tile_{0:s}{1:05.2f}_{2:s}{3:06.2f}_{4:s}{5:05.2f}_{6:s}{7:06.2f}.h5".format(\
                                                            "S" if (tile_ymin < 0) else "N",
                                                            abs(tile_ymin),
                                                            "W" if (tile_xmin < 0) else "E",
                                                            abs(tile_xmin),
                                                            "S" if (tile_ymax < 0) else "N",
                                                            abs(tile_ymax),
                                                            "W" if (tile_xmax < 0) else "E",
                                                            abs(tile_xmax)))

                    tile_polygon = shapely.geometry.Polygon([(tile_xmin,tile_ymin),
                                                             (tile_xmin,tile_ymax),
                                                             (tile_xmax,tile_ymax),
                                                             (tile_xmax,tile_ymin),
                                                             (tile_xmin,tile_ymin)])

                    tile_filenames[i_count] = tile_fname
                    tile_xmins[i_count] = tile_xmin
                    tile_xmaxs[i_count] = tile_xmax
                    tile_ymins[i_count] = tile_ymin
                    tile_ymaxs[i_count] = tile_ymax
                    tile_geometries[i_count] = tile_polygon

                    i_count += 1

        data_dict = {'filename': tile_filenames,
                     'xmin'    : tile_xmins,
                     'xmax'    : tile_xmaxs,
                     'ymin'    : tile_ymins,
                     'ymax'    : tile_ymaxs,
                     'numphotons'       : tile_numphotons,
                     'numphotons_canopy': tile_numphotons_canopy,
                     'numphotons_ground': tile_numphotons_ground,
                     'is_populated'     : tile_is_populated,
                     'geometry': tile_geometries
                     }

        # Create the geodataframe
        self.gdf = geopandas.GeoDataFrame(data_dict, geometry='geometry', crs=self.crs)
        # Compute the spatial index, just to see if it saves it (don't think so but ???)
        # sindex = self.gdf.sindex

        if verbose:
            print("gdf has", len(self.gdf), "tile bounding boxes.")

        if populate_with_existing_tiles:
            # Read all the existing tiles, get the stats, put them in there.
            existing_mask = self.gdf['filename'].apply(os.path.exists, convert_dtype=False)
            # Get the subset of tiles where the file currently exists on disk.
            gdf_existing = self.gdf[existing_mask]

            if verbose:
                print("Reading", len(gdf_existing), "existing tiles to populate database.")

            for i,row in enumerate(gdf_existing.itertuples()):
                tile_df = pandas.read_hdf(row.filename, mode='r')

                self.gdf.loc[row.Index, "numphotons"] = len(tile_df)
                self.gdf.loc[row.Index, "numphotons_canopy"] = numpy.count_nonzero(tile_df['class_code'].between(2,3,inclusive='both'))
                self.gdf.loc[row.Index, "numphotons_ground"] = numpy.count_nonzero(tile_df['class_code']==1)
                self.gdf.loc[row.Index, 'is_populated'] = True

                if verbose:
                    utils.progress_bar.ProgressBar(i+1, len(gdf_existing), suffix="{0}/{1}".format(i+1, len(gdf_existing)))

        # Save it out to an HDF file.
        self.save_geopackage(verbose=verbose)

        return self.gdf

    def get_bbox_from_copernicus_filename(self, filename):
        """From a CopernicusDEM filename, get the bbox [xmin,ymin,xmax,ymax].

        We use this to generate where our icesat-2 photon tiles need to be.

        This is specifically the Coperinicus 30m datasets (1 arc-second),
        which are 1-degree tiles."""
        # Copernicus tiles are all in Coperinicus_DSM_COG_10_NYY_00_EXXX_00_DEM.tif formats. Get the bboxes from there.
        lat_letter_regex_filter = r"(?<=Copernicus_DSM_COG_10_)[NS](?=\d{2}_00_)"
        lat_regex_filter = r"(?<=Copernicus_DSM_COG_10_[NS])\d{2}(?=_00_)"
        lon_letter_regex_filter = r"(?<=Copernicus_DSM_COG_10_[NS]\d{2}_00_)[EW](?=\d{3}_00_DEM\.tif)"
        lon_regex_filter = r"(?<=Copernicus_DSM_COG_10_[NS]\d{2}_00_[EW])\d{3}(?=_00_DEM\.tif)"

        min_y = (1 if (re.search(lat_letter_regex_filter, filename).group() == "N") else -1) * \
                int(re.search(lat_regex_filter, filename).group())
        max_y = min_y + 1

        min_x = (1 if (re.search(lon_letter_regex_filter, filename).group() == "E") else -1) * \
                int(re.search(lon_regex_filter, filename).group())
        max_x = min_x + 1

        bbox = (min_x, min_y, max_x, max_y)

        # print(filename, bbox)
        return bbox

    def fill_in_missing_tile_entries(self, delete_csvs = True, save_to_disk = True, verbose = True):
        """Sometimes a photon_tile gets created and the _summary.csv file got deleted,
        but the database update wasn't saved. Loop through the existing photon tiles, fill
        in any missing entries in the database, and save it back out."""
        # First, let's ingest the CSV summary files in the directory.
        gdf = self.get_gdf(verbose=verbose)
        self.update_gpkg_with_csvfiles(gdf = gdf,
                                       use_tempfile = True,
                                       delete_when_finished = delete_csvs,
                                       save_to_disk = save_to_disk,
                                       verbose=verbose)

        existing_tiles = [os.path.join(self.tiles_directory, fn) for fn in os.listdir(self.tiles_directory) if (re.search("\Aphoton_tile_[\w\.]+\.h5\Z", fn) != None)]
        num_filled_in = 0
        for tilename in existing_tiles:
            tile_record = gdf.loc[gdf.filename == tilename]
            # If the tile exists and it says it's populated, move along.
            if tile_record["is_populated"].tolist()[0] == True:
                continue

            # Otherwise, let's get the data from the tile ane enter it.
            idx = tile_record.index.tolist()[0]
            try:
                tile_df = pandas.read_hdf(tilename, mode='r')
            except KeyboardInterrupt as e:
                raise e
            except Exception:
                # The tile might have an error if it was incompletely written before. If so, remove it.
                os.remove(tilename)
                summary_csv_name = os.path.splitext(tilename)[0] + "_summary.csv"
                if os.path.exists(summary_csv_name):
                    os.remove(summary_csv_name)

                continue

            gdf.loc[idx, 'numphotons']        = len(tile_df)
            gdf.loc[idx, 'numphotons_canopy'] = numpy.count_nonzero(tile_df["class_code"].between(2,3,inclusive="both"))
            gdf.loc[idx, 'numphotons_ground'] = numpy.count_nonzero(tile_df["class_code"] == 1)
            gdf.loc[idx, 'is_populated']      = True
            num_filled_in += 1

        if num_filled_in > 0:
            if verbose:
                print(num_filled_in, "missing tiles entered into the database.")
            # Only re-save this to disk if we've actually updated anything. Otherwise,
            # it would be finished after the previous update_gpkg_with_csvfiles() call.
            if save_to_disk:
                self.save_geopackage(gdf=gdf, use_tempfile=True, also_delete_redundant_csvs=False, verbose=verbose)

        if not gdf is self.gdf:
            self.gdf = gdf

        return gdf

    def save_geopackage(self, gdf=None,
                              use_tempfile = False,
                              also_delete_redundant_csvs=False,
                              verbose=True):
        """After writing or altering data in the geo-dataframe, save it back out to disk.

        If gdf is None, use whatever object is in self.gdf.
        """
        if gdf is None:
            if self.gdf is None:
                raise ValueError("No geodataframe object available to ICESat2_Database.save_gpkg().")
            gdf = self.gdf

        if not gdf is self.gdf:
            self.gdf = gdf

        if use_tempfile:
            base, ext = os.path.splitext(self.gpkg_fname)
            tempfile_name = base + "_TEMP" + ext
            if os.path.exists(tempfile_name):
                if verbose:
                    print(tempfile_name, "already exists.",
                          "\n\tExiting ICESat2_Database.save_geopackage(use_tempfile=True). Other processes may be writing to it.",
                          "\n\tIf this is in error, delete that file before running again.")
                return
            success = False
            while not success:
                try:
                    gdf.to_file(tempfile_name, layer="icesat2", driver="GPKG")
                    os.remove(self.gpkg_fname)
                    shutil.move(tempfile_name, self.gpkg_fname)
                    success = True
                except:
                    # Delete the tempfile, then re-raise the exception.
                    if os.path.exists(tempfile_name):
                        os.remove(tempfile_name)
                    if verbose:
                        print("Error occurred while writing", os.path.split(self.gpkg_fname) + ". Waiting 30 seconds to retry...")
                    time.sleep(30)
        else:
            # Write the file.
            gdf.to_file(self.gpkg_fname, layer="icesat2", driver='GPKG')

        if verbose:
            print(os.path.split(self.gpkg_fname)[1], "written with", len(gdf), "entries.")
        return

        if also_delete_redundant_csvs:
            self.delete_csvs_already_in_database(gdf = gdf,
                                                 force_read_from_disk = False,
                                                 verbose = verbose)

    def query_geopackage(self, polygon_or_bbox, use_sindex=True, return_whole_records=True):
        """Return the photon database tile filenames that intersect the polygon in question.

        Called by get_photon_database().

        'use_sindex' determines whether or not we use a spatial index (geopackage.sindex) to
        spatially query the database or not. If I'm querying it repeatedly it makes sense.
        If I'm only opening the file once and querying once, it doesn't make sense since it
        has to re-generate the spatial index upon the first use of it.

        If return_whole_records is True, then return the entire dataset subset, including all the fields.
            If False, then just return the list of filenames.
        """
        # If we have a bounding box, query it using the bounds
        if (type(polygon_or_bbox) in (list, tuple)) and (len(polygon_or_bbox) == 4):
            polygon = shapely.geometry.box(*polygon_or_bbox, ccw=False)
        else:
            assert type(polygon_or_bbox) == shapely.geometry.Polygon
            polygon = polygon_or_bbox

        # Use the polygon intersection tool to find the intersection.
        gdf = self.get_gdf()

        # Subset the records that overlap but don't just "touch" (on an edge or corner).
        gdf_subset = gdf[ gdf.intersects(polygon) & ~gdf.touches(polygon)]

        if return_whole_records:
            return gdf_subset
        else:
            return gdf_subset["filename"].tolist()

    def get_photon_database(self, polygon_or_bbox,
                                  build_tiles_if_nonexistent = True,
                                  verbose=True):
        """Given a polygon or bounding box, return the combined database of all
        the photons within the polygon or bounding box.

        Polygon is a shapely.geometry.Polygon object, or bounding-box is a 4-value list/tuple
        of (xmin,ymin,xmax,ymax). Coordinates should be in WGS84 (EPSG:4326) lat/lon. Any other
        coordinate system transformations should be done before being sent to this method.

        Return value:
            A pandas.DataFrame object containing the photon data of all tiles that overlap the
            given bounding-box.
        """
        df_tiles_subset = self.query_geopackage(polygon_or_bbox, return_whole_records=True)
        if verbose:
            print(len(df_tiles_subset.index), "ICESat-2 photon tiles overlap this polygon. Retrieving them.")

        dataframes_list = [None] * len(df_tiles_subset.index)

        for i,(idx, df_row) in enumerate(df_tiles_subset.iterrows()):
            fname = df_row['filename']
            # If the file already exists, read it and get the data.
            if os.path.exists(fname):
                if verbose:
                    print("\t{0}/{1} Reading".format(i+1, len(df_tiles_subset)), os.path.split(fname)[1], "...", end="")
                dataframes_list[i] = self.read_photon_tile(fname)
                if verbose:
                    print("Done.")
            # If the file doesn't exist, create it and get the data.
            elif build_tiles_if_nonexistent:
                if verbose:
                    print("\t{0}/{1} Creating".format(i+1, len(df_tiles_subset)), os.path.split(fname)[1], "...")
                dataframes_list[i] = self.create_photon_tile(df_row['geometry'],
                                                             fname,
                                                             overwrite=False,
                                                             write_stats = True,
                                                             verbose=verbose)

        # Get rid of any dataframes where data wasn't read.
        dataframes_list = [df for df in dataframes_list if (df is not None)]

        # Concatenate the dataframes together.
        if len(dataframes_list) > 0:
            combined_df = pandas.concat(dataframes_list, ignore_index=True)
            return combined_df
        else:
            return None

    def read_empty_tile(self, verbose=True):
        """Sometimes the "create_photon_tile() function gets literally zero photon dataframes back
        from its quiery. In that case, just return a copy of an empty dataframe we've
        made and saved with all the correct fields but no data records.

        If the empty dataframe doesn't exist, just read a random dataframe (hoping
        one of those exists), empty it, and save it out to the
        etopo_config.icesat2_photon_empty_tile file."""
        if os.path.exists(self.etopo_config.icesat2_photon_empty_tile):
            empty_df = pandas.read_hdf(self.etopo_config.icesat2_photon_empty_tile, mode="r")
        else:
            # If we can't find the empty tile, create it by gleaming off one of the other databases.
            # NOTE: This assumes at least one photon tile or one photon granul
            # database already exists in their respective folders. It might be a good idea down
            # the line to not rely upon this assumption. Maybe include the empty tile in the
            # git repository so we ensure it's there.
            list_of_files = [fn for fn in os.listdir(self.tiles_directory) if (re.search("\Aphoton_tile_[\w\.]+\.((h5)|(feather))\Z", fn) != None)]
            if len(list_of_files) > 0:
                example_file = os.path.join(self.tiles_directory, list_of_files[0])
            else:
                list_of_files = [fn for fn in os.listdir(self.etopo_config.icesat2_granules_directory) if re.search("\AATL03_(\w)+_photons\.((h5)|(feather))\Z", fn) != None]
                if len(list_of_files) == 0:
                    raise FileNotFoundError("Could not find an existing photon tile or granule to use to create the file", self.etopo_config.icesat2_photon_empty_tile)
                example_file = os.path.join(self.etopo_config.icesat2_granules_directory, list_of_files[0])

            df = pandas.read_hdf(example_file, mode="r")
            # Empty out all the records and return the empty dataframe.
            empty_df = df[[False] * len(df)]
            empty_df.to_hdf(self.etopo_config.icesat2_photon_empty_tile, key="icesat2")
            if verbose:
                print(self.etopo_config.icesat2_photon_empty_tile, "written.")

        assert len(empty_df) == 0
        return empty_df

    def create_photon_tile(self, bbox_polygon,
                                 tilename,
                                 date_range = ['2021-01-01','2021-12-31'], # Calendar-year 2021 is the dates we're using for ETOPO. TODO: Change this to read from the config file later.
                                 overwrite = False,
                                 write_stats = True,
                                 verbose = True):
        """If a photon tile doesn't exist yet, download the necessary granules and create it."""
        # If the tile exists, either delete it (if overwrite=True) or read it
        # and return the datafarme (if overwrite=False)
        if type(bbox_polygon) == shapely.geometry.Polygon:
            # Get (xmin, ymin, xmax, ymax)
            bbox_bounds = bbox_polygon.bounds
        else:
            # If it's already a 4-length tuple, assume it's the bbox and go from there.
            assert type(bbox_polygon) in (list, tuple, numpy.ndarray) and len(bbox_polygon) == 4
            bbox_bounds = bbox_polygon

        # Read the tile. If it doesn't exist, create it.
        if os.path.exists(tilename):
            if overwrite:
                os.remove(tilename)
            else:
                try:
                    # If overwrite=False and the file exists, read it and just return the dataframe.
                    return self.read_photon_tile(tilename)
                except KeyboardInterrupt as e:
                    # If we hit a keyboard interrupt while reading this, don't do
                    # anything, just re-raise the interrupt. I'm probably trying
                    # to kill the program halfway through working (just let me).
                    raise e
                except Exception:
                    # If the file is incompliete or somehow corrupted, delete it and we'll try this again.
                    if verbose:
                        print("Error encountered while attempting to read photon tile {0}. Removing it to re-create it.".format(os.path.split(tilename)[1]))
                    os.remove(tilename)

        # Okay, so the tile doesn't exist. Query the icesat-2 photons files that overlap this bounding-box.
        granule_names = nsidc_download.download_granules(short_name=["ATL03", "ATL08"],
                                                         region = bbox_bounds,
                                                         local_dir = self.etopo_config.icesat2_granules_directory,
                                                         dates = date_range,
                                                         download_only_matching_granules = True,
                                                         query_only = True,
                                                         quiet = True)

        atl03_granules = [fn for fn in granule_names if os.path.split(fn)[1].find("ATL03") > -1]
        atl08_granules = [fn for fn in granule_names if os.path.split(fn)[1].find("ATL08") > -1]
        assert len(atl03_granules) == len(atl08_granules)

        # Generate the names of the _photon.h5 files from the ATL03 filenames.
        atl03_photon_db_filenames = [(base + "_photons" + ext) for (base, ext) in [os.path.splitext(fn) for fn in atl03_granules]]

        # Then, see if the _photon.h5 databases exist for all these tiles. If they do, read them.
        photon_dfs = [None]*len(atl03_photon_db_filenames)
        # gdf = None

        if verbose:
            print("Reading {0} _photons.h5 databases to generate {1}.".format(len(atl03_photon_db_filenames), os.path.split(tilename)[1]))
        for i,(photon_db,atl3,atl8) in enumerate(zip(atl03_photon_db_filenames, atl03_granules, atl08_granules)):
            df = None

            # I am making is so the photon databases can be either .h5 or .feather database formats.
            # .h5 (saved flat)
            base, ext = os.path.splitext(photon_db)
            ext = ext.lower()
            if ext == ".h5":
                photon_db_other = base + ".feather"
            else:
                photon_db_other = base + ".h5"

            # Generate a temporary empty textfile to indicate this file is currently being downloaded.
            # This helps prevent multiple processes from downloading the files all at the same time.
            # CAUTION: If this process is terminated or does not complete successfully and the _TEMP_DOWNLOADING.txt
            # file is not removed, this could cause a locking problem where some/all of the processes in subsequent runs are
            # waiting on some non-existent process to supposedly finish downloading. A locking problem.
            # I sort of pre-empted this somewhat by
            # deleting the file in the exception block, but if the code truly is killed instantly
            # (say, if the power suddenly goes out on your computer)
            # the exception block will not be reached and that will not work. This is a somewhat reliable solution
            # under normal operating conditions and handles most typical execution errors, but is not a 100% thread-safe locking solution.
            # A better solution would take more time to implement and this counts as "good enough for now, until the photon database is complete."
            downloading_fbase, downloading_fext = os.path.splitext(photon_db)
            downloading_fname = downloading_fbase + "_TEMP_DOWNLOADING.txt"

            df_is_read = False
            while not df_is_read:
                # If the tile doesn't exist, get the granules needed for it.
                if not (os.path.exists(photon_db) or os.path.exists(photon_db_other)):
                    # If the granules don't exist, download them.

                    # Check for existence of "_TEMP_DOWNLOADING.txt" file.
                    # Skip if this file already exists.
                    if os.path.exists(downloading_fname):
                        time.sleep(1)
                        continue

                    if not os.path.exists(atl3) or not os.path.exists(atl8):

                        # Create an emtpy textfile marking the existence of a process
                        # that is currently downloading the data for this granule.
                        with open(downloading_fname, 'w') as f:
                            f.close()

                        # If either of the granules aren't there, download them from NSIDC
                        try:
                            list_of_files = nsidc_download.download_granules(short_name=["ATL03", "ATL08"],
                                                                             region=bbox_bounds,
                                                                             local_dir = self.etopo_config.icesat2_granules_directory,
                                                                             dates = date_range,
                                                                             download_only_matching_granules = False,
                                                                             query_only = False,
                                                                             fname_filter = os.path.split(atl3)[1][5:], # Get only the files with this granule ID number
                                                                             quiet = not verbose
                                                                             )
                        except Exception as e:
                            # The download_granules() method can return an exception,
                            # or raise one, either way, just assign it to the return
                            # variable to be handled immediately below.
                            list_of_files = e

                        if isinstance(list_of_files, Exception):
                            # If the download script return an exception, or one is raised, delete the file saying that a download is happening.
                            if os.path.exists(downloading_fname):
                                os.remove(downloading_fname)

                            if isinstance(list_of_files, KeyboardInterrupt):
                                # If the session ended because it was killed, then just exit the hell outta here. We're done.
                                raise list_of_files
                            else:
                                # Otherwise, go back to the top of the loop and try again.
                                continue

                    # Create the photon database if it doesn't already exist. (And grab the datframe from it.)
                    df = classify_icesat2_photons.save_granule_ground_photons(atl3,
                                                                              output_db = photon_db,
                                                                              overwrite = False,
                                                                              verbose=verbose)

                    # TODO: Remove the temp downloading file.
                    if os.path.exists(downloading_fname):
                        os.remove(downloading_fname)

                # At this point, the photon database should exist locally. So read it.
                # Then, subset within the bounding box.
                if df is None:
                    try:
                        if ext == ".h5":
                            df = pandas.read_hdf(photon_db, mode='r')
                        else:
                            df = pandas.read_feather(photon_db)
                    except (AttributeError, tables.exceptions.HDF5ExtError):
                        db_to_remove = photon_db if os.path.exists(photon_db) else photon_db_other
                        print("===== ERROR: Photon database {0} corrupted. Will build anew. =====".format(os.path.split(db_to_remove)[1]))
                        print("Removing", db_to_remove)
                        # Remove the corrupted database.
                        os.remove(photon_db)
                        continue
                    except FileNotFoundError:
                        # If the file is not found, try to find the other one. One of them should be in here.
                        if ext == ".h5":
                            df = pandas.read_feather(photon_db_other)
                        else:
                            df = pandas.read_hdf(photon_db_other, mode='r')

                # Select only photons within the bounding box, that are land (class_code==1) or canopy (==2,3) photons
                df_subset = df[df.longitude.between(bbox_bounds[0], bbox_bounds[2], inclusive="left") & \
                               df.latitude.between(bbox_bounds[1], bbox_bounds[3], inclusive="left") & \
                               df.class_code.between(1,3, inclusive="both")]
                photon_dfs[i] = df_subset

                df_is_read = True

        # Now concatenate the databases.
        # If there are no files to concatenate, just read the empty database and return that.
        if len(photon_dfs) == 0:
            tile_df = self.read_empty_tile()
        else:
            tile_df = pandas.concat(photon_dfs, ignore_index=True)
        # Save the database.
        ext_out = os.path.splitext(tilename)[1].lower()

        if ext_out == ".h5":
            tile_df.to_hdf(tilename, "icesat2", complib="zlib", complevel=3, mode='w')
        elif ext_out == ".feather":
            tile_df.to_feather(tilename,
                               compression=self.etopo_config.feather_database_compress_algorithm,
                               compression_level=self.etopo_config.feather_database_compress_level)
        if verbose:
            print(os.path.split(tilename)[1], "written.")

        if write_stats:
            # Write out the stats to a single-record .csv file.
            # For later ingestion into the database.
            summary_csv_fname = os.path.splitext(tilename)[0] + "_summary.csv"
            # Update the database to reflect that this tile is already written.
            # For now, just quickly spit out a csv file.
            data_dict = {'filename': [tilename],
                         'xmin'    : [bbox_bounds[0]],
                         'xmax'    : [bbox_bounds[2]],
                         'ymin'    : [bbox_bounds[1]],
                         'ymax'    : [bbox_bounds[3]],
                         'numphotons'       : [len(tile_df)],
                         'numphotons_canopy': [numpy.count_nonzero(tile_df["class_code"].between(2,3,inclusive="both"))],
                         'numphotons_ground': [numpy.count_nonzero(tile_df["class_code"] == 1)],
                         'is_populated'     : [True]
                         }

            csv_df = pandas.DataFrame(data=data_dict)
            csv_df.to_csv(summary_csv_fname, index=False)
            if verbose:
                print(os.path.split(summary_csv_fname)[1], "written.")

        return tile_df

    def delete_csvs_already_in_database(self, gdf = None, force_read_from_disk=False, verbose=True):
        """Sometimes in parallelization, a tile _summary.csv file gets entered into the database
        but never erased from disk. Go through the database,
        delete any CSV files that are already entered as populated in the database.

        Since (with multiprocessing) this version of the ICESat2_photon_database
        geopackage could be more "recent" than what's on disk, use
        'force_read_from_disk' to only use the version of the gpkg that is on disk,
        not the one in memory here."""
        csv_filenames = [os.path.join(self.tiles_directory,fname) for fname in os.listdir(self.tiles_directory) if (re.search("_summary\.csv\Z", fname) != None)]
        if force_read_from_disk:
            gdf = geopandas.read_file(self.gpkg_fname, mode='r')
            if verbose:
                print(os.path.split(self.gpkg_fname)[1], "read.")
        elif gdf is None:
            gdf = self.get_gdf(verbose=verbose)
        else:
            assert type(gdf) == geopandas.GeoDataFrame

        num_files_removed = 0
        for csv_fname in csv_filenames:
            tile_fname = csv_fname.replace("_summary.csv", ".h5") # Right now this only works if .h5 names are in the database. TOOD: Change for future inclusion of .h5 or .feather.
            gdf_record = gdf.loc[gdf.filename == tile_fname]
            assert len(gdf_record) == 1
            if gdf_record['is_populated'].tolist()[0] == True:
                os.remove(csv_fname)
                num_files_removed += 1

        if verbose:
            print(num_files_removed, "tile _summary.csv files removed.")

        return

    def update_gpkg_with_csvfiles(self, gdf=None,
                                        use_tempfile = True,
                                        delete_when_finished=True,
                                        save_to_disk = True,
                                        verbose=True):
        """Look through the photon tiles directory, look for any "_summary.csv" files that have been written.
        Ingest them into the database.

        Sometimes there creates conflicts when this process is writing the gpkg (which takes a WHILE) and
        another process tries to write to it. Help minimize those conflicts with
        'use_tempfile', which will save the geopackage to a temporary-named file first, and then
        rename it to the self.gpkg_fname when it's done, which is quite fast.
        TODO: Implement some kind of an os-level locking convention for this, to fully avoid conflicts in the future.
        But this should be fast enough to keep them to a bare minimum (a dangerous promise, lol).

        If 'delete_when_finished' is set, delete the _summary.csv files after we've
        included them in the database. This is set by default. If the database gets
        corrupted, it can be rebuit with the
            ICESat2_Database.create_new_geopackage(populate_with_existing_tiles = True)
        method+option. This is slower than reading the summary files, but it works.
        """
        if gdf is None:
            gdf = self.get_gdf(verbose=verbose)

        # Get the filenames from the csv files.
        csv_filenames = [os.path.join(self.tiles_directory,fname) for fname in os.listdir(self.tiles_directory) if (re.search("_summary\.csv\Z", fname) != None)]
        if verbose and len(csv_filenames) > 0:
            print("Found", len(csv_filenames), "csv records to update the tile database. ", end="")

        # print(gdf)

        for csv_fname in csv_filenames:
            # Read the 1-record CSV from the file.
            csv_gdf = pandas.read_csv(csv_fname)
            # print(csv_gdf)
            # print(csv_gdf['filename'])
            # print(csv_gdf['filename'].tolist())
            # print(csv_gdf['filename'].tolist()[0])
            # insert the record into the database.
            gdf_record = gdf.loc[gdf.filename == csv_gdf['filename'].tolist()[0]]
            # print(gdf_record)
            idx = gdf_record.index
            # print(idx)
            # All these records should be the same.
            assert len(gdf_record) == 1
            assert gdf_record['xmin'].tolist()[0] == csv_gdf['xmin'].tolist()[0]
            assert gdf_record['xmax'].tolist()[0] == csv_gdf['xmax'].tolist()[0]
            assert gdf_record['ymin'].tolist()[0] == csv_gdf['ymin'].tolist()[0]
            assert gdf_record['ymax'].tolist()[0] == csv_gdf['ymax'].tolist()[0]
            assert csv_gdf['is_populated'].tolist()[0] == True
            # foobar
            # Update the photon counts.
            gdf.loc[idx,'numphotons'] = csv_gdf['numphotons'].tolist()[0]
            gdf.loc[idx,'numphotons_canopy'] = csv_gdf['numphotons_canopy'].tolist()[0]
            gdf.loc[idx,'numphotons_ground'] = csv_gdf['numphotons_ground'].tolist()[0]
            gdf.loc[idx,'is_populated'] = True

        # Update the gdf we have on record, make sure it matches.
        self.gdf = gdf

        if verbose and len(csv_filenames) > 0:
            print("Done.")

        if len(csv_filenames) > 0 and save_to_disk:
            if verbose:
                print("Writing geopackage...")
            self.save_geopackage(gdf=gdf, use_tempfile = use_tempfile, verbose=verbose)

        if delete_when_finished:
            for csv_fname in csv_filenames:
                os.remove(csv_fname)

        return self.gdf

    def read_photon_tile(self, tilename):
        """Read a photon tile. If the tilename doesn't exist, return None."""
        # Check to make sure this is actually an HDF5 file we're reading.
        ext = os.path.splitext(tilename)[1].lower()
        assert ext in (".h5",".feather")
        # Read it here and return it. Pretty simple.

        # To make the HDF5 and Feather formats basically interchangeable, first look for the one.
        # Then if you can't find it, look for the other.
        if ext == ".h5":
            try:
                return pandas.read_hdf(tilename, mode='r')
            except FileNotFoundError:
                feather_name = os.path.splitext(tilename)[0] + ".feather"
                if os.path.exists(feather_name):
                    return pandas.read_feather(tilename)
                else:
                    return None
        else:
            try:
                return pandas.read_feather(tilename)
            except FileNotFoundError:
                h5_name = os.path.splitext(tilename)[0] + ".h5"
                if os.path.exists(h5_name):
                    return pandas.read_hdf(tilename, mode='r')
                else:
                    return None

    def get_tiling_progress_mapname(self):
        """Output a map of the tiling progress so far.
        This must be called within download_all_icesat2_granules.py to avoid circular import conflicts.
        """
        return os.path.abspath(os.path.splitext(self.gpkg_fname)[0] + "_progress_map.png")

    def update_and_fix_photon_database(self):
        """Sometimes the download_all_icesat2_granules.py -- photon_tiling process
        creates files without updating the database correctly.

        This will loop through all entries in the database, as well as all files, and
        check:
            1) That all entries with "is_populated" actually have valid files associated with them.
            2) That each of those files has the correct number of photons in it, matching up with "numphotons"
            3) That all database files with valid data are included in the database.

        It will fix any errors it finds. If files are corrupted, it will delete them and zero-out the
        entry in the database so they can be rebuilt.
        """
        # TODO: Finish.

if __name__ == "__main__":
    is2db = ICESat2_Database()
    is2db.create_new_geopackage()
    # phd = is2db.get_photon_database((27, 22.5, 27.75, 23))
    # print(phd)
