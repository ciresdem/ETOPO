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
# import utils.progress_bar

class ICESat2_Database:
    """A database to manage ICESat-2 photon clouds for land-surface validations.
    The database is a set of a tiles, managed spatially by a GeoPackage object,
    with each tile being an HDF5 database of all land & canopy photons classified
    from ICESat-2 granules that overlap each tile."""

    def __init__(self, tile_resolution_deg = 0.25):
        """tile_resolutin_deg should be some even fraction of 1. I.e. 1, or 0.5, or 0.25, or 0.1, etc."""
        self.etopo_config = utils.configfile.config()
        self.gpkg_fname = self.etopo_config.icesat2_photon_geopackage
        self.database_directory = self.etopo_config.icesat2_photon_databases_directory

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

    def create_new_geopackage(self, verbose=True):
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
                    tile_fname = os.path.join(self.database_directory, "photon_tile_{0:s}{1:05.2f}_{2:s}{3:06.2f}_{4:s}{5:05.2f}_{6:s}{7:06.2f}.h5".format(\
                    # tile_fname = "photon_tile_{0:s}{1:05.2f}_{2:s}{3:06.2f}_{4:s}{5:05.2f}_{6:s}{7:06.2f}.h5".format(\
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
        sindex = self.gdf.sindex
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


    def save_geopackage(self, gdf=None, verbose=True):
        """After writing or altering data in the geo-dataframe, save it back out to disk.

        If gdf is None, use whatever object is in self.gdf.
        """
        if gdf is None:
            if self.gdf is None:
                raise ValueError("No geodataframe object available to ICESat2_Database.save_gpkg().")
            gdf = self.gdf

        # Write the file.
        gdf.to_file(self.gpkg_fname, layer="icesat2", driver='GPKG')

        if verbose:
            print(os.path.split(self.gpkg_fname)[1], "written with", len(gdf), "entries.")
        return

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
        # TODO: Look at using the spatial index here (use_sindex)
        gdf_subset = gdf[ gdf.intersects(polygon) & ~gdf.touches(polygon)]

        if return_whole_records:
            return gdf_subset
        else:
            return gdf_subset["filename"].tolist()

    def get_photon_database(self, polygon_or_bbox, build_tiles_if_nonexistent = True, verbose=True):
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
                dataframes_list[i] = self.read_photon_tile(fname)
            # If the file doesn't exist, create it and get the data.
            else:
                dataframes_list[i] = self.create_photon_tile(df_row['geometry'],
                                                             fname,
                                                             overwrite=False,
                                                             write_stats_into_database = True,
                                                             verbose=verbose)

        # Concatenate the dataframes together.
        combined_df = pandas.concat(dataframes_list)
        return combined_df

    def create_photon_tile(self, bbox_polygon,
                                 tilename,
                                 date_range = ['2021-01-01','2021-12-31'], # Calendar-year 2021 is the dates we're using for ETOPO. TODO: Change this to read from the config file later.
                                 overwrite = False,
                                 write_stats_into_database = True,
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
        gdf = None

        for i,(photon_db,atl3,atl8) in enumerate(zip(atl03_photon_db_filenames, atl03_granules, atl08_granules)):
            df = None
            # If the tile exists, get it.
            if not os.path.exists(photon_db):
                if not os.path.exists(atl3) or not os.path.exists(atl8):
                    # If either of the granules aren't there, download them from NSIDC
                    nsidc_download.download_granules(short_name=["ATL03", "ATL08"],
                                                     region=bbox_bounds,
                                                     local_dir = self.etopo_confg.icesat2_granules_directory,
                                                     dates = date_range,
                                                     download_only_matching_granules = False,
                                                     query_only = False,
                                                     fname_filter = os.path.split(atl3)[1][5:], # Get only the files with this granule ID number
                                                     quiet = not verbose
                                                     )

                # Create the photon database if it doesn't already exist. (And grab the datframe from it.)
                df = classify_icesat2_photons.save_granule_ground_photons(atl3,
                                                                          output_h5 = photon_db,
                                                                          overwrite = False,
                                                                          verbose=verbose)

            # At this point, the photon database should exist locally. So read it.
            # Then, subset within the bounding box.
            if df is None:
                df = pandas.read_hdf(photon_db, mode='r')
            # Select only photons within the bounding box, that are land (class_code==1) or canopy (==2,3) photons
            df_subset = df[df.longitude.between(bbox_bounds[0], bbox_bounds[2], inclusive="left") & \
                           df.latitude.between(bbox_bounds[1], bbox_bounds[3], inclusive="left") & \
                           df.class_code.between(1,3, inclusive="both")]
            photon_dfs[i] = df_subset

        # Now concatenate the databases.
        tile_df = pandas.concat(photon_dfs)
        # Save the database.
        tile_df.to_hdf(tilename, "icesat2", complib="zlib", complevel=3, mode='w')
        if verbose:
            print(os.path.split(tilename)[1], "written.")

        if write_stats_into_database:
            # Update the database to reflect that this tile is already written.
            if gdf is None:
                gdf = self.get_gdf()

            gdf_record = gdf.loc[gdf.filename == tilename]
            idx = gdf_record.index
            assert len(gdf_record) == 1
            gdf.loc[idx,'numphotons'] = len(tile_df)
            gdf.loc[idx,'numphotons_canopy'] = numpy.count_nonzero(tile_df["class_code"].between(2,3,inclusive="both"))
            gdf.loc[idx,'numphotons_ground'] = numpy.count_nonzero(tile_df["class_code"] == 1)
            gdf.loc[idx,'is_populated'] = True

        self.save_geopackage(gdf=gdf, verbose=False)

        return tile_df

    def read_photon_tile(self, tilename):
        """Read a photon tile. If the tilename doesn't exist, return None."""
        # Check to make sure this is actually an HDF5 file we're reading.
        assert os.path.splitext(tilename)[1].lower() == ".h5"
        # Read it here and return it. Pretty simple.
        return pandas.read_hdf(tilename, mode='r')

if __name__ == "__main__":
    is2db = ICESat2_Database()
    phd = is2db.get_photon_database((27, 22.5, 27.75, 23))
    print(phd)
    # is2db.create_new_geopackage()
