# -*- coding: utf-8 -*-

"""source_dataset.py -- Defines the definition of the parent class for each subset
dataset.

Each individual dataset source code is inside a folder in /src/datasets with the name of the
dataset, and the code for that package is defined in /src/datasets/DatasetName/source_dataset_DatasetName.py.

The parent class defined here is SourceDataset. The sub-classes defined in each sub-module
should be named SourceDataset_DatasetName and directly inherit SourceDataset.

This allows the code to dynamically ingest and inherit new datasets as they come online rather
than having them all hard-coded."""

# SourceDataset is the base class for each of the source dataset classes that inherit from this.
# This defines the methods that all the child sub-classes should use.

import os
import geopandas
import importlib

###############################################################################
# Import the project /src directory into PYTHONPATH, in order to import all the
# other modules appropriately.
import import_parent_dir; import_parent_dir.import_src_dir_via_pythonpath()
###############################################################################

import utils.configfile
import datasets.dataset_geopackage as dataset_geopackage

def get_source_dataset_object(dataset_name):
    """Given the name of a dataset, import the dataset object from the subdirectory that contains that derived object.
    Using ETOPO source naming convention, the source code will reside in
    datasets.[name].source_dataset_[name].py:source_dataset_[name]"""
    dname = dataset_name.strip()
    module_name = "datasets.{0}.source_dataset_{0}".format(dname)
    module = importlib.import_module(module_name)
    class_obj = getattr(module, "source_dataset_{0}".format(dname))()
    return class_obj


class ETOPO_source_dataset:
    # The base directory of the project is two levels up. Retrieve the absolute path to it on this machine.
    # This file resides in [project_basedir]/src/datasets
    project_basedir = os.path.abspath(os.path.join(os.path.split(__file__)[0], "..", ".."))

    def __init__(self, dataset_name,
                       configfile):
        # Get information from the configuration file. See fields in the [dataset]_config.ini
        self.config = utils.configfile.config(configfile=configfile)

        # Local variables.
        self.dataset_name = dataset_name
        self.geopackage_filename = self.config._abspath(self.config.geopackage_filename)
        # print(self.geopackage_filename)
        # foobar
        self.default_ranking_score = self.config.default_ranking_score

        # The geodataframe of all the tile outlines in the dataset.
        self.geopkg = None

        # The Coordinte Reference System of this dataset. NOTE: All files within the dataset
        # should have the same coordainte reference system.

    def is_active(self):
        """A switch to see if thais dataset is yet being used."""
        # Switch to 'True' when the .ini is filled out and this is ready to go.
        return self.config.is_active

    def get_geopkg_object(self, verbose=True):
        """Get the dataset_geopackage.DatasetGeopackage object associated with this dataset.
        If it doesn't exist, create it.
        """
        if self.geopkg is None:
            self.geopkg = dataset_geopackage.DatasetGeopackage(self.config)
        return self.geopkg

    def get_geodataframe(self, verbose=True):
        """Retrieve the geodataframe of the tile outlines. The geometries are polygons.
        If the dataframe does not exist where it says, it will be created.
        """
        geopkg = self.get_geopkg_object(verbose=verbose)
        return geopkg.get_gdf(verbose=verbose)

    def get_crs(self, as_epsg=True):
        """Get the CRS or EPSG of the coordinate reference system associated with this dataset."""
        gdf = self.get_geodataframe(verbose=False)
        if as_epsg:
            return gdf.crs.to_epsg()
        else:
            return gdf.crs

    def retrieve_list_of_datafiles_within_polygon(self, polygon, polygon_crs, verbose=True):
        """Given a shapely polygon object, return a list of source data files that
        intersect that polygon (even if only partially)."""
        geopkg = self.get_geopkg_object(verbose=verbose)
        subset = geopkg.subset_by_polygon(polygon, polygon_crs)
        return list(subset["filename"])

    def vdatum_shift_original_tiles(self, input_tile_fname,
                                          output_tile_fname,
                                          output_vdatum):
        """If a source tile is not in the needed vertical datum, first shift it before
        regridding it."""


    def create_intermediate_grids(self, etopo_config_obj,
                                        include_ranking = True,
                                        resolution_s = 1,
                                        vdatum_out = "irtf2014",
                                        overwrite = False,
                                        verbose = True):
        """Generate intermeiate grid files from the source dataset.
        This will take the source topo data, re-grid it (using 'waffles') into
        the same grids at the ETOPO dataset, and use the ETOPO empty values for
        all no-data regions.

        If 'include_ranking' is set (default True), it will also create an idential
        grid of ranking for the dataset values.

        These intermediate grids will be combined together (using the ranking scores
        of each source dataset) to create the final ETOPO grids.
        """
        # Step 2: Get the GPKG for each source dataset
        # ds_df = self.get_geodataframe(verbose=verbose)
        # print(ds_df)

        # Step 3: Get the GPKG for the ETOPO grids dataset (from the empty grids.)
        etopo_gpkg = etopo_config_obj.etopo_tile_geopackage_1s if resolution_s == 1 else \
                     etopo_config_obj.etopo_tile_geopackage_15s

        etopo_df = geopandas.read_file(etopo_gpkg)

        # Sort the tiles first by lon, then by lat. This optimizes the ICESat-2 validation later.
        etopo_df.sort_values(['xleft', 'ytop'], ascending=[True, True], inplace=True)
        # print(etopo_df)

        # print([col for col in ds_df.columns])
        # print([col for col in etopo_df.columns])

        # Step 4: Loop through each ETOPO grids dataset feature (each empty-tile DEM)
        etopo_files = etopo_df['filename'].tolist()
        etopo_geometries = etopo_df['geometry'].tolist()

        for i,(efile, egeo) in enumerate(zip(etopo_files, etopo_geometries)):
            # print(i, efile, egeo)
            print('\t',self.retrieve_list_of_datafiles_within_polygon(egeo,
                                                                      polygon_crs=etopo_df.crs,
                                                                      verbose=verbose))
            source_files = [os.path.join(self.config.source_datafiles_directory, fn) \
                            for fn in self.retrieve_list_of_datafiles_within_polygon(egeo,
                                                                                     polygon_crs=etopo_df.crs,
                                                                                     verbose=verbose)
                           ]

            # Resample each source dataset into the ETOPO grid, and combine to make a tile from it, with
            if i>15:
                break
        # Step 5: Check the vertical datum, change it if needed (into a temp file)
        # Step 6: Regrid (waffles) the source dataset to the ETOPO grid.
        #    - Save in the intermediate files directory.
        #    - Fill in empty space with the default nodata value.
        # Step 7: Create empty tile (w/ 16-bit float) for ranking score.
        #    - Fill in non-empty spaces with ranking score.
        pass

    def get_dataset_ranking_score(self, region):
        """Given a polygon region, compute the quality (i.e. ranking) score of the dataset in that region.
        If the dataset contains no files in that region, return the 'default_ranking_score' of
        the dataset, provided in the constructor."""

        # TODO: Use ICESat-2 to calculate accuracies, provide a ranking score spatially through
        # the dataset.
        return self.default_ranking_score

# if __name__ == "__main__":
#     print(get_source_dataset_object("CopernicusDEM"))
