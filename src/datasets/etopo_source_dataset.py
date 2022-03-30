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

###############################################################################
# Import the project /src directory into PYTHONPATH, in order to import all the
# other modules appropriately.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
###############################################################################

import utils.configfile
import datasets.dataset_geopackage as dataset_geopackage

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
        self.geopackage_filename = self.config.geopackage_filename
        self.default_ranking_score = self.config.default_ranking_score

        # The geodataframe of all the tile outlines in the dataset.
        self.geopkg = None

    def is_active(self):
        """A switch to see if thais dataset is yet being used."""
        # Switch to 'True' when the .ini is filled out and this is ready to go.
        return self.config.is_active

    def get_geopkg_object(self, verbose=True):
        """Get the dataset_geopackage.DatasetGeopackage object associated with this dataset.
        If it doesn't exist, create it.
        """
        if not self.geopkg:
            self.geopkg = dataset_geopackage.DatasetGeopackage(self.geopackage_filename,
                                                               base_dir = self.config.source_datafiles_directory)
        return self.geopkg

    def get_geodataframe(self, verbose=True):
        """Retrieve the geodataframe of the tile outlines. The geometries are polygons.
        If the dataframe does not exist where it says, it will be created.
        """
        geopkg = self.get_geopkg_object(verbose=verbose)
        return geopkg.get_gdf(verbose=verbose)

    def retrieve_list_of_datafiles_within_polygon(self, polygon, polygon_crs=None, verbose=True):
        """Given a shapely polygon object, return a list of source data files that
        intersect that polygon (even if only partially)."""
        geopkg = self.get_geopkg_object(verbose=verbose)
        subset = geopkg.subset_by_polygon(polygon,
                                          polygon_crs=polygon_crs,
                                          check_if_same_crs=True)
        return list(subset["filename"])

    def create_intermediate_grids(self, include_ranking = True,
                                        resolution_s = 1,
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
        # TODO: Finish
        # Step 2: Get the GPKG for each source dataset
        # Step 3: Get the GPKG for the ETOPO grids dataset (from the empty grids.)
        # Step 4: Loop through each ETOPO grids dataset feature (each empty-tile DEM)
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
