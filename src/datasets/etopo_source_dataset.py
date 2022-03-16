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
import dataset_geopackage
import importlib

###############################################################################
# Import the project base directory into PYTHONPATH, in order to import all the
# other modules appropriately.
import import_parent_dir
import_parent_dir.import_parent_dir_via_pythonpath()
###############################################################################

class ETOPO_Source_Dataset:
    # The base directory of the project is two levels up. Retrieve the absolute path to it on this machine.
    # This file resides in [project_basedir]/src/datasets
    project_basedir = os.path.abspath(os.path.join(os.path.split(__file__)[0], "..", ".."))

    def __init__(self, dataset_name,
                       geopackage_filename,
                       default_ranking_score = 0):
        # These variables should be defined in the __init__ function of the
        self.dataset_name = dataset_name
        self.geopackage_filename = geopackage_filename
        self.default_ranking_score = default_ranking_score

        # These attributes can remain "None" until they are needed, requiring files
        # to be created or read.

        # The geodataframe of all the tile outlines in the dataset.
        self.geopkg = None

    def get_geodataframe(self, verbose=True):
        """Retrieve the geodataframe of the tile outlines. The geometries are polygons.
        """
        if not self.geopkg:
            self.geopkg = dataset_geopackage.DatasetGeopackage(self.geopackage_filename)

        return self.geopkg.get_gdf(verbose=verbose)


    def retrieve_list_of_datafiles_within_polygon(self, polygon, polygon_crs=None):
        """Given a shapely polygon object, return a list of source data files that
        intersect that polygon (even if only partially)."""
        pass

    def get_dataset_ranking_score(self, region):
        """Given a polygon region, compute the quality (i.e. ranking) score of the dataset in that region.
        If the dataset contains no files in that region, return the 'default_ranking_score' of
        the dataset, provided in the constructor."""

        # TODO: Use ICESat-2 to calculate accuracies, provide a ranking score.
        return self.default_ranking_score
