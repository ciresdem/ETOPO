# -*- coding: utf-8 -*-

"""Source code for the BOEM ETOPO source dataset class."""

import os
import shapely.geometry
import pyproj

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset

class source_dataset_BOEM(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "BOEM_config.ini" )):
        """Initialize the BOEM source dataset object."""

        super(source_dataset_BOEM, self).__init__("BOEM", configfile)

    def reproject_boem_into_tiles(self):
        """Reproject the BOEM dataset into lat/lon tiles. It just isn't working in its current projection."""


    def make_datalist_of_boem_tiles(self):
        """Create a 2-entry datalist of the original BOEM tiles."""
        world_poly = shapely.geometry.Polygon(((-180,-90),(-180,90),(180,90),(180,-90),(-180,-90)))
        wgs_crs = pyproj.crs.CRS.from_epsg(4326)
        boem_lines = self.generate_tile_datalist_entries(world_poly, wgs_crs, verbose=True)
        dlist_fname = os.path.join(self.config._abspath(self.config.source_datafiles_directory), "BOEM_tiles.datalist")
        with open(dlist_fname, 'w') as f:
            f.write("\n".join(boem_lines))
            print("\n".join(boem_lines))
            print(dlist_fname, "written.")

        return dlist_fname


if "__main__" == __name__:
    boem = source_dataset_BOEM()
    # boem.make_datalist_of_boem_tiles()
    # poly = shapely.geometry.Polygon(((-105,15),(-105,30),(-90,30),(-90,15),(-105,15)))
    # # poly = shapely.geometry.Polygon(((-180,-90),(-180,90),(180,90),(180,-90),(-180,-90)))
    # print(boem.get_geodataframe())
    # print(boem.get_geodataframe().crs)
    # subset = boem.retrieve_list_of_datafiles_within_polygon(poly, pyproj.crs.CRS.from_epsg(4326))
    # print(subset)