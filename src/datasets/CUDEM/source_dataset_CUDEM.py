# -*- coding: utf-8 -*-

"""Source code for the CUDEM ETOPO source dataset class."""

import os
import numpy
from osgeo import gdal

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset

class source_dataset_CUDEM(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "CUDEM_config.ini" )):
        """Initialize the CUDEM source dataset object."""

        super(source_dataset_CUDEM, self).__init__("CUDEM", configfile)

    def delete_empty_tiles(self, recreate_if_deleted=True, verbose=True):
        """For some fucking reason, some of the CUDEM converted tiles don't have any valid data in them.

        Find those files and re-convert them."""
        gdf = self.get_geodataframe()

        for i,row in enumerate(gdf.iterrows()):
            _, row = row
            print("{0}/{1} {2}".format(i+1, len(gdf), row.filename[len(os.path.dirname(os.path.dirname(os.path.dirname(row.filename))))+1:]), end="")

            ds = gdal.Open(row.filename, gdal.GA_ReadOnly)
            band = ds.GetRasterBand(1)
            ndv = band.GetNoDataValue()
            array = band.ReadAsArray()
            if numpy.all(array == ndv):
                print(" NODATA.")
                print(" "*3,"Removing", os.path.basename(row.filename))
                band = None
                ds = None
                os.remove(row.filename)

            else:
                print(" fine.")

            band = None
            ds = None

        if recreate_if_deleted:
            self.reproject_tiles_from_nad83(overwrite=False, verbose=verbose)


# If the Geopackage database doesn't exist (i.e. it's been deleted after some new files were created or added), this will create it.
if __name__ == "__main__":
    # gdf = source_dataset_CUDEM().get_geodataframe()
    cudem = source_dataset_CUDEM()
    cudem.delete_empty_tiles()