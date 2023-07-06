"""create_data_mask_raster.py -- Create a binary 1/0 mask of valid/ndv data in a polygon. Useful for polygonizing,
subsetting, or extracting data from the raster."""

from osgeo import gdal
import argparse

