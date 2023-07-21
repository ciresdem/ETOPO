"""clip_polygons_from_dataset.py - A command-line utility for clipping polygons out of a raster dataset and setting
those values to NDV, with the option of interpolating over them."""
import os
import argparse
import geopandas
import pyproj
import shapely.ops


def read_poygon_geoseries(filename: str,
                          merge: bool=True,
                          verbose: bool=True):
    if verbose:
        print("Reading", os.path.basename(filename))
    gdf = geopandas.read_file(filename)
    if len(gdf) == 1 or not merge:
        return gdf
    else:
        if verbose:
            print("Merging", len(gdf), "polygons into one.")
        return geopandas.GeoSeries(shapely.ops.unary_union(gdf.geometry.tolist()), crs=gdf.crs)


def create_list_of_polygons_names_and_tiles():


def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Cut values out of tiles in a raster dataset from polygons defined " +
                                                 "in a vector dataset. Vector file should be in the same projection as raster file.")
    parser.add_argument("dataset_or_raster_name", type=str,
                        help="Name of an ETOPO dataset, or a path to a specific raster file. If a dataset name " +
                             "(such as \"GEBCO\" or \"BlueTopo\"), it will pull all overlapping tiles from that dataset to mask out.")
    parser.add_argument("vector_filename", type=str,
                        help="Name of a vector files (.shp or .gpkg) in which to find polygons.")
    parser.add_argument("-output_dir", "-od", type=str, default="",
                        help="Output directory to write masked raster files. Ignored if '--modify_originals' is used. " +
                             "Defaults to the same directory as each of the input raster files.")
    parser.add_argument("--fill_ndvs", "-f", action="store_true", default=False,
                        help="After masking out polygons areas, Fill in all NDV values using gdal_fillnodata.py.")
    parser.add_argument("--modify_originals", "-orig", action="store_true", default=False,
                        help="Modify the original tiles. If not selected, it will pull out buffer areas from each " +
                             "polygon and save those to (presumably) smaller raster files with values edited into " +
                             "the directory set in -output_dir.")
    parser.add_argument("--overwrite", "-o", action="store_true", default=False,
                        help="Overwrite output files. Default: Just skip generating output files that are already written.")
    parser.add_argument("--quiet", "-q", action="store_true", default=False,
                        help="Run silently.")

    return parser.parse_args()

if __name__ == "__main__":
    args = define_and_parse_args()