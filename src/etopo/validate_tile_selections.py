# -*- coding: utf-8 -*-

# We have a set of tiles in a CSV file to validate against ICESat-2. Do it here.


import os
import pandas
import geopandas
import pyproj
# import re
# import numpy
# import geopandas as gpd
from shapely import geometry
import argparse
import traceback
import shutil

####################################3
# Include the base /src/ directory of thie project, to add all the other modules.
import import_parent_dir; import_parent_dir.import_src_dir_via_pythonpath()
####################################3

import utils.configfile
# import utils.traverse_directory
import datasets.etopo_source_dataset
import icesat2.validate_dem_collection
import utils.traverse_directory
import icesat2.plot_validation_results

my_config = utils.configfile.config()

tile_list_csv = os.path.abspath(os.path.join(os.path.split(__file__)[0], "..", "..", "scratch_data", "sample_tile_list.csv"))
data_sources = ['Copernicus', 'NASADEM', 'FABDEM', 'AW3D30', "TanDEMX"]


def read_tile_list_csv(csv_file = tile_list_csv):
    """Read the CSV file containing the list of selected tiles to validate.
    Return a geodataframe of the tile locations we wish to validate."""
    df = pandas.read_csv(csv_file, header=0)\
    # If we want to turn this into a geodataframe, complete the code here.
    # (Ignoring for now)
    df['geometry'] = df.apply(build_wkt_bbox, axis=1)
    # Convert the dataframe into a geodataframe with the bounding boxes as geometries
    return geopandas.GeoDataFrame(df, geometry='geometry', crs=pyproj.CRS.from_epsg(4326))

def build_wkt_bbox(df_row):
    xmin = df_row["Xmin"]
    xmax = df_row["Xmax"]
    ymin = df_row["Ymin"]
    ymax = df_row["Ymax"]
    return geometry.Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)])


def copy_all_gebco_coastline_masks(working_dir, dest_dir, regex_str = r"((GEBCO)|(gebco))[\w\.\-]+_coastline_mask\.tif\Z"):
    """Due to their huge size, the GEBCO coastline masks take FOREVER to make
    (they take 8100+ Copernicus files to assemble) and are often repeated in this analysis.
    Find all the ones that have already been created and just copy them into the new working icesat2 directory.
    It's kind of a wonky solution, but should speed things up significantly."""
    list_of_gebco_files = utils.traverse_directory.list_files(working_dir, regex_match=regex_str, include_base_directory = True)
    # Make sure we don't re-copy ones we've already copied.
    list_of_unique_gebco_files = []
    set_of_gebco_files_copied = set()
    for fpath in list_of_gebco_files:
        fname = os.path.split(fpath)[1]
        if fname in set_of_gebco_files_copied:
            continue
        list_of_unique_gebco_files.append(fpath)
        set_of_gebco_files_copied.add(fname)

    for fpath in list_of_unique_gebco_files:
        dest_path = os.path.join(dest_dir, os.path.split(fpath)[1])
        if os.path.exists(dest_path):
            continue
        print("Fetching", os.path.split(fpath)[1], "from other directories.")
        shutil.copyfile(fpath, dest_path)

    return

def validate_tile_selections(tile_csv = tile_list_csv,
                             source_dataset ="CopernicusDEM",
                             working_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","scratch_data","sample_tiles_list"))
                             ):
    """Given a list of grid cells to validate against, get all the source DEMs that
    overlap these boxes and then validate against each one of those.

    It will return a list of .h5 results files generated from these validations."""
    csv_df = read_tile_list_csv(tile_csv)

    dset_object = datasets.etopo_source_dataset.get_source_dataset_object(source_dataset)

    area_names = csv_df["Name"].tolist()
    area_xmins = csv_df["Xmin"].tolist()
    area_ymins = csv_df["Ymin"].tolist()
    polygons = csv_df["geometry"].tolist()

    list_of_results_files = []

    for area_name, xmin, ymin, polygon in zip(area_names, area_xmins, area_ymins, polygons):
        print("========", area_name, ",", source_dataset, "========")
        # First, create the needed directories if they don't yet exist.

        try:

            # Sanity check to make sure our working dir exists.
            assert os.path.exists(working_dir) and os.path.isdir(working_dir)
            # Make /sample_tiles_list/[area_name]
            area_name_dirname = area_name.replace("'","").replace(".","").replace(" ", "")
            dirname = os.path.join(working_dir, area_name_dirname)
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            # Make /sample_tiles_list/[area_name]/icesat2
            icesat2_dir = os.path.join(dirname, "icesat2")
            if not os.path.exists(icesat2_dir):
                os.mkdir(icesat2_dir)
            # Make /sample_tiles_list/[area_name]/[source_dataset] to sub-validation against this specific source dataset.
            dset_subdir = os.path.join(dirname, source_dataset)
            if not os.path.exists(dset_subdir):
                os.mkdir(dset_subdir)

            if source_dataset=="GEBCO":
                copy_all_gebco_coastline_masks(working_dir, icesat2_dir)

            # Get a list of the source tiles within the area.
            list_of_source_files = dset_object.retrieve_list_of_datafiles_within_polygon(polygon, csv_df.crs)
            if len(list_of_source_files) == 0:
                continue

            # Name of the photon h5 to create, based on the area name.
            photon_h5_name = os.path.join(dset_subdir, area_name_dirname.replace(" ", "") + "_" + source_dataset + "_photons.h5")
            # Name of the restuls h5 to create, based on the area name and the source dataset name.
            results_h5_name = os.path.join(dset_subdir, area_name_dirname.replace(" ", "") + "_" + source_dataset + "_results.h5")

            if os.path.exists(results_h5_name):
                print(results_h5_name, "already exists. Moving on...")
                list_of_results_files.append(results_h5_name)
                continue

            # print(area_name, polygon)
            # for fname in list_of_source_files:
                # print("\t", fname)
                # if not os.path.exists(fname):
                    # raise FileNotFoundError(fname, "not found. Prolly a broken path. Exiting.")
            dsconfig = dset_object.config

            if dsconfig.dataset_vdatum_name.strip().lower() == "msl":
                vdatum_in = 3855 # For mean sea level, just swap out with EGM2008 for now.
            else:
                vdatum_in = dsconfig.dataset_vdatum_epsg

            # Do ATL03 and ATL08 granules exist here? If so, we'll skip downloading them again. (This isn't perfect
            # as the exact footprints of the various sets of tiles in each data source aren't identical. However, it's
            # still an optimization that works for us and gives us a signficant enough amount of data.)
            h5_files_in_icesat2_dir = [fn for fn in os.listdir(icesat2_dir) if os.path.splitext(fn)[1] == ".h5"]
            atl03_granules_exist = len([fn for fn in h5_files_in_icesat2_dir if (fn.find("ATL03") >= 0)]) > 0
            atl08_granules_exist = len([fn for fn in h5_files_in_icesat2_dir if (fn.find("ATL08") >= 0)]) > 0

            icesat2_data_exists = atl03_granules_exist and atl08_granules_exist

            icesat2.validate_dem_collection.validate_list_of_dems(list_of_source_files,
                                                                  photon_h5_name,
                                                                  results_h5=results_h5_name,
                                                                  fname_filter=dsconfig.datafiles_regex,
                                                                  output_dir=dset_subdir,
                                                                  icesat2_dir=icesat2_dir,
                                                                  input_vdatum=vdatum_in,
                                                                  output_vdatum="EGM2008",
                                                                  overwrite=False,
                                                                  place_name=area_name_dirname,
                                                                  create_individual_results=False,
                                                                  date_range=["2021-01-01","2021-12-31"],
                                                                  skip_icesat2_download=icesat2_data_exists,
                                                                  delete_datafiles=False,
                                                                  verbose=True
                                                                  )

        except Exception:
            print("-->", area_name, ",", source_dataset, "analysis failed. Here is the error:")
            print(traceback.format_exc())
            print("Moving on....")

        if os.path.exists(results_h5_name):
            list_of_results_files.append(results_h5_name)

def plot_dataset_results(source_dataset="CopernicusDEM",
                         working_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","scratch_data","sample_tiles_list")),
                         plot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","scratch_data","sample_tiles_list", "plots"))):
    # Get all the results tiles from teh working dir for this dataset.
    # regex_match = source_dataset + '/(\w)+' + source_dataset + '(\w)*_results\.h5'
    regex_match = source_dataset + '_results\.h5$'
    results_list = utils.traverse_directory.list_files(working_dir, regex_match=regex_match)

    if len(results_list) == 0:
        return

    plotname = os.path.join(plot_dir, source_dataset+ "_plot.png")
    icesat2.plot_validation_results.plot_histogram_and_error_stats_4_panels(results_list,
                                                                            plotname,
                                                                            place_name = source_dataset)


def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Validate a list of tiles for a given data set.")
    parser.add_argument("dataset_name", nargs="*", help="Name of the dataset(s) to validate against.")
    parser.add_argument("--reverse", default=False, action="store_true",
                        help="""Reverse the order of the datasets. Useful for kicking off two processes at the same time, so they
aren't conflicting with each other.""")
    return parser.parse_args()

if __name__ == "__main__":
    args = define_and_parse_args()

    if len(args.dataset_name) == 0:
        args.dataset_name = ["NASADEM", "CopernicusDEM", "FABDEM", "GEBCO", "TanDEMX", "AW3D30", "ArcticDEM", "REMA"] #, "EMODnet"]

    # for dset in args.dataset_name:
    #     print(dset)
    #     plot_dataset_results(dset)

    # if args.reverse:
    #     args.dataset_name.reverse()

    for dataset in args.dataset_name:
        print("\n========================================================")
        print(dataset)
        validate_tile_selections(source_dataset=dataset)

    # print(read_tile_list_csv())
    # print("NASADEM")
    # validate_tile_selections(source_dataset="NASADEM")
    # print("CopernicusDEM")
    # validate_tile_selections(source_dataset="CopernicusDEM")
    # print("FABDEM")
    # validate_tile_selections(source_dataset="FABDEM")
    # print("TanDEMX")
    # validate_tile_selections(source_dataset="TanDEMX")
    # print("AW3D30")
    # validate_tile_selections(source_dataset="AW3D30")
    # print("ArcticDEM")
    # validate_tile_selections(source_dataset="ArcticDEM")
    # print("REMA")
    # validate_tile_selections(source_dataset="REMA")
    # print("EMODnet")
    # validate_tile_selections(source_dataset="EMODnet")
    # print("GEBCO")
    # validate_tile_selections(source_dataset="GEBCO")
