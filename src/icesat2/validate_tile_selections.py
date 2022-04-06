# -*- coding: utf-8 -*-

# We have a set of tiles in a CSV file to validate against ICESat-2. Do it here.


import os
import pandas
import geopandas
import re
import numpy
# import geopandas as gpd
from shapely import geometry
# import numpy

####################################3
# Include the base /src/ directory of thie project, to add all the other modules.
import import_parent_dir; import_parent_dir.import_src_dir_via_pythonpath()
####################################3

import utils.configfile
import utils.traverse_directory

my_config = utils.configfile.config()

tile_list_csv = os.path.abspath(os.path.join(os.path.split(__file__)[0], "..", "..", "data", "sample_tile_list.csv"))
data_sources = ['Copernicus', 'NASADEM', 'FABDEM', 'AW3D30', "TanDEMX"]


def read_tile_list_csv(csv_file = tile_list_csv):
    """Read the CSV file containing the list of selected tiles to validate.
    Return a geodataframe of the tile locations we wish to validate."""
    df = pandas.read_csv(csv_file, header=0)\
    # If we want to turn this into a geodataframe, complete the code here.
    # (Ignoring for now)
    df['geometry'] = df.apply(build_wkt_bbox, axis=1)
    # Convert the dataframe into a geodataframe with the bounding boxes as geometries
    return geopandas.GeoDataFrame(df, geometry='geometry')

def build_wkt_bbox(df_row):
    xmin = df_row["Xmin"]
    xmax = df_row["Xmax"]
    ymin = df_row["Ymin"]
    ymax = df_row["Ymax"]
    return geometry.Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)])

# def build_tile_list(source="Copernicus", csv_file = tile_list_csv):
#     """Create a list of source data from the given type of DEM, using the CSV file."""
#     csv_df = read_tile_list_csv(csv_file = csv_file)

#     assert source in data_sources

#     if source == "Copernicus":
#         tile_dir = my_config.copernicus_dem_dir
#         tile_size_deg = my_config.copernicus_tile_size_deg
#         regex_search = "\ACopernicus_DSM_COG_\w*_DEM\.tif"
#         lat_regex = "[N|S]\d{2}"
#         lon_regex = "[E|W]\d{3}"

#     elif source == "NASADEM":
#         tile_dir = my_config.nasa_dem_dir
#         tile_size_deg = my_config.nasa_tile_size_deg
#         regex_search = "\ANASADEM_HGT\w*\.tif"
#         lat_regex = "[n|s]\d{2}"
#         lon_regex = "[e|w]\d{3}"

#     elif source == "MERIT":
#         tile_dir = my_config.merit_dem_dir
#         tile_size_deg = my_config.merit_tile_size_deg
#         regex_search = "\A\w*_dem\.tif"
#         lat_regex = "[n|s]\d{2}"
#         lon_regex = "[e|w]\d{3}"

#     elif source == "AW3D30":
#         tile_dir = my_config.aw3d30_dem_dir
#         tile_size_deg = my_config.aw3d30_tile_size_deg
#         regex_search = "\A\w*_DSM\.tif"
#         lat_regex = "[N|S]\d{3}"
#         lon_regex = "[E|W]\d{3}"

#     elif source == "TanDEM-X":
#         raise NotImplementedError("TanDEM-X processing not yet implemented.")

#     else:
#         raise ValueError(source)

#     paths_list = traverse_directory.list_files(tile_dir,
#                                                regex_match=regex_search,
#                                                ordered=True,
#                                                include_base_directory = True)
#     fnames_list = [os.path.split(p)[1] for p in paths_list]

#     latstrings = [re.search(lat_regex, fname).group() for fname in fnames_list]
#     latmins = numpy.array([int(latstr[1:]) * (1 if (latstr[0].upper() == "N") else -1) for latstr in latstrings])
#     lonstrings = [re.search(lon_regex, fname).group() for fname in fnames_list]
#     lonmins = numpy.array([int(lonstr[1:]) * (1 if (lonstr[0].upper() == "E") else -1) for lonstr in lonstrings])

#     if source == "TanDEM-X":
#         # TanDEM-X has strange tile sizes, depending upon latitude.
#         pass
#     else:
#         latmaxs = latmins + tile_size_deg
#         lonmaxs = lonmins + tile_size_deg

#     # tile_list = os.listdir(tile_dir)
#     pandas.set_option('display.max_columns', None)
#     # print(csv_df)

#     tile_list = []

#     for i,trow in csv_df.iterrows():
#         tilemask = (latmins < trow["Ymax"]) & (latmaxs > trow["Ymin"]) & \
#                    (lonmins < trow["Xmax"]) & (lonmaxs > trow["Xmin"])

#         tile_indices = numpy.where(tilemask)[0]
#         if len(tile_indices) == 0:
#             continue
#         assert len(tile_indices) == 1
#         tile_list.append(paths_list[tile_indices[0]])

#         # print(trow)
#         # print([fname for fname,tbit in zip(fnames_list, tilemask) if tbit])
#         # print()
#         # Find the lat/lon pair
#     # print(lats, lons)

#     return tile_list

# def validate_list_of_tiles(sources_list=data_sources, working_dir=os.path.splitext(tile_list_csv)[0]):


if __name__ == "__main__":
    print(read_tile_list_csv())
    # build_tile_list(source="Copernicus")
    # build_tile_list(source="NASADEM")
    # build_tile_list(source="MERIT")
    # build_tile_list(source="AW3D30")
