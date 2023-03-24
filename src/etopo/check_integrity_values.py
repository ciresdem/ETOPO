#!/usr/bin/env python
import sys
from osgeo import gdal
import os

#####################################################
# Code snippet to import the base directory into
# PYTHONPATH to aid in importing from all the other
# modules in other subdirs.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
#####################################################
import utils.traverse_directory
import utils.configfile
etopo_config = utils.configfile.config()

f = open("corrupt.txt", "a")

dirname = os.path.join(etopo_config.project_base_directory, "data", "ETOPO_2022_release_compressed")
# file_list = utils.traverse_directory.list_files(dirname, regex_match=r'\.((tif)|(nc))\Z')
#
# # print('Checking integrity of', str(sys.argv[1]))
# for fi,fn in enumerate(file_list):
#     if fi < 240:
#         continue
#     print("{0}/{1}".format(fi+1, len(file_list)), fn[(len(dirname) + 1):], end=" ")
#     ds = gdal.Open(fn)
#     for i in range(1,ds.RasterCount+1):
#         srcband = ds.GetRasterBand(i)
#         srcband.Checksum()
#         if gdal.GetLastErrorType() != 0:
#             print('ERROR')
#             f.write(fn + "\n")
#             break
#             # sys.exit(1)
#         else:
#             # print('Now checking stats')
#             stats = srcband.GetStatistics(True, True)
#             # print(stats)
#             if len(stats) == 0 or stats.count(stats[0]) == len(stats):
#                 print('ERROR - NO STATS')
#                 f.write(fn + "\n")
#                 break
#                 # sys.exit(1)
#     print("fine.")

xml_files = utils.traverse_directory.list_files(dirname, regex_match="\.xml\Z")
if len(xml_files) > 0:
    print("Removing", len(xml_files), "xml files.")
    for xml in xml_files:
        os.remove(xml)

# sys.exit(0)
