# -*- coding: utf-8 -*-

"""test_feather_format.py -- Just some code to test out the speed and storage size
of the 'feather' file format, as opposed to the HDF5 format that I'm currently using
for all the ICESat-2 stuff."""

import pandas
import os
import re
import time
import numpy

####################################3
# Include the base /src/ directory of thie project, to add all the other modules.
import import_parent_dir; import_parent_dir.import_src_dir_via_pythonpath()
####################################3
import utils.configfile

my_config = utils.configfile.config()

tile_dir = my_config.icesat2_photon_tiles_directory
granule_dir = my_config.icesat2_granules_directory

tile_test_dir = os.path.join(os.path.split(tile_dir)[0], "test")

N_TO_TEST = 15

print(tile_dir)

read_times = [None] * N_TO_TEST
read_times_table = [None] * N_TO_TEST
read_times_feather = [None] * N_TO_TEST
f_sizes = [None] * N_TO_TEST
f_sizes_table = [None] * N_TO_TEST
f_sizes_feather = [None] * N_TO_TEST

h5_tiles = [os.path.join(tile_dir, fn) for fn in os.listdir(tile_dir) if re.search("photon_tile_([\w\.]+)\.h5\Z", fn) != None][:N_TO_TEST]

# print(len(h5_files_in_tile_dir), "tiles.")
# print(h5_files_in_tile_dir[0:10])

h5_granules = [os.path.join(granule_dir, fn) for fn in os.listdir(granule_dir) if re.search("_photons\.h5\Z", fn) != None][:N_TO_TEST]

# print(len(h5_files_in_granule_dir), "granules.")
# print(h5_files_in_granule_dir[0:10])

# h5_tiles_with_table_names =

for i,h5 in enumerate(h5_tiles):
    t1 = time.time()
    df = pandas.read_hdf(h5, mode='r')
    print(os.path.split(h5)[1], "read.")
    t2 = time.time()
    read_times[i] = t2 - t1
    # print(df.columns.to_list())
    # print(df)
    f_sizes[i] = os.stat(h5).st_size

    print(len(df))

    # First, let's test saving this file in 'table' format rather than 'fixed' hdf5 format.
    table_fname = os.path.join(tile_test_dir, os.path.splitext(os.path.split(h5)[1])[0] + "_table.h5")
    if not os.path.exists(table_fname):
        if len(df) == 0:
            df.to_hdf(table_fname, 'icesat2', mode='w', complib='zlib', complevel=3)
        else:
            df.to_hdf(table_fname, 'icesat2', mode='w', data_columns=True, format='table', complib="zlib", complevel=3)
        print(os.path.split(table_fname)[1], "written.")

    t1 = time.time()
    dft = pandas.read_hdf(table_fname, mode='r')
    t2 = time.time()
    read_times_table[i] = t2 - t1
    f_sizes_table[i] = os.stat(table_fname).st_size

    feather_fname = os.path.join(tile_test_dir, os.path.splitext(os.path.split(h5)[1])[0] + ".feather")
    if not os.path.exists(feather_fname):
        df.reset_index().to_feather(feather_fname, compression="lz4", compression_level = 9)
        print(os.path.split(feather_fname)[1], "written.")

    t1 = time.time()
    dff = pandas.read_feather(feather_fname)
    t2 = time.time()
    read_times_feather[i] = t2 - t1
    f_sizes_feather[i] = os.stat(feather_fname).st_size


print("Read times (.h5)             : {0:0.02f} +/- {1:0.02f}".format(numpy.mean(read_times), numpy.std(read_times)))

print("Read times table (.h5)       : {0:0.02f} +/- {1:0.02f}".format(numpy.mean(read_times_table), numpy.std(read_times_table)))

print("Read times feather (.feather): {0:0.02f} +/- {1:0.02f}".format(numpy.mean(read_times_feather), numpy.std(read_times_feather)))

print("fsize (.h5)              : {0:0.0f} +/- {1:0.0f}".format(numpy.mean(f_sizes), numpy.std(f_sizes)))
print("fsize table (.h5)        : {0:0.0f} +/- {1:0.0f}".format(numpy.mean(f_sizes_table), numpy.std(f_sizes_table)))
print("fsize feather (.feather) : {0:0.0f} +/- {1:0.0f}".format(numpy.mean(f_sizes_feather), numpy.std(f_sizes_feather)))

# These appear to be the results so far. Feather does rapidly increase read times. which is great.
# But it also dramatically increases the file size by 70-80%, which is NOT great.
# Read times (.h5)             : 0.52 +/- 0.39
# Read times table (.h5)       : 1.16 +/- 0.87
# Read times feather (.feather): 0.09 +/- 0.08
# fsize (.h5)              : 32934449 +/- 23305128
# fsize table (.h5)        : 68246527 +/- 48150969
# fsize feather (.feather) : 57520246 +/- 42617562
