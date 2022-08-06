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

N_TO_TEST = 25

print(tile_dir)

read_times = [None] * N_TO_TEST
read_times_table = [None] * N_TO_TEST
read_times_feather = numpy.empty((N_TO_TEST, 9), dtype=float)
f_sizes = [None] * N_TO_TEST
f_sizes_table = [None] * N_TO_TEST
f_sizes_feather = numpy.empty((N_TO_TEST, 9), dtype=int)

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

    for L in range(1,10):

        feather_fname = os.path.join(tile_test_dir, os.path.splitext(os.path.split(h5)[1])[0] + "_L{0}.feather".format(L))
        if not os.path.exists(feather_fname):
            df.reset_index().to_feather(feather_fname, compression="zstd", compression_level = L)
            print(os.path.split(feather_fname)[1], "written.")

        t1 = time.time()
        dff = pandas.read_feather(feather_fname)
        t2 = time.time()
        read_times_feather[i, L-1] = t2 - t1
        f_sizes_feather[i, L-1] = os.stat(feather_fname).st_size


print("Read times (.h5)             : {0:0.02f} +/- {1:0.02f}".format(numpy.mean(read_times), numpy.std(read_times)))

print("Read times table (.h5)       : {0:0.02f} +/- {1:0.02f}".format(numpy.mean(read_times_table), numpy.std(read_times_table)))

print("fsize (.h5)              : {0:0.0f} +/- {1:0.0f}".format(numpy.mean(f_sizes), numpy.std(f_sizes)))
print("fsize table (.h5)        : {0:0.0f} +/- {1:0.0f}".format(numpy.mean(f_sizes_table), numpy.std(f_sizes_table)))

for L in range(1,10):
    print("Read times feather (.feather) L{2}: {0:0.02f} +/- {1:0.02f}".format(numpy.mean(read_times_feather[:,L-1]), numpy.std(read_times_feather[:,L-1]), L))

for L in range(1,10):
    print("fsize feather (.feather)      L{2}: {0:0.0f} +/- {1:0.0f}".format(numpy.mean(f_sizes_feather[:,L-1]), numpy.std(f_sizes_feather[:,L-1]), L))


# These appear to be the results so far. Feather does rapidly increase read times. which is great.
# But it also dramatically increases the file size by 70-80%, which is NOT great.
# However, with zstd compression, it's actually MUCH better.

# Compression options:
    # Table: "zlib", 3
    # Feather: "lz4", 9

# Read times (.h5)             : 0.52 +/- 0.39
1# Read times table (.h5)       : 1.16 +/- 0.87
# Read times feather (.feather): 0.09 +/- 0.08
# fsize (.h5)              : 32934449 +/- 23305128
# fsize table (.h5)        : 68246527 +/- 48150969
# fsize feather (.feather) : 57520246 +/- 42617562

# Compression options:
    # Table: "zlib", 3
    # Feather: "zstd", 1-9
# Read times (.h5)             : 0.58 +/- 0.60
# Read times table (.h5)       : 1.32 +/- 1.40
# fsize (.h5)              : 30235391 +/- 31202004
# fsize table (.h5)        : 62416213 +/- 64557752
# Read times feather (.feather) L1: 0.10 +/- 0.11
# Read times feather (.feather) L2: 0.09 +/- 0.11
# Read times feather (.feather) L3: 0.10 +/- 0.11
# Read times feather (.feather) L4: 0.10 +/- 0.11
# Read times feather (.feather) L5: 0.10 +/- 0.11
# Read times feather (.feather) L6: 0.10 +/- 0.11
# Read times feather (.feather) L7: 0.10 +/- 0.11
# Read times feather (.feather) L8: 0.10 +/- 0.11
# Read times feather (.feather) L9: 0.10 +/- 0.12
# fsize feather (.feather)      L1: 40204141 +/- 42467611
# fsize feather (.feather)      L2: 34101591 +/- 34085345
# fsize feather (.feather)      L3: 32566425 +/- 33218722
# fsize feather (.feather)      L4: 32521921 +/- 33114279
# fsize feather (.feather)      L5: 32484293 +/- 33099152
# fsize feather (.feather)      L6: 32505758 +/- 33121000
# fsize feather (.feather)      L7: 32383634 +/- 32934131
# fsize feather (.feather)      L8: 32350690 +/- 32907205
# fsize feather (.feather)      L9: 32347140 +/- 32903369
