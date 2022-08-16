# -*- coding: utf-8 -*-

"""h5_to_feather.py -- Quick code for converting pandas HDF5 files to Feather databases
for faster read/write times.
"""

import pandas
import os
import argparse
import re

####################################
# Include the base /src/ directory of thie project, to add all the other modules.
import import_parent_dir; import_parent_dir.import_src_dir_via_pythonpath()
####################################
import utils.progress_bar
import utils.configfile
import utils.sizeof_format
my_config = utils.configfile.config()

def convert_file(h5_name,
                 feather_name = None,
                 delete_h5 = False,
                 delete_if_corrupted = True,
                 compression = my_config.feather_database_compress_algorithm,
                 comp_level  = my_config.feather_database_compress_level,
                 verbose = True):

    try:
        df = pandas.read_hdf(h5_name)
    except KeyboardInterrupt as e:
        # If the process just got interrupted, don't intervene. Just re-raise it.
        raise e
    except:
        # Otherwise if the file can't be read, delete it.
        if delete_if_corrupted:
            if verbose:
                print(os.path.split(h5_name)[1], "appears corrupted. Deleting.")
            os.remove(h5_name)
        return None

    if feather_name is None:
        feather_name = os.path.splitext(h5_name)[0] + ".feather"
    df.reset_index().to_feather(feather_name, compression = compression, compression_level = comp_level)
    if delete_h5:
        os.remove(h5_name)
        if verbose:
            print(os.path.split(h5_name)[1], "removed.")

    if verbose:
        print(os.path.split(feather_name)[1], "written.")

    return feather_name


def convert_files_in_directory(dirname,
                               h5_regex = "\.h5\Z",
                               delete_h5 = False,
                               delete_if_corrupted = True,
                               compression = my_config.feather_database_compress_algorithm,
                               comp_level  = my_config.feather_database_compress_level,
                               verbose = True):

    h5_files = sorted([os.path.join(dirname, fn) for fn in os.listdir(dirname) if re.search(h5_regex, fn) != None])
    N = len(h5_files)

    for i,h5 in enumerate(h5_files):
        h5_size = os.stat(h5).st_size
        feather_name = convert_file(h5,
                                    delete_h5=delete_h5,
                                    delete_if_corrupted = delete_if_corrupted,
                                    compression = compression,
                                    comp_level = comp_level,
                                    verbose=False)
        if verbose:
            prog_str = ("{0}/{1}" if N<1000 else "{0:,}/{1:,}").format(i+1, N)
            if feather_name:
                feather_size = os.stat(feather_name).st_size
                print(prog_str,
                      os.path.split(feather_name)[1],
                      "{0:>8s} -> {1:>8s}".format(utils.sizeof_format.sizeof_fmt(h5_size),
                                                  utils.sizeof_format.sizeof_fmt(feather_size)),
                      "({0:>6s})".format("{0:+.1f}%".format(100.0 * (feather_size - h5_size) / h5_size)))
            else:
                print(os.path.split(feather_name)[1], "NOT written." + ("" if os.path.exists(h5) else " Deleted."))

def read_and_parse_args():
    parser = argparse.ArgumentParser(description='Utility to convert .h5 pandas databases into .feather pandas databases for more rapid read/write.')
    parser.add_argument("DIRNAME_OR_FILE", help="Location of file or directory to convert.")
    parser.add_argument("--keep_h5", "-k", default=False, action="store_true", help="Keep the original .h5 file in place. (Default: delete after the .feather file is created.)")
    parser.add_argument("--quiet", "-q", default=False, action="store_true", help="Run in quiet mode.")
    return parser.parse_args()

if __name__ == "__main__":
    args = read_and_parse_args()

    assert os.path.exists(args.DIRNAME_OR_FILE)
    if os.path.isdir(args.DIRNAME_OR_FILE):
        convert_files_in_directory(args.DIRNAME_OR_FILE,
                                   delete_h5 = not args.keep_h5,
                                   verbose = not args.quiet)
    else:
        convert_file(args.DIRNAME_OR_FILE,
                     delete_h5= not args.keep_h5,
                     verbose = not args.quiet)

    # dirname = my_config.icesat2_granules_directory
    # convert_files_in_directory(dirname, delete_h5 = True)
