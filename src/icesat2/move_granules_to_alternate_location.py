# -*- coding: utf-8 -*-

import os
import shutil
import re

####################################
# Include the base /src/ directory of thie project, to add all the other modules.
import import_parent_dir; import_parent_dir.import_src_dir_via_pythonpath()
####################################
import utils.configfile
import utils.progress_bar

my_config = utils.configfile.config()


# Define a quick little function for taking an int/float/str and turning it into a 2-digit zero-padded integer string representation.
# We use this little function several times in the following lines.
def num_to_2digit_str(int_float_or_str):
    return "{0:02d}".format(int(int_float_or_str))


def bytes_available_on_drive(file_location):
    """For a given file/directory location, give the size in bytes of available free space on that drive."""
    st = os.statvfs(file_location)
    return st.f_frsize * st.f_bavail

def move_granules_to_alternate_location(src = my_config.icesat2_granules_directory,
                                        dst = my_config.icesat2_granules_directory_alternate,
                                        region_numbers = [1,2,6,7,8,9,13,14]):
    """Hard drive is filling up, making it difficult to work. Here, move some of the granules off
    to an external hard drive. Specify region numbers so that we can know to only move granules
    that have already been fully-processed into "tile" files, so it won't affect our processing
    further to move them off-disk. Keep tabs on disk-space remaining nad stop moving when the external drive is full."""

    # Will replace [SEGMENT_NUMBER] with the 2-digit segment number here, or the regex to select from a list of segment numbers.
    # Looking either for raw ATL03/ATL08 grnaule names, or the derived "_photons.h5" files generated from them.
    icesat2_granule_template = r"ATL0[38]_(\d){14}_(\d){6}[SEGMENT_NUMBER]_(\d){3}_(\d){2}(_photons)?\.h5"

    if type(region_numbers) in (int, float, str):
        region_str = num_to_2digit_str(region_numbers)

    elif type(region_numbers) in (list, tuple):
        if len(region_numbers) == 1:
            region_str = num_to_2digit_str(region_numbers[0])
        else:
            region_str = "(" + "|".join([("(" + num_to_2digit_str(n) + ")") for n in region_numbers]) + ")"

    # print(region_str)
    # Replace the [SEGMENT_NUMBER] with the string we just constructed above.
    icesat2_granule_search_str = icesat2_granule_template.replace("[SEGMENT_NUMBER]", region_str)
    # print(icesat2_granule_search_str)

    file_matches = [fn for fn in os.listdir(src) if re.search(icesat2_granule_search_str, fn) != None]
    N = len(file_matches)
    print("{0:,} granule files to copy to".format(N), dst)

    for i, fname in enumerate(file_matches):

        fpath_src = os.path.join(src, fname)
        fpath_dst = os.path.join(dst, fname)

        # Check to make sure there's room on disk.
        if os.path.getsize(fpath_src) > bytes_available_on_drive(dst):
            print("\nOut of disk space in", dst + ". Stopping.")
            break

        shutil.move(fpath_src, fpath_dst)
        utils.progress_bar.ProgressBar(i+1, N, suffix=("{0:d}/{1:d}".format(i+1, N)))

if __name__ == "__main__":
    move_granules_to_alternate_location()
