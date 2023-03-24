# There's a bug in the coastline masking right now, causing *some* coastline files to be erroneously created with all-zero
# values. In this case, go through a directory, eliminate all coastline_mask.tif files that have all-zero values in them,
# and eliminate all _EMTPY.txt files associated with them.

# Then I need to remove all cached temp-files, and re-run the analysis.

import os
import argparse
import subprocess

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import utils.traverse_directory
import utils.progress_bar

def remove_empty_files(dirname: str):

    coastline_masks = utils.traverse_directory.list_files(dirname, regex_match="_coastline_mask\.tif\Z")

    num_deleted = 0

    for i, fname in enumerate(coastline_masks):
        if i < 615:
            continue

        gdal_args = ["gdalinfo", fname, "-stats"]
        p = subprocess.run(gdal_args, capture_output=True, text=True)

        utils.progress_bar.ProgressBar(i+1, len(coastline_masks), suffix="{0}/{1}".format(i+1, len(coastline_masks)))

        # If there are 1 values in this, just skip and move alone. We're good.
        if p.stdout.find("STATISTICS_MAXIMUM=1") >= 0:
            # Get rid of the XML that was just created.
            if os.path.exists(fname + ".aux.xml"):
                os.remove(fname + ".aux.xml")
            continue

        # But if it's all zeros, then we have some doing to do.
        try:
            assert p.stdout.find("STATISTICS_MAXIMUM=0") >= 0
        except AssertionError as e:
            print (fname)
            raise e

        # Delete the file.
        os.remove(fname)
        num_deleted += 1

        fbase = os.path.basename(fname)[0:fname.find("_coastline_mask.tif")]
        dirname = os.path.dirname(fname)
        # Find all other files associated with this file in the same directory.
        other_files_with_this_base = [os.path.join(dirname, fn) for fn in os.listdir(dirname) if (fn.find(fbase) == 0)]

        for of in other_files_with_this_base:
            # Remove the empty results file associated with this file.
            if os.path.basename(of).find("_EMPTY.txt") > 0:
                os.remove(of)
            # Also, remove the XML file associated with the gdal process we just ran.
            elif os.path.basename(of).find(".aux.xml") > 0:
                os.remove(of)

    print(num_deleted, "empty coastline files (and associated files) deleted.")

def test_coastline_file_and_delete_if_empty(fname: str):
    gdal_args = ["gdalinfo", fname, "-stats"]
    p = subprocess.run(gdal_args, capture_output=True, text=True)

    # If there are 1 values in this, just skip and move alone. We're good.
    if p.stdout.find("STATISTICS_MAXIMUM=1") >= 0:
        # Get rid of the XML that was just created.
        if os.path.exists(fname + ".aux.xml"):
            os.remove(fname + ".aux.xml")
        return False

    # But if it's all zeros, then we have some doing to do.
    assert p.stdout.find("STATISTICS_MAXIMUM=0") >= 0

    # Delete the file.
    os.remove(fname)

    fbase = os.path.basename(fname)[0:fname.find("_coastline_mask.tif")]
    dirname = os.path.dirname(fname)
    # Find all other files associated with this file in the same directory.
    other_files_with_this_base = [os.path.join(dirname, fn) for fn in os.listdir(dirname) if (fn.find(fbase) == 0)]

    for of in other_files_with_this_base:
        # Remove the empty results file associated with this file.
        if os.path.basename(of).find("_EMPTY.txt") > 0:
            os.remove(of)
        # Also, remove the XML file associated with the gdal process we just ran.
        elif os.path.basename(of).find(".aux.xml") > 0:
            os.remove(of)

    return True


def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Remove any empty 'coastline_mask.tif' files with all-zero values. Also remove _EMPTY.txt files associated with them.")
    parser.add_argument("dirname_or_filename", help="The directory name in which to search recursively for empty coastline_mask.tif files.")

    return parser.parse_args()

if __name__ == "__main__":
    args = define_and_parse_args()
    path = args.dirname_or_filename

    if os.path.isdir(path):
        remove_empty_files(path)

    else:
        test_coastline_file_and_delete_if_empty(path)