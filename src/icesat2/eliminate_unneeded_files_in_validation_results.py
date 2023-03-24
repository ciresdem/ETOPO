
# Some files are generated during ETOPO validation that are not needed after the results are done. Delete them here.

import os
import re

dirname = r'/home/mmacferrin/Research/DATA/ETOPO/data/validation_results/15s/2022.09.29'
filenames_all = os.listdir(dirname)
tilenames = sorted([os.path.join(dirname, fn) for fn in filenames_all if (re.search("_2022.09.29.tif\Z", fn) is not None)])

for i, tilename in enumerate(tilenames):
    empty_name = tilename.replace(".tif", "_results_EMPTY.txt")
    results_name = tilename.replace(".tif", "_results.h5")

    num_removed = 0
    if os.path.exists(empty_name):
        # Find all files with this tag except the tilename
        empty_base = os.path.basename(empty_name)
        tagname = empty_base[:-len("_results_EMPTY.txt")]
        other_files_with_this_tagname = [os.path.join(dirname, fn) for fn in filenames_all if ((fn.find(tagname) == 0) and (fn != empty_base))]
        for fn in other_files_with_this_tagname:
            os.remove(fn)
            num_removed += 1

    if os.path.exists(results_name):
        try:
            os.remove(tilename)
            num_removed += 1
        except FileNotFoundError:
            pass

        try:
            os.remove(tilename.replace(".tif", "_coastline_mask_simple.tif"))
            num_removed += 1
        except FileNotFoundError:
            pass

        try:
            os.remove(tilename.replace(".tif", "_coastline_mask.tif"))
            num_removed += 1
        except FileNotFoundError:
            pass

    else:
        try:
            os.remove(tilename)
            num_removed += 1
        except FileNotFoundError:
            pass

    print("{0}/{1} {2}, {3} removed.".format(i+1, len(tilenames), re.search("[NS]\d{2}[EW]\d{3}", os.path.basename(tilename)).group(), num_removed))
