## Some ETOPO validation results had a mismatch between the filename and the actual subset of pixels that were validated.
# The bug was fixed but the mis-matched file results weren't. Fix this.

# Also, go ahead and clean out empty tile results that have accompanying "_results_EMPTY.txt" files associated with them. Delete any other files with that same tile tag in them.

import os
import re
from osgeo import gdal
import numpy

dirname = r'/home/mmacferrin/Research/DATA/ETOPO/data/validation_results/15s/2022.09.29'
tilenames = [os.path.join(dirname, fn) for fn in os.listdir(dirname) if (re.search("_2022.09.29.tif\Z", fn) is not None)]

print(len(tilenames), "tiles found validated so far.")

for i,tilename in enumerate(tilenames):
    # Get the geotransform of the image.
    ds = gdal.Open(tilename, gdal.GA_ReadOnly)
    gt = ds.GetGeoTransform()
    ysize = ds.RasterYSize
    # Get rid of the reference to the image dataset.
    del ds
    # Get the lower-left corner of the image. That's what we want.
    llx = gt[0]
    lly = numpy.round(gt[3] + (gt[5]*ysize)) # Since we're dealing with whole-degree 1* images, rounding works here.
    # Create the new tag based on these coordintes.
    new_tag = "{0}{1:02d}{2}{3:03d}".format(("N" if lly >= 0 else "S"),
                                            int(abs(lly)),
                                            ("E" if llx >= 0 else "W"),
                                            int(abs(llx)))

    # Get the existing (old) tag of the image.
    old_tag = re.search("[NS]\d{2}[EW]\d{3}", tilename).group()

    # If they don't match, we need to move the files!
    if old_tag != new_tag:
        print("{0}/{1}".format(i+1, len(tilenames)), old_tag, "->", new_tag)

        # Find all files that contain the old tag.
        files_with_old_tag = [fn for fn in os.listdir(dirname) if re.search("_" + old_tag + "_", fn) is not None]
        for old_fn in files_with_old_tag:
            new_fn = os.path.join(dirname, old_fn.replace(old_tag, new_tag))
            old_fn = os.path.join(dirname, old_fn)
            if os.path.exists(new_fn):
                print("  Removing", os.path.basename(old_fn))
                os.remove(old_fn)
            else:
                print("  Moving", os.path.basename(old_fn), "->", os.path.basename(new_fn))
                os.rename(old_fn, new_fn)

        tilename = tilename.replace(old_tag, new_tag)

    # If the new tag doesn't equal the existing tag, find all files that have the existing tag. Switch them.
    # Move the files to the new names.
    # If the new files already exist, just delete the existing.

    # Finally, look for the _results_EMPTY.txt file with the new tag. If it exists, delete literally all the other files associated with that tag.
    # This will help clean up the directory somewhat while saving progress on already-done EMPTY files.
