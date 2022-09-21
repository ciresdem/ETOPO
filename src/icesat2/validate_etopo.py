# -*- coding: utf-8 -*-
"""validate_etopo.py -- Utility for running post-validation on the entire ETOPO global dataset."""

import os
import random
import re

#####################################################
# Code snippet to import the base directory into
# PYTHONPATH to aid in importing from all the other
# modules in other subdirs.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
#####################################################
import datasets.dataset_geopackage
import utils.configfile
import validate_dem
etopo_config = utils.configfile.config()

def validate_etopo(resolution_s=15,
                   subdir=None,
                   randomize=True,
                   verbose=True):
    outdir = os.path.abspath(os.path.join(etopo_config._abspath(etopo_config.project_base_directory),
                                          "data",
                                          "validation_results",
                                          "{0}s".format(resolution_s)))

    etopo_gdf = datasets.dataset_geopackage.ETOPO_Geopackage(resolution_s).get_gdf(crm_only_if_1s=True,
                                                                                         bed=False,
                                                                                         verbose=verbose)

    # Right now, filter out any tile where we don't have all the icesat-2 photon data yet.
    # Gotta make it north of 82 deg S.
    # print(etopo_gdf.yres)
    etopo_gdf = etopo_gdf[(etopo_gdf.ytop + (etopo_gdf.yres * etopo_gdf.ysize))  >= -82.0].copy()

    etopo_fnames = etopo_gdf.filename.tolist()

    if randomize:
        random.shuffle(etopo_fnames)

    src_dirname = os.path.join(os.path.dirname(etopo_fnames[0].replace("/empty_tiles/", "/finished_tiles/")), subdir)
    files_in_src_dir = [fn for fn in os.listdir(src_dirname) if re.search(r"\.tif\Z", fn) is not None]

    for i, fname in enumerate(etopo_fnames):
        tilename_matches = [os.path.join(src_dirname, fn) for fn in files_in_src_dir if re.search(os.path.splitext(os.path.basename(fname))[0], fn) is not None]
        assert len(tilename_matches) == 1
        tilename = tilename_matches[0]

        if verbose:
            print("========= {0}/{1} {2} ==========".format(i+1, len(etopo_fnames), os.path.basename(tilename)))

        validate_dem.validate_dem_parallel(tilename,
                                           use_icesat2_photon_database=True,
                                           photon_dataframe_name=os.path.join(outdir, os.path.splitext(os.path.basename(fname))[0] + "_results.h5"),
                                           interim_data_dir=outdir,
                                           quiet=not verbose)


if __name__ == "__main__":
    validate_etopo(15, subdir = "2022.09.19")