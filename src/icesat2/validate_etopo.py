# -*- coding: utf-8 -*-
"""validate_etopo.py -- Utility for running post-validation on the entire ETOPO global dataset."""

import os
import random
import re
import subprocess
import argparse
import numpy

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
import etopo.coastline_mask
etopo_config = utils.configfile.config()

def validate_etopo(resolution_s=15,
                   subdir=None,
                   randomize=True,
                   subset_box_of_15d_tiles=False,
                   osm_planet=False,
                   verbose=True):
    outdir = os.path.abspath(os.path.join(etopo_config._abspath(etopo_config.project_base_directory),
                                          "data",
                                          "validation_results",
                                          "{0}s".format(resolution_s)))

    if subdir:
        outdir = os.path.join(outdir, subdir)
        if not os.path.exists(outdir):
            os.mkdir(outdir)

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

    if subdir:
        src_dirname = os.path.join(os.path.dirname(etopo_fnames[0].replace("/empty_tiles/", "/finished_tiles/")), subdir)
    else:
        src_dirname = os.path.dirname(etopo_fnames[0].replace("/empty_tiles/", "/finished_tiles/"))

    files_in_src_dir = [fn for fn in os.listdir(src_dirname) if re.search(r"\.tif\Z", fn) is not None]

    for i, fname in enumerate(etopo_fnames):
        tilename_matches = [os.path.join(src_dirname, fn) for fn in files_in_src_dir
                            if re.search(os.path.splitext(os.path.basename(fname))[0], fn) is not None
                            and (fn.find("_w.tif") == -1)
                            and (fn.find("_bed") == -1)]
        if len(tilename_matches) == 0:
            continue
        assert len(tilename_matches) == 1
        tilename = tilename_matches[0]

        if verbose:
            print("\n========= {0}/{1} {2} ==========".format(i+1, len(etopo_fnames), os.path.basename(tilename)))

        if resolution_s == 15 and subset_box_of_15d_tiles:
            # We're temporarily having issues validating the 15-deg tiles due to too many OSM requests.
            # Take a random 1-degree subset box of that tile, and validate that instead.
            xoff_int = random.randrange(0, 15)
            yoff_int = random.randrange(0, 15)
            xoff = xoff_int * int((3600/15))
            yoff = yoff_int * int((3600/15))
            xsize = int(3600/15)
            ysize = int(3600/15)
            tn_tag = re.search(r"[NS](\d{2})[EW](\d{3})", os.path.basename(tilename)).group()
            y_orig = (-1 if tn_tag[0] == "S" else 1) * int(tn_tag[1:3])
            x_orig = (-1 if tn_tag[3] == "W" else 1) * int(tn_tag[4:])
            subset_tilename = os.path.join(outdir, os.path.basename(tilename).replace(tn_tag,
                                           "1d_{0}{1:02d}{2}{3:03d}".format(tn_tag[0],
                                                                            abs(int(y_orig + 15-yoff_int-1)),
                                                                            tn_tag[3],
                                                                            abs(int(x_orig + xoff_int)))
                                           ))
            gdal_cmd = ["gdal_translate",
                        "-srcwin", str(xoff), str(yoff), str(xsize), str(ysize),
                        tilename, subset_tilename]
            print(" ".join(gdal_cmd))
            subprocess.run(gdal_cmd)
            tilename = subset_tilename

        # Then, make a simple (much quicker) coastline mask to see if this tile has any land before trying to validate it.
        if not does_coastline_mask_contain_land(tilename, verbose=verbose):
            print(os.path.basename(tilename), "does not contain land data. Moving on.")
            empty_fname = os.path.splitext(tilename)[0] + "_results_EMPTY.txt"
            print("Creating", os.path.basename(empty_fname), "to signify no results data here.")
            with open(empty_fname, 'w') as fn:
                fn.close()
            continue

        validate_dem.validate_dem_parallel(tilename,
                                           use_icesat2_photon_database=True,
                                           photon_dataframe_name=os.path.join(outdir, os.path.splitext(os.path.basename(tilename))[0] + "_results.h5"),
                                           mask_out_lakes=True,
                                           mask_out_buildings=True,
                                           use_osm_planet=osm_planet,
                                           interim_data_dir=outdir,
                                           include_photon_level_validation=True,
                                           quiet=not verbose)

def does_coastline_mask_contain_land(tilename, outdir=None, verbose=True):
    """Create a simple (no buildings, no lakes) coastline mask of this tile to see if it contains any land.

    Return True or False."""
    coastline_mask_simple_fname = os.path.splitext(tilename)[0] + "_coastline_mask_simple.tif"
    if outdir:
        coastline_mask_simple_fname = os.path.join(outdir, os.path.basename(coastline_mask_simple_fname))

    _, _, _, _, _, _, coastline_mask_array = etopo.coastline_mask.get_coastline_mask_and_other_dem_data(
                                                                tilename,
                                                                mask_out_lakes = False,
                                                                mask_out_buildings = False,
                                                                use_osm_planet = False,
                                                                include_gmrt = False,
                                                                target_fname_or_dir = coastline_mask_simple_fname,
                                                                verbose=verbose)

    return numpy.any(coastline_mask_array)

def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Validate the ETOPO dataset.")
    parser.add_argument("-subdir", default=None, help="Sub-directory of the 'finished_tiles' directory to validate.")
    parser.add_argument("-resolution", "-r", default=15, type=int, help="Resolution to validate.")
    parser.add_argument("--subset", default=False, action="store_true", help="For efficiency's sake, only validate 2-deg sections of the 15-deg tiles.")
    parser.add_argument("--photon_results", default=False, action="store_true", help="Output photon database results as well.")
    parser.add_argument("--osm_planet", default=False, action="store_true", help="Use the OSM whole-planet file. (Default, do multiple requests to OSM in small tiles.)")
    parser.add_argument("--quiet", default=False, action="store_true", help="Run quietly.")

    return parser.parse_args()

if __name__ == "__main__":
    args = define_and_parse_args()
    validate_etopo(args.resolution,
                   subdir = args.subdir,
                   subset_box_of_15d_tiles=args.subset,
                   osm_planet=args.osm_planet,
                   verbose=not args.quiet)