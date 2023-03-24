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
import utils.parallel_funcs as parallel_funcs
etopo_config = utils.configfile.config()

def validate_etopo(resolution_s=15,
                   subdir=None,
                   randomize=True,
                   subset_box_of_15d_tiles=False,
                   photon_results=False,
                   run_list_backward=False,
                   numprocs="SCALE_BY_LATITUDE", #parallel_funcs.physical_cpu_count(),
                   # osm_planet=False,
                   verbose=True):
    """Numprocs can be an integer number >= 1, or the string "SCALE_BY_LATITUDE", which will scale the number of subprocessing
    inversely to the latitude, for handling the larger tile sizes there."""

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
    # Gotta make it north of 84 deg S.
    # print(etopo_gdf.yres)
    etopo_gdf = etopo_gdf[(etopo_gdf.ytop + (etopo_gdf.yres * etopo_gdf.ysize))  >= -84.0].copy()

    etopo_fnames = etopo_gdf.filename.tolist()

    # Key is the latitude of the y-min of the 15* tile.
    # Value is the number of parallel processes to run.
    latitude_scaling_dictionary = {0: 15,  -15: 15,
                                   15: 15, -30: 15,
                                   30: 15, -45: 15,
                                   45: 12, -60: 12,
                                   60: 8,  -75: 8,
                                   75: 5,  -90: 5}

    # Shuffle the tiles if we're randomizing this.
    if randomize:
        random.shuffle(etopo_fnames)

    if subdir:
        src_dirname = os.path.join(os.path.dirname(etopo_fnames[0].replace("/empty_tiles/", "/finished_tiles/")), subdir)
    else:
        src_dirname = os.path.dirname(etopo_fnames[0].replace("/empty_tiles/", "/finished_tiles/"))

    files_in_src_dir = [fn for fn in os.listdir(src_dirname) if re.search(r"\.tif\Z", fn) is not None]

    if run_list_backward:
        etopo_fnames.reverse()

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
            # xoff_int = random.randrange(0, 15)
            # yoff_int = random.randrange(0, 15)
            # xoff = xoff_int * int((3600/15))
            # yoff = yoff_int * int((3600/15))
            xoff_int = numpy.arange(15, dtype=int)
            yoff_int = numpy.arange(15, dtype=int)
            xsize = int(3600/15)
            ysize = int(3600/15)
            # Put in xoff, yoff here, but as arrays.
            xoff_int_grid, yoff_int_grid = numpy.meshgrid(xoff_int, yoff_int)
            xoff_int_grid = xoff_int_grid.flatten()
            yoff_int_grid = yoff_int_grid.flatten()
            xoffs = xoff_int_grid * xsize
            yoffs = yoff_int_grid * ysize
            tn_tag = re.search(r"[NS](\d{2})[EW](\d{3})", os.path.basename(tilename)).group()
            y_orig = (-1 if tn_tag[0] == "S" else 1) * int(tn_tag[1:3])
            x_orig = (-1 if tn_tag[3] == "W" else 1) * int(tn_tag[4:])
            subset_tilenames = [os.path.join(outdir, os.path.basename(tilename).replace(tn_tag,
                                                "1d_{0}{1:02d}{2}{3:03d}".format(tn_tag[0],
                                                                            abs(int(y_orig + 15-yoi-1)),
                                                                            tn_tag[3],
                                                                            abs(int(x_orig + xoi)))
                                           )) for (xoi, yoi) in zip(xoff_int_grid, yoff_int_grid)]

            # For now, just pick 10 random sub-tiles of the 225 sub-tiles. Validate those, then move on.
            k_lists = list(range(len(subset_tilenames)))
            random.shuffle(k_lists)
            k_subset = k_lists[:10]
            # Subset all the lists by the first 10 random subset
            subset_tilenames = [subset_tilenames[k] for k in k_subset]
            xoffs = [xoffs[k] for k in k_subset]
            yoffs = [yoffs[k] for k in k_subset]
            subset_tilenames = subset_tilenames[:10] # TODO: Turn this subsetting off when doing the rest of ETOPO validation.

            for j,(sub_tilename, xoff, yoff) in enumerate(zip(subset_tilenames, xoffs, yoffs)):
                if verbose:
                    print("\n== Sub-tile {2}/{3} of big-tile {0}/{1}: {4}".format(
                        i+1, len(etopo_fnames),
                        j+1, len(subset_tilenames), os.path.basename(sub_tilename)))

                empty_fname = os.path.splitext(sub_tilename)[0] + "_results_EMPTY.txt"
                results_fname = os.path.splitext(sub_tilename)[0] + "_results.h5"
                if os.path.exists(empty_fname) or os.path.exists(results_fname):
                    continue

                if not os.path.exists(sub_tilename):
                    gdal_cmd = ["gdal_translate",
                                "-srcwin", str(xoff), str(yoff), str(xsize), str(ysize),
                                tilename, sub_tilename]
                    print(" ".join(gdal_cmd))
                    subprocess.run(gdal_cmd)
                    # tilename = subset_tilename

                # # Then, make a simple (much quicker) coastline mask to see if this tile has any land before trying to validate it.
                if not does_coastline_mask_contain_land(sub_tilename, verbose=verbose):
                    print(os.path.basename(sub_tilename), "does not contain land data. Moving on.")
                    print("Creating", os.path.basename(empty_fname), "to signify no results data here.")
                    with open(empty_fname, 'w') as fn:
                        fn.close()
                    continue

                if numprocs == "SCALE_BY_LATITUDE":
                    numprocs_to_use = latitude_scaling_dictionary[y_orig]
                elif type(numprocs) in (int,float) and numprocs >= 1:
                    numprocs_to_use = int(numprocs)
                else:
                    raise ValueError("Illegal number of processes: {0}".format(numprocs))

                validate_dem.validate_dem_parallel(sub_tilename,
                                                   use_icesat2_photon_database=True,
                                                   photon_dataframe_name=os.path.join(outdir, os.path.splitext(os.path.basename(tilename))[0] + "_results.h5"),
                                                   mask_out_lakes=True,
                                                   mask_out_buildings=False,
                                                   mask_out_urban=True,
                                                   use_osm_planet=False,
                                                   interim_data_dir=outdir,
                                                   include_photon_level_validation=photon_results,
                                                   measure_coverage=True,
                                                   overwrite=False,
                                                   numprocs=numprocs_to_use,
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
                                                                mask_out_urban = False,
                                                                use_osm_planet = False,
                                                                include_gmrt = False,
                                                                run_in_tempdir = True,
                                                                target_fname_or_dir = coastline_mask_simple_fname,
                                                                verbose=verbose)

    return numpy.any(coastline_mask_array)

def print_progress(resolution_s=15,
                   subdir=None,
                   subset_box_of_15d_tiles=False):
    """Print out the total percentage of tiles that have been finished processing. Looking for a "_results_EMPTY.txt" or a "_results.h5" file."""

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
                                                                                   verbose=False)

    # Right now, filter out any tile where we don't have all the icesat-2 photon data yet.
    # Gotta make it north of 84 deg S.
    # print(etopo_gdf.yres)
    etopo_gdf = etopo_gdf[(etopo_gdf.ytop + (etopo_gdf.yres * etopo_gdf.ysize))  >= -84.0].copy()

    etopo_fnames = sorted(etopo_gdf.filename.tolist())

    if subdir:
        src_dirname = os.path.join(os.path.dirname(etopo_fnames[0].replace("/empty_tiles/", "/finished_tiles/")), subdir)
    else:
        src_dirname = os.path.dirname(etopo_fnames[0].replace("/empty_tiles/", "/finished_tiles/"))

    files_in_src_dir = [fn for fn in os.listdir(src_dirname) if re.search(r"\.tif\Z", fn) is not None]

    numtiles_total = 0
    numtiles_finished = 0

    for i, fname in enumerate(etopo_fnames):
        tilename_matches = [os.path.join(src_dirname, fn) for fn in files_in_src_dir
                            if re.search(os.path.splitext(os.path.basename(fname))[0], fn) is not None
                            and (fn.find("_w.tif") == -1)
                            and (fn.find("_bed") == -1)]
        if len(tilename_matches) == 0:
            continue
        assert len(tilename_matches) == 1
        tilename = tilename_matches[0]

        # if verbose:
        #     print("\n========= {0}/{1} {2} ==========".format(i+1, len(etopo_fnames), os.path.basename(tilename)))

        if resolution_s == 15 and subset_box_of_15d_tiles:
            # We're temporarily having issues validating the 15-deg tiles due to too many OSM requests.
            # Take a random 1-degree subset box of that tile, and validate that instead.
            # xoff_int = random.randrange(0, 15)
            # yoff_int = random.randrange(0, 15)
            # xoff = xoff_int * int((3600/15))
            # yoff = yoff_int * int((3600/15))
            xoff_int = numpy.arange(15, dtype=int)
            yoff_int = numpy.arange(15, dtype=int)
            xsize = int(3600/15)
            ysize = int(3600/15)
            # Put in xoff, yoff here, but as arrays.
            xoff_int_grid, yoff_int_grid = numpy.meshgrid(xoff_int, yoff_int)
            xoff_int_grid = xoff_int_grid.flatten()
            yoff_int_grid = yoff_int_grid.flatten()
            xoffs = xoff_int_grid * xsize
            yoffs = yoff_int_grid * ysize
            tn_tag = re.search(r"[NS](\d{2})[EW](\d{3})", os.path.basename(tilename)).group()
            y_orig = (-1 if tn_tag[0] == "S" else 1) * int(tn_tag[1:3])
            x_orig = (-1 if tn_tag[3] == "W" else 1) * int(tn_tag[4:])
            subset_tilenames = [os.path.join(outdir, os.path.basename(tilename).replace(tn_tag,
                                                "1d_{0}{1:02d}{2}{3:03d}".format(tn_tag[0],
                                                                            abs(int(y_orig + 15-yoi-1)),
                                                                            tn_tag[3],
                                                                            abs(int(x_orig + xoi)))
                                           )) for (xoi, yoi) in zip(xoff_int_grid, yoff_int_grid)]

            tilenames = subset_tilenames
        else:
            tilenames = [os.path.join(outdir, os.path.basename(tilename))]

        for tilename in tilenames:
            empty_fname = tilename.replace(".tif", "_results_EMPTY.txt")
            results_fname = tilename.replace(".tif", "_results.h5")
            if os.path.exists(empty_fname) or os.path.exists(results_fname):
                numtiles_finished += 1
            numtiles_total += 1

    print(numtiles_finished, "of", numtiles_total, "completed. ({0:0.2f}%)".format(numtiles_finished * 100 / numtiles_total))


def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Validate the ETOPO dataset.")
    parser.add_argument("-subdir", default=None, help="Sub-directory of the 'finished_tiles' directory to validate.")
    parser.add_argument("-resolution", "-r", default=15, type=int, help="Resolution to validate.")
    parser.add_argument("-numprocs", "-np", default="SCALE_BY_LATITUDE", help="Number of sub-processes to use in validation. Default will scale the processes depending on the latitude to handle memory constraints (look in the code to see or change these values.)")
    parser.add_argument("-repeat", default=1, type=int, help="Repeat the process N times. Helpful if using 'randomize' and/or 'subset' options.")
    parser.add_argument("--subset", default=False, action="store_true", help="For efficiency's sake, only validate 1-deg sections of the 15-deg tiles.")
    parser.add_argument("--photon_results", default=False, action="store_true", help="Output photon database results as well.")
    # parser.add_argument("--osm_planet", default=False, action="store_true", help="Use the OSM whole-planet file. (Default, do multiple requests to OSM in small tiles.)")
    parser.add_argument("--randomize", default=False, action="store_true", help="Randomize the order of tiles to process.")
    parser.add_argument("--backward", default=False, action="store_true", help="Start from the back end of the list. This helps have more than 1 process running simultaneously while not interfering with each other. In this case, it will use a 'temp' excecution dir to keep processes from interfering with each other.")
    parser.add_argument("--quiet", default=False, action="store_true", help="Run quietly.")
    parser.add_argument("--print_progress", default=False, action="store_true", help="Just print the progress (tiles completed so far). Ignore everything else.")

    return parser.parse_args()

if __name__ == "__main__":
    args = define_and_parse_args()
    if args.print_progress:
        print_progress(resolution_s=args.resolution,
                       subdir=args.subdir,
                       subset_box_of_15d_tiles=args.subset)
        import sys
        sys.exit(0)

    try:
        numprocs = int(args.numprocs)
    except ValueError:
        numprocs = args.numprocs

    for step_i in range(args.repeat):

        validate_etopo(args.resolution,
                       subdir = args.subdir,
                       subset_box_of_15d_tiles=args.subset,
                       # osm_planet=args.osm_planet,
                       photon_results=args.photon_results,
                       randomize=args.randomize,
                       run_list_backward=args.backward,
                       numprocs=numprocs,
                       verbose=not args.quiet)

