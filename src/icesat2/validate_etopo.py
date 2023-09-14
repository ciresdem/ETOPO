# -*- coding: utf-8 -*-
"""validate_etopo.py -- Utility for running post-validation on the entire ETOPO global dataset."""

import os
import random
import re
import subprocess
import argparse
import numpy
import shutil
import shapely.geometry
import pandas
import geopandas
import datetime

#####################################################
# Code snippet to import the base directory into
# PYTHONPATH to aid in importing from all the other
# modules in other subdirs.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
#####################################################
import datasets.dataset_geopackage
import utils.configfile
import utils.progress_bar
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
                   just_make_coastline_masks=False,
                   max_photons_per_cell=700,
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
    # etopo_gdf = etopo_gdf[(etopo_gdf.ytop + (etopo_gdf.yres * etopo_gdf.ysize))  >= -84.0].copy()

    etopo_fnames = etopo_gdf.filename.tolist()

    # Key is the latitude of the aboslute value of the y-min of the 15* tile.
    # Value is the number of parallel processes to run.
    latitude_scaling_dictionary = {0: 15,  -15: 15,
                                   15: 15, -30: 15,
                                   30: 15, -45: 15,
                                   45: 12, -60: 12,
                                   60: 7,  -75: 7,
                                   75: 5,  -90: 6}

    # Shuffle the tiles if we're randomizing this.
    if randomize:
        random.shuffle(etopo_fnames)
    else:
        # If we're not randomizing, then reorder them from the equator outward.
        etopo_fnames.sort(key=lambda fn: int(re.search(r"(?<=[NS])\d{2}(?=[EW]\d{3})", fn).group()) - \
                                         (14 if (re.search(r"(?<=_)[NS](?=\d{2}[EW]\d{3})", fn).group() == "S") else 0))

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
            if randomize:
                random.shuffle(k_lists)
            elif run_list_backward:
                k_lists.reverse()
            # k_subset = k_lists[:10]
            # Subset all the lists by the first 10 random subset that aren't yet finished.
            # TODO: Turn this subsetting off (below) when doing the rest of ETOPO validation.
            if just_make_coastline_masks:
                k_subset = k_lists
            else:
                k_subset = [k for k in k_lists if not is_finished(subset_tilenames[k],
                                                                  resolution_s=resolution_s,
                                                                  subdir=subdir)]
                # If we're doing random subsets, just do 10 at a time then move on to the next.
                # But if we're not randomizing, then do them all in that tile.
                if randomize:
                    k_subset = k_subset[:10]
            subset_tilenames = [subset_tilenames[k] for k in k_subset]
            xoffs = [xoffs[k] for k in k_subset]
            yoffs = [yoffs[k] for k in k_subset]

            for j, (sub_tilename, xoff, yoff) in enumerate(zip(subset_tilenames, xoffs, yoffs)):
                if verbose:
                    print("\n== Sub-tile {2}/{3} of big-tile {0}/{1}: {4}".format(
                        i + 1, len(etopo_fnames),
                        j + 1, len(subset_tilenames), os.path.basename(sub_tilename)))

                empty_fname = os.path.splitext(sub_tilename)[0] + "_results_EMPTY.txt"
                results_fname = os.path.splitext(sub_tilename)[0] + "_results.h5"
                if os.path.exists(empty_fname) or os.path.exists(results_fname):
                    if verbose:
                        print("    Already validated.")
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
                ########################################################################
                # Code for just creating land masks (pre-processing) but not actually doing any of the validations.
                elif just_make_coastline_masks:
                    base, ext = os.path.splitext(sub_tilename)
                    simple_cmask = base + "_coastline_mask_simple" + ext
                    normal_cmask = base + "_coastline_mask" + ext
                    if os.path.exists(normal_cmask):
                        continue
                    if verbose:
                        print("Contains land. Creating better coastline mask for validation.")
                    # Generating the coastline mask here, ahead of time, helps the validation processes run more efficiently so they
                    # don't have to generate it themselves.
                    # Only do this if it's above S60 latitude. Over Antarctica, don't bother lookin for lakes or urban areas.
                    tile_id_str = re.search(r"(?<=_)[SN]\d{2}[EW]\d{3}", os.path.basename(sub_tilename)).group()
                    lat = (-1 if tile_id_str[0] == "S" else 1) * int(tile_id_str[1:3])
                    if lat > -60:
                        etopo.coastline_mask.get_coastline_mask_and_other_dem_data(sub_tilename,
                                                                                   mask_out_lakes=True,
                                                                                   mask_out_buildings=False,
                                                                                   mask_out_urban=True,
                                                                                   use_osm_planet=False,
                                                                                   run_in_tempdir=True,
                                                                                   return_coastline_array_only=True,
                                                                                   verbose=verbose)
                    else:
                        # If it's in the Antarctic region, just copy the simple coastline over to the regular coastline.
                        assert os.path.exists(simple_cmask)
                        if not os.path.exists(normal_cmask):
                            if verbose:
                                print("Copying", os.path.basename(simple_cmask), "->", os.path.basename(normal_cmask))
                            shutil.copy(simple_cmask, normal_cmask)
                    continue
                ########################################################################

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
                                                   max_photons_per_cell=max_photons_per_cell,
                                                   numprocs=numprocs_to_use,
                                                   quiet=not verbose)

def does_coastline_mask_contain_land(tilename, outdir=None, verbose=True):
    """Create a simple (no buildings, no lakes) coastline mask of this tile to see if it contains any land.

    Return True or False."""
    coastline_mask_simple_fname = os.path.splitext(tilename)[0] + "_coastline_mask_simple.tif"
    if outdir:
        coastline_mask_simple_fname = os.path.join(outdir, os.path.basename(coastline_mask_simple_fname))

    coastline_mask_array = etopo.coastline_mask.get_coastline_mask_and_other_dem_data(
                                            tilename,
                                            mask_out_lakes = False,
                                            mask_out_buildings = False,
                                            mask_out_urban = False,
                                            use_osm_planet = False,
                                            include_gmrt = False,
                                            run_in_tempdir = True,
                                            target_fname_or_dir = coastline_mask_simple_fname,
                                            return_coastline_array_only=True,
                                            verbose=verbose)

    return numpy.any(coastline_mask_array)


def is_finished(tilename, resolution_s=15, subdir=None, use_number_codes=True):
    """Check the results directory to see if the tilename has already been validated or not.

    Looking for a "_results_EMPTY.txt" or a "_results.h5" file.

    If 'use_number_codes', return 0 for not finished, 1 for finished (empty) and 2 for finished (validated)."""
    outdir = os.path.abspath(os.path.join(etopo_config._abspath(etopo_config.project_base_directory),
                                          "data",
                                          "validation_results",
                                          "{0}s".format(resolution_s)))

    if subdir:
        outdir = os.path.join(outdir, subdir)
        if not os.path.exists(outdir):
            os.mkdir(outdir)

    tilename = os.path.join(outdir, os.path.basename(tilename))
    empty_fname = tilename.replace(".tif", "_results_EMPTY.txt")
    results_fname = tilename.replace(".tif", "_results.h5")

    if use_number_codes:
        if os.path.exists(empty_fname):
            return 1
        elif os.path.exists(results_fname):
            return 2
        else:
            return 0
    else:
        return os.path.exists(empty_fname) or os.path.exists(results_fname)

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
    # etopo_gdf = etopo_gdf[(etopo_gdf.ytop + (etopo_gdf.yres * etopo_gdf.ysize))  >= -84.0].copy()

    etopo_fnames = sorted(etopo_gdf.filename.tolist())

    if subdir:
        src_dirname = os.path.join(os.path.dirname(etopo_fnames[0].replace("/empty_tiles/", "/finished_tiles/")), subdir)
    else:
        src_dirname = os.path.dirname(etopo_fnames[0].replace("/empty_tiles/", "/finished_tiles/"))

    files_in_src_dir = [fn for fn in os.listdir(src_dirname) if re.search(r"\.tif\Z", fn) is not None]

    numtiles_total = 0
    numtiles_finished_empty = 0
    numtiles_finished_results = 0

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
            finished_result = is_finished(tilename, resolution_s=resolution_s, subdir=subdir, use_number_codes=True)
            if finished_result == 1:
                numtiles_finished_empty += 1
            elif finished_result == 2:
                numtiles_finished_results += 1
            numtiles_total += 1

    print("{0:,} ({1:0.2f}%) of {2:,} completed at {3}:\n".format(numtiles_finished_empty + numtiles_finished_results,
                                        (numtiles_finished_empty + numtiles_finished_results) * 100 / numtiles_total,
                                        numtiles_total,
                                        datetime.datetime.now()),
          "    {0:,} ({1:0.2f}%) empty,\n".format(numtiles_finished_empty,
                                                  numtiles_finished_empty * 100 / numtiles_total),
          "    {0:,} ({1:0.2f}%) results,\n".format(numtiles_finished_results,
                                                    numtiles_finished_results * 100 / numtiles_total),
          "    {0:,} ({1:0.2f}%) to go.".format(numtiles_total - (numtiles_finished_empty + numtiles_finished_results),
                                                (numtiles_total - (numtiles_finished_empty + numtiles_finished_results)) * 100 / numtiles_total))


def export_results_gpkg(tile_size_deg=15,
                        validation_tile_size_deg=1,
                        nprocs=10,
                        subdir=None,
                        ):
    """Go through all the results and summary files, look for basic stats, export them to a Geopackage."""
    # Both the directory and the filename have

    gpkg_fname = etopo_config._abspath(os.path.join(etopo_config.etopo_validation_results_directory,
                                                    "ETOPO_SUMMARY_RESULTS_{0}s.gpkg")).format(tile_size_deg)
    results_dir = os.path.abspath(os.path.join(os.path.dirname(gpkg_fname), subdir if subdir else "."))
    # We shouldn't be validating at a lower resolution than the
    assert validation_tile_size_deg <= tile_size_deg

    # Create a full coverage map of the planet at the validation resolution.
    xmins = numpy.arange(-180, 180, validation_tile_size_deg)
    ymins = numpy.arange(-90, 90, validation_tile_size_deg)

    # xymins_mesh = list(zip([a[0] for a in numpy.meshgrid(xmins, ymins)]))
    xymins_mesh = numpy.array(numpy.meshgrid(xmins, ymins)).reshape([2, 180 * 360]).transpose()
    print(xymins_mesh.shape)

    boxes = [shapely.geometry.box(xmin, ymin, xmin + validation_tile_size_deg, ymin + validation_tile_size_deg)
             for (xmin,ymin) in xymins_mesh]

    list_of_files = os.listdir(results_dir)

    def export_gpkg_subproc(sub_xmins, sub_ymins, sub_bboxes, outfile):
        """A sub-process of tallying up a portion of the boxes."""

        sub_statuses = [None] * len(sub_bboxes)
        sub_errors = numpy.zeros((len(sub_bboxes),)) + numpy.nan

        for si, (xmin, ymin, box) in enumerate(zip(sub_xmins, sub_ymins, sub_bboxes)):
            tag = "{0}{1:02d}{2}{3:03d}".format(("S" if ymin < 0 else "N"),
                                                int(abs(ymin)),
                                                ("W" if xmin < 0 else "E"),
                                                int(abs(xmin)))

            results_regex = r"_" + tag + r"[\w\.]*_results.h5\Z"
            summary_regex = r"_" + tag + r"[\w\.]*_results_summary_stats.txt\Z"
            empty_regex = r"_" + tag + r"[\w\.]*_results_EMPTY.txt\Z"

            fnames_with_tag = [fn for fn in list_of_files if re.search(tag, fn) != None]
            fnames_empty = [fn for fn in fnames_with_tag if re.search(empty_regex, fn) != None]
            if len(fnames_empty) > 0:
                assert len(fnames_empty) == 1
                sub_statuses[si] = "EMPTY"
            else:
                fnames_results = [fn for fn in fnames_with_tag if re.search(results_regex, fn) != None]
                fnames_summary = [fn for fn in fnames_with_tag if re.search(summary_regex, fn) != None]
                if (len(fnames_results) == 1) and (len(fnames_summary) == 1):
                    sub_statuses[si] = "DONE"
                    summary_text = open(os.path.join(results_dir, fnames_summary[0]), 'r').read()
                    sub_errors[si] = float(re.findall(r"(?<=RMSE error \(m\): )\d+\.\d+", summary_text)[0])
                elif (len(fnames_results) == 0) and (len(fnames_summary) == 0):
                    sub_statuses[si] = "PENDING"
                else:
                    raise ValueError(f"Error on file conditions for tag {tag}. Should not get here.")

            # utils.progress_bar.ProgressBar(i+1, len(bboxes), suffix="{0:,}/{1:,}".format(i+1, len(boxes)))

        gdf = geopandas.GeoDataFrame(data={"status": sub_statuses,
                                           "rmse": sub_errors,
                                           "geometry": sub_bboxes},
                                     geometry="geometry",
                                     crs="EPSG:4326")

        gdf.to_file(outfile, driver="GPKG", layer="etopo")

    files_per_chunk = round(len(boxes) / (nprocs * 20))
    n_chunks = int(numpy.ceil(len(boxes) / files_per_chunk))
    args_lists = [None] * n_chunks
    outfiles = [None] * n_chunks
    for i in range(n_chunks):
        sublist_slice = slice(i * files_per_chunk, None) if (i == (n_chunks - 1)) \
                        else slice(i * files_per_chunk, (i + 1) * files_per_chunk)
        outfiles[i] = os.path.splitext(gpkg_fname)[0] + "_" + str(i) + ".gpkg"
        args_lists[i] = [xymins_mesh[sublist_slice, 0],
                         xymins_mesh[sublist_slice, 1],
                         boxes[sublist_slice],
                         outfiles[i]]

    parallel_funcs.process_parallel(export_gpkg_subproc,
                                    args_lists,
                                    outfiles=outfiles,
                                    max_nprocs=nprocs,
                                    )

    # Now that we've created all these sub-geopackages, create the master one from all the outfiles
    print("Combining", len(outfiles), "geopackages...")
    sub_gdfs = [geopandas.read_file(fn) for fn in outfiles]
    master_gdf = pandas.concat(sub_gdfs, ignore_index=True)
    master_gdf.to_file(gpkg_fname, driver='GPKG', layer="etopo")
    print(os.path.basename(gpkg_fname), "written with", len(master_gdf), "entries.")
    print("Removing partial gpkgs...", end="")
    for gpkg in outfiles:
        os.remove(gpkg)
    print("Done.")


def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Validate the ETOPO dataset.")
    parser.add_argument("-subdir", default=None, help="Sub-directory of the 'finished_tiles' directory to validate.")
    parser.add_argument("-resolution", "-r", default=15, type=int, help="Resolution to validate (1, 15, 30, 60). Default: 15. Has not been tested on other resolutions yet.")
    parser.add_argument("-numprocs", "-np", default="SCALE_BY_LATITUDE", help="Number of sub-processes to use in validation. Default will scale the processes depending on the latitude to handle memory constraints (look in the code to see or change these values.)")
    parser.add_argument("-repeat", default=1, type=int, help="Repeat the process N times. Helpful if using 'randomize' and/or 'subset' options.")
    parser.add_argument("-max_photons", "-mp", type=int, default=600, help="For efficiency, limit the number of photons that will be processed per grid-cell. Default 600, recommended in the several-hundred range.")
    parser.add_argument("--subset", default=False, action="store_true", help="For efficiency's sake, only validate 1-deg sections of the 15-deg tiles.")
    parser.add_argument("--photon_results", default=False, action="store_true", help="Output photon database results as well.")
    # parser.add_argument("--osm_planet", default=False, action="store_true", help="Use the OSM whole-planet file. (Default, do multiple requests to OSM in small tiles.)")
    parser.add_argument("--randomize", default=False, action="store_true", help="Randomize the order of tiles to process.")
    parser.add_argument("--backward", default=False, action="store_true", help="Start from the back end of the list. This helps have more than 1 process running simultaneously while not interfering with each other. In this case, it will use a 'temp' excecution dir to keep processes from interfering with each other.")
    parser.add_argument("--quiet", default=False, action="store_true", help="Run quietly.")
    parser.add_argument("--coastline_masks_only", "-cmo", default=False, action="store_true", help="Just make the coastline masks, don't actually validate the tiles. Useful for pre-processing coastling masking.")
    parser.add_argument("--export_results", "-ex", default=False, action="store_true", help="Just export the results to the results GeoPackage. Ignore all other options.")
    parser.add_argument("--print_progress", default=False, action="store_true", help="Just print the progress (tiles completed so far). Ignore everything else.")

    return parser.parse_args()

if __name__ == "__main__":
    args = define_and_parse_args()
    if args.print_progress:
        print_progress(resolution_s=args.resolution,
                       subdir=args.subdir,
                       subset_box_of_15d_tiles=args.subset)

    elif args.export_results:
        export_results_gpkg(tile_size_deg=args.resolution,
                            validation_tile_size_deg=((args.resolution/15) if args.subset else args.resolution),
                            nprocs=10 if (args.numprocs == "SCALE_BY_LATITUDE") else int(args.numprocs),
                            subdir=args.subdir,
                            )

    else:
        try:
            numprocs = int(args.numprocs)
        except ValueError:
            # If args.numprocs is None, just keep it None.
            numprocs = args.numprocs

        for step_i in range(args.repeat):

            validate_etopo(args.resolution,
                           subdir = args.subdir,
                           subset_box_of_15d_tiles=args.subset,
                           # osm_planet=args.osm_planet,
                           photon_results=args.photon_results,
                           randomize=args.randomize,
                           run_list_backward=args.backward,
                           max_photons_per_cell=args.max_photons,
                           just_make_coastline_masks=args.coastline_masks_only,
                           numprocs=numprocs,
                           verbose=not args.quiet)

