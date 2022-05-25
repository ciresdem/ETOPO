# -*- coding: utf-8 -*-

import os
import numpy
import argparse
import multiprocessing
import time
import re
import shutil
import pandas
import datetime
import tables.exceptions

####################################3
# Include the base /src/ directory of thie project, to add all the other modules.
import import_parent_dir; import_parent_dir.import_src_dir_via_pythonpath()
####################################3
import utils.configfile
my_config = utils.configfile.config()
import icesat2.nsidc_download as nsidc_download
import icesat2.classify_icesat2_photons as classify_icesat2_photons
import icesat2.icesat2_photon_database as icesat2_photon_database
import etopo.generate_empty_grids
import utils.progress_bar
import utils.sizeof_format

# ICESAT2_DIR = os.path.join(my_config.etopo_cudem_cache_directory, ".cudem_cache")
ICESAT2_DIR = my_config.icesat2_granules_directory


def move_all_granules_to_icesat2_granules_dir(old_dir=os.path.join(my_config.etopo_cudem_cache_directory, ".cudem_cache"),
                                              new_dir = ICESAT2_DIR,
                                              h5_regex = "ATL(\w+)\.h5\Z"):
    """I'd previously been putting all the ICESAT-2 granules in /scratch_data/.cudem_cache with the
    waffles cached files. I decided I want them in the data/icesat2/granules directory instead. So
    I'm gonna move all the existing icesat-2 files over there."""
    icesat2_files = [fn for fn in os.listdir(old_dir) if re.search(h5_regex, fn) != None]

    if len(icesat2_files) == 0:
        print("No existing icesat-2 files in", old_dir + ". Exiting.")
        return

    print ("Moving", len(icesat2_files), "files to new ICESat-2 directory.")
    for i,fname in enumerate(icesat2_files):
        old = os.path.join(old_dir, fname)
        new = os.path.join(new_dir, fname)
        shutil.move(old, new)
        utils.progress_bar.ProgressBar(i+1, len(icesat2_files), suffix="{0}/{1}".format(i+1, len(icesat2_files)))


def create_query_bounding_boxes(xmin=-180, xmax=180, xres=1, ymin=-89, ymax=89, yres=1):
    """We can't query & download the whole entire world instantly. So, let's
    generate a series of 5x5-degree bounding boxes in which to query for data.
    """
    bboxes = []

    assert xmin < xmax
    assert ymin < ymax

    # Go either direction from the equator. Get to the polar regions last.
    if ymax > 0 and ymin < 0:
        y_vals = list(zip(range(0,ymax,yres), range(-1,ymin,-1)))
        y_vals = numpy.array(y_vals).flatten()
    else:
        y_vals = range(ymin, ymax, yres)

    for y in y_vals:
        for x in range(xmin, xmax, xres):
            bbox = (x,y,x+1,y+1)
            bbox = tuple([float(n) for n in bbox])
            bboxes.append(bbox)

    return bboxes

def create_icesat2_progress_csv(fname=my_config.icesat2_download_progress_csv,
                                verbose=True):
    """Create a CSV file so that we can keep track of which icesat-2 files
    have been downloaded already and which ones have not. This is much more efficient than
    contantly querying NSIDC again and again to see if we have the files for every
    given 1x1-deg tile over the planet.
    """
    if verbose:
        print("Fetching all ICESat-2 1x1-deg bounding boxes. This can take a sec.")
    tile_tuples = etopo.generate_empty_grids.create_list_of_tile_tuples(resolution=1, verbose=False)
    bboxes = [(int(x),int(y),int(x+1),int(y+1)) for (y,x) in tile_tuples]
    df = pandas.DataFrame(data={"xmin": [bb[0] for bb in bboxes],
                                "ymin": [bb[1] for bb in bboxes],
                                "xmax": [bb[2] for bb in bboxes],
                                "ymax": [bb[3] for bb in bboxes],
                                "numgranules": numpy.zeros((len(bboxes),), dtype=int),
                                "is_downloaded": numpy.zeros((len(bboxes),), dtype=bool)})

    if verbose:
        print(len(bboxes), "bounding boxes total.")
    #     print("Building", os.path.split(fname)[1], "and querying to see which squares are already downloaded.")

    # num_downloaded = 0
    # for i,row in df.iterrows():
    #     bbox = row['bbox']
    #     granules_list = nsidc_download.download_granules(short_name=["ATL03", "ATL08"],
    #                                                    region=bbox,
    #                                                    local_dir = ICESAT2_DIR,
    #                                                    dates = ['2021-01-01',',2021-12-31'],
    #                                                    version='005',
    #                                                    fname_filter = None,
    #                                                    force = False,
    #                                                    query_only = True,
    #                                                    fname_python_regex = "\.h5\Z",
    #                                                    download_only_matching_granules = True,
    #                                                    quiet = True)

    #     row["numgranules"] = len(granules_list)

    #     all_granules_are_downloaded = True
    #     for granule in granules_list:
    #         if not os.path.exists(granule):
    #             all_granules_are_downloaded = False
    #             break

    #     if all_granules_are_downloaded:
    #         row["is_downloaded"] = True
    #         num_downloaded += 1

    #     if verbose:
    #         utils.progress_bar.ProgressBar(i+1, len(bboxes), suffix="{0}/{1}".format(i+1, len(bboxes)))

    df.to_csv(fname)
    if verbose:
        print(fname, "written.")
        # print(num_downloaded, "of", len(bboxes), "are fully downloaded.", len(bboxes) - num_downloaded, "remain.")

    return


def subset_existing_photon_databases_to_ground_and_canopy_only():
    """The original version of these database has all the photons included.
    Read through them all, eliminate all that aren't ground (1), canopy (2), or canopy_top(3).
    This should help reduce file sizes some."""

    # NOTE: SHould only need to run this function once! Do not bother running again!
    # I've since updated the classify_icesat2_photons.save_granule_ground_photons() to only save ground and canopy photons rather than all of them.
    # Anything saved later than the "datetime_cutoff" should have this filtering already included and not need to be subset.
    print("Surveying ICESat-2 directory. Hold on a sec...")
    db_files = [os.path.join(ICESAT2_DIR, fn) for fn in os.listdir(ICESAT2_DIR) if (re.search("_photons\.h5\Z", fn) != None)]

    # Any files that have been last written past this date likely do not need to be reduced.
    # I'd started subsetting the files before saving after this.
    datetime_cutoff = datetime.datetime(2022,5,21,12,0,0)

    print(len(db_files), "photon database files found.")
    for i,db in enumerate(db_files):
        try:
            # Check to see if it's older than when we made the changes to how new databases were subset.
            # If it's a newer file, it would alreaady be subset, so skip this step.
            if datetime.datetime.fromtimestamp(os.path.getmtime(db)) <= datetime_cutoff:
                print("\r" + " "*120 + "\r", end="")
                print("{0}/{1}".format(i+1, len(db_files)), end=" ")
                # print(os.path.split(db)[1], "is newer than cutoff date..")
                print(os.path.split(db)[1], "is older than cutoff date and has already been done..")
                # TODO: Run this again on all the *newer* files than this cutoff date.
                # So far I'm seeing basically none of them being "not reduced", that tells
                # me that I'm probably missing some after this cutoff date.
                # It'll be a slow process but I can do it and it'll cut down at least a bit of space.
                # NOTE: I'd fixed this on Friday (I think?), so maybe just use Saturday or Sunday (5.21-22-ish) as a new newest-cutoff-date?

            else:
                fsize_orig = os.path.getsize(db)
                df = pandas.read_hdf(db, mode='r')

                # Get only the photons whose "class_code" is 1 to 3
                subset_mask = df.class_code.between(1,3,inclusive="both")
                if numpy.all(subset_mask):
                    print("\r" + " "*120 + "\r", end="")
                    print("{0}/{1}".format(i+1, len(db_files)), end=" ")
                    print(os.path.split(db)[1], "not reduced.")
                else:
                    df_subset = df.loc[subset_mask]
                    df_subset.to_hdf(db, "icesat2", complib="zlib", complevel=3, mode='w')
                    fsize_new = os.path.getsize(db)
                    reduction_pct = (fsize_orig - fsize_new)/fsize_orig * 100
                    print("\r" + " "*120 + "\r", end="")
                    print("{0}/{1}".format(i+1, len(db_files)), end=" ")
                    print(os.path.split(db)[1],
                          utils.sizeof_format.sizeof_fmt(fsize_orig), "->",
                          utils.sizeof_format.sizeof_fmt(fsize_new) + ",",
                          "{0:.1f}% reduction".format(reduction_pct))
        except (AttributeError, tables.exceptions.HDF5ExtError):
            # If we can't find the module in the file that we're looking for, or
            # if an HDF5ExtError popped up, it
            # was probably incorrectly or incompletely saved. Delete it.
            os.remove(db)
            print("\r" + " "*120 + "\r", end="")
            print("{0}/{1}".format(i+1, len(db_files)), end=" ")
            print(os.path.split(db)[1], "has an error. Deleting.")

        utils.progress_bar.ProgressBar(i+1, len(db_files), suffix = "{0}/{1}".format(i+1, len(db_files)))


def delete_granules_where_photon_databases_exist():
    """Get rid of granules in which we've already created a _photon.h5 database.

    We don't need the original granules anymore.
    """
    print("Surveying ICESat-2 directory. Hold on a sec...")
    db_files = [os.path.join(ICESAT2_DIR, fn) for fn in os.listdir(ICESAT2_DIR) if (re.search("_photons\.h5\Z", fn) != None)]

    files_deleted = 0

    print(len(db_files), "photon database files found.")
    for i,db in enumerate(db_files):
        atl03_fname = db.replace("_photons.h5", ".h5")
        atl08_fname = atl03_fname.replace("ATL03", "ATL08", 1)
        if os.path.exists(atl03_fname):
            os.remove(atl03_fname)
            files_deleted += 1
        if os.path.exists(atl08_fname):
            os.remove(atl08_fname)
            files_deleted += 1
        utils.progress_bar.ProgressBar(i+1, len(db_files), suffix = "{0}/{1}".format(i+1, len(db_files)))

    print(files_deleted, "files deleted.")

def download_all(overwrite = False,
                 track_progress = True,
                 verbose=True):
    """Square by square, go through and download the whole fuckin' world of ICESat-2 photons."""
    tile_tuples = etopo.generate_empty_grids.create_list_of_tile_tuples(resolution=1, verbose=False)

    # Get rid of bottom and top latitudes here, since icesat-2 has a 1.2* pole-hole.
    tile_array = numpy.array([tt for tt in tile_tuples if tt[0] not in (-90,89)], dtype=numpy.dtype([('y',numpy.int16), ('x',numpy.int16)]))
    # put in order from zero-latitude northward & southward... do the poles last.
    tile_array_to_argsort = tile_array.copy()
    tile_array_to_argsort['y'] = numpy.abs(tile_array_to_argsort['y'])
    sort_mask = numpy.argsort(tile_array_to_argsort, order=('y','x'))
    tile_array_sorted = tile_array[sort_mask]
    # print(sort_mask, sort_mask.shape)
    # print(tile_array[sort_mask], tile_array[sort_mask].shape, tile_array[sort_mask].dtype)

    # for (y,x) in tile_array[sort_mask]:
        # print((y,x), end=" ")
    # foobar

    bboxes = [(int(x),int(y),int(x+1),int(y+1)) for (y,x) in tile_array_sorted]

    # Use the download_progress.csv file to track which tiles have been downloaded and which have not yet.
    if track_progress:
        progress_fname = my_config.icesat2_download_progress_csv
        progress_df = pandas.read_csv(progress_fname)
        if verbose:
            print(progress_fname, "read.")

    for i,bbox in enumerate(bboxes):
        xmin_letter = "W" if (bbox[0] < 0) else "E"
        xmax_letter = "W" if (bbox[2] < 0) else "E"
        ymin_letter = "S" if (bbox[1] < 0) else "N"
        ymax_letter = "S" if (bbox[3] < 0) else "N"
        print("\n=== {0}/{1} === BBOX: {2}{3}-{4}{5}, {6}{7}-{8}{9}".format(
                     i+1,
                     len(bboxes),
                     abs(bbox[0]),
                     "" if (xmin_letter == xmax_letter) else xmin_letter,
                     abs(bbox[2]),
                     xmax_letter,
                     abs(bbox[1]),
                     "" if (ymin_letter == ymax_letter) else ymin_letter,
                     abs(bbox[3]),
                     ymax_letter))
        if track_progress:
            # Update the row and move along
            row = progress_df.loc[(progress_df.xmin == bbox[0]) & \
                                  (progress_df.ymin == bbox[1]) & \
                                  (progress_df.xmax == bbox[2]) & \
                                  (progress_df.ymax == bbox[3])]
            assert len(row) == 1
            if row.iloc[0]['is_downloaded']:
                if verbose:
                    print("Already downloaded. Moving on.")
                continue

        granules_list = nsidc_download.download_granules(short_name=["ATL03", "ATL08"],
                                                         region=bbox,
                                                         local_dir = ICESAT2_DIR,
                                                         dates = ['2021-01-01',',2021-12-31'],
                                                         version='005',
                                                         fname_filter = None,
                                                         force = overwrite,
                                                         query_only = False,
                                                         fname_python_regex = "\.h5\Z",
                                                         download_only_matching_granules = True,
                                                         skip_granules_if_photon_db_exists = True,
                                                         quiet = not verbose)

        if track_progress:
            # Update the data record
            progress_df.loc[row.index,'is_downloaded'] = True
            progress_df.loc[row.index,'numgranules'] = len(granules_list)
            # Re-write the file out to disk. This helps if we cut this process short.
            progress_df.to_csv(progress_fname)
            if verbose:
                print(os.path.split(progress_fname)[1], "re-written.")

def generate_photon_databases_from_existing_granules(overwrite = False,
                                                     parallelize = True,
                                                     numprocs = 20,
                                                     verbose = True):
    """For all ATL08+ATL03 granule pairs that exist, generate a photon database.

    Both files must exist. Files can be downloaded from the download_all() function
    or using the nsidc_download.py module to get missing granules.

    If overwrite, generate a databaase even if it already exists. Otherwise, don't."""

    if verbose:
        print("Querying ICESat-2 directory for matching ATL03 & ATL08 granules that do not yet have photon databases.")
    granules_list = [fname for fname in os.listdir(ICESAT2_DIR) if ((fname[0:4] == "ATL0") and (os.path.splitext(fname)[1].lower() == ".h5") and (fname.find("_photons") == -1))]
    ATL03_granules = [gn for gn in granules_list if gn[0:5] == "ATL03"]
    ATL08_granules = [gn for gn in granules_list if gn[0:5] == "ATL08"]
    assert (len(ATL03_granules) + len(ATL08_granules)) == len(granules_list)

    ATL03_matching_granules = sorted([os.path.join(ICESAT2_DIR, atl03) for atl03 in ATL03_granules if atl03.replace("ATL03", "ATL08") in ATL08_granules])
    ATL03_database_names = [atl03.replace(".h5", "_photons.h5") for atl03 in ATL03_matching_granules]

    # If the _photons.h5 alreaady exists and we aren't overwriting, skip & move along.
    new_granules_to_create_pairs = [(granule, database) for (granule, database) in zip(ATL03_matching_granules, ATL03_database_names) if (overwrite or not os.path.exists(database))]

    if len(new_granules_to_create_pairs) == 0:
        if verbose:
            print("Nothing to do. Exiting.")
        return

    elif verbose:
        print("Creating", len(new_granules_to_create_pairs), "new granule photon databases.")

    if parallelize:
        procs_list = [None] * len(new_granules_to_create_pairs)
        N = len(procs_list)
        # First, generate a list of multiprocessing.Process objects to peruse through when doing this.
        for i,(granule, h5name) in enumerate(new_granules_to_create_pairs):
            procs_list[i] = multiprocessing.Process(target=classify_icesat2_photons.save_granule_ground_photons,
                                                    args=(granule,),
                                                    kwargs = {"output_h5": h5name,
                                                              "overwrite": overwrite,
                                                              "verbose"  : False}
                                                    )
        # Then, start executing those processes until they're all done.
        running_procs = []
        running_databases = []
        finished_procs = 0
        utils.progress_bar.ProgressBar(0, N, suffix="0/{0}".format(N))

        try:
            # Keep looping while there are still processes running or processes to yet finish.
            while (len(procs_list) > 0) or (len(running_procs) > 0):
                # Sanity checks. These should always be true.
                assert len(running_procs) == len(running_databases)
                assert len(procs_list) == len(new_granules_to_create_pairs)

                # First, loop through the list of running procs and see what's finished up.
                for running_p, running_db in zip(running_procs, running_databases):
                    if not running_p.is_alive():
                        # If the process is finished. Join it, roll it up and toss it.
                        running_p.join()
                        running_p.close()
                        # Get rid of the proc, the db
                        running_procs.remove(running_p)
                        running_databases.remove(running_db)
                        finished_procs += 1
                        # Print the status update, and update the progress bar.
                        if verbose:
                            print("\r" + " "*120, end="")
                            print("\r{0}/{1} {2} written.".format(finished_procs, N, os.path.split(running_db)[1]))
                            utils.progress_bar.ProgressBar(finished_procs, N, suffix="{0}/{1}".format(finished_procs, N))

                # Now loop through the unfinished procs and start them.
                while (len(running_procs) < numprocs) and (len(procs_list) > 0):
                    proc = procs_list.pop(0)
                    (_, dbname) = new_granules_to_create_pairs.pop(0)
                    proc.start()
                    running_procs.append(proc)
                    running_databases.append(dbname)

                time.sleep(0.05)

        except KeyboardInterrupt:
            # Kill the active processes and delete any databases that were currently being written to.
            for running_p, running_db in zip(running_procs, running_databases):
                running_p.kill()
                running_p.close()
                if os.path.exists(running_db):
                    if verbose:
                        print("Deleting incomplete file", os.path.split(running_db)[1])
                    os.remove(running_db)

    else:
        try:
            for i, (granule, h5name) in enumerate(new_granules_to_create_pairs):
                if verbose:
                    print("{0}/{1}".format(i+1, len(new_granules_to_create_pairs)), end=" ")
                classify_icesat2_photons.save_granule_ground_photons(granule,
                                                                     output_h5 = h5name,
                                                                     overwrite = overwrite,
                                                                     verbose = verbose)
        except KeyboardInterrupt:
            os.remove(h5name)

    return

def generate_all_photon_tiles(overwrite = False,
                              verbose = True):
    """Run through creating all the icesat2_photon_database tiles.

    I could try to parallelize this, but each process would be writing to the same
    geodataframe database, and I don't feel like having to install conflict-handling code around that.
    So I won't unless it's necessary.
    """
    # Get all the icesat-2 photon database tiles. Use icesat2_photon_database object for this.
    # Assign a process to each tile generation, and then loop through running them.
    is2db = icesat2_photon_database.ICESat2_Database()
    gdf = is2db.get_gdf(verbose=verbose)
    for tilerow in gdf.iterrows():
        pass
        # TODO: Finish this.

def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Utility for downloading and pre-processing the whole fucking world of ICESat-2 photon data. Right now set to do the whole year 2021 (Jan 1-Dec 30).")

    parser.add_argument("-numprocs", "-np", default=0, type=int, help="The number of parallel processes to run. Default run sequentially on 1 process.")
    parser.add_argument("--generate_photon_databases", "-g", action="store_true", default=False, help="For all pairs of granules that have both ATL03 and ALT08 downloaded, generate a photon database for that granule.")
    parser.add_argument("--move_icesat2_files", "-m", action="store_true", default=False, help="Move all the icesat-2 granuels and photon_databases from the old .cudem_cache directory into the data/icesat2/granules directory.")
    parser.add_argument("--delete_redundant_granules", "-d", action="store_true", default=False, help="Delete ICESat-2 granules where a _photon.h5 database already exists. Free up some disk space.")
    parser.add_argument("--generate_photon_tiles", "-t", action="store_true", default=False, help="Generate all the 0.25-degree photon tiles using the ICESat2 photon database code.")
    parser.add_argument("--overwrite", '-o', action="store_true", default=False, help="Overwrite existing files.")
    parser.add_argument("--quiet", '-q', action="store_true", default=False, help="Run in quiet mode.")

    return parser.parse_args()

if __name__ == "__main__":
    # Temporarily uncomment to create the icesat-2 download progress csv.
    # create_icesat2_progress_csv()
    # delete_granules_where_photon_databases_exist()
    # print("Running with subset_existing_photon_databases_to_ground_and_canopy_only() enabled.")

    # subset_existing_photon_databases_to_ground_and_canopy_only()
    # import sys
    # sys.exit(0)

    args = define_and_parse_args()

    if args.move_icesat2_files:
        move_all_granules_to_icesat2_granules_dir()

        if args.delete_redundant_granules:
            delete_granules_where_photon_databases_exist()

    elif args.generate_photon_tiles:
        generate_all_photon_tiles(overwrite = args.overwrite,
                                  verbose = not args.quiet)

    elif args.generate_photon_databases:
        generate_photon_databases_from_existing_granules(parallelize = not (args.numprocs == 0),
                                                         numprocs = args.numprocs,
                                                         overwrite = args.overwrite,
                                                         verbose = not args.quiet)

        if args.delete_redundant_granules:
            delete_granules_where_photon_databases_exist()

    elif args.delete_redundant_granules:
        delete_granules_where_photon_databases_exist()

    else:
        download_all(overwrite = args.overwrite,
                     verbose = not args.quiet)
