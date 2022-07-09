# -*- coding: utf-8 -*-

"""find_bad_icesat2_granules.py -- code for identifying and eliminating bad ICESat-2 granules from analyses."""

import os
import pandas
import numpy
import matplotlib.pyplot as plt
# import scipy.stats

####################################3
# Include the base /src/ directory of thie project, to add all the other modules.
import import_parent_dir; import_parent_dir.import_src_dir_via_pythonpath()
####################################3
import icesat2.validate_dem as validate_dem
import icesat2.atl_granules as atl_granules
import utils.configfile
import utils.traverse_directory
import datasets.etopo_source_dataset

my_config = utils.configfile.config()

# def create_bad_granules_master_csv(bad_granule_list_fname = my_config._abspath(my_config.icesat2_bad_granules_list),
#                                     dataset_name = "CopernicusDEM",
#                                     overwrite=False,
#                                     verbose=True):
#     """Create a CSV file with a record of the bad granules we've identified from validating that dataset.

#     We assume that """




# def add_file_to_bad_granule_list(granule_fname,
#                                  bad_granule_list_fname = my_config._abspath(my_config.icesat2_bad_granules_list)):
#     """If we've found a bad granule in the list, add it to the list of bad granules.
#     In this case we're using the ATL03..._photons.h5 files."""

# def identify_bad_granules_in_single_validation(dem_fname,
#                                                verbose=True):
#     """Run a single-validation on a DEM, on a photon by photon basis.
#     Identify granules where the photon errors are consistently outside the ranges
#     of others by more than 2 s.d. (3 s.d.?) away from the mean range of the overall photons."""

# def get_summary_h5_fname_from_tilename(dem_fname):
#     """Given a DEM fname, return the file path of validation results file in the granule_validation_results directory."""
#     fname_short = os.path.split(dem_fname)
#     fname_base, ext = os.path.splitext(fname_short)
#     return os.path.join(my_config.icesat2_granule_validation_results_directory,
#                         fname_base + "_granule_results.h5")

# def plot_granule_histograms(granule_ids,
#                             granule_data_series,
#                             hist_fname = None,
#                             bad_granule_ids_to_label = None):
#     """Plot a stacked histogram showing the bad granules."""

def collect_granule_level_data(photon_results_df,
                               lo_pct_cutoff = 0.05,
                               hi_pct_cutoff = 99.95):
    """Given a photon-level validation results file, return granule-level comparisons of data.
    Also compute the histogram if it's asked for, and plot it.
    """
    if type(photon_results_df) == str:
        df = pandas.read_hdf(photon_results_df, mode='r')
    elif type(photon_results_df) == pandas.DataFrame:
        df = photon_results_df
    else:
        raise TypeError("Unknown type of 'photon_results_dataframe' in granule_hist():", type(photon_results_df))

    # Get some cutoff boundaries in order to eliminate distant outliers from the histogram plots.
    low, hi = numpy.percentile(df.dem_minus_is2_m, (lo_pct_cutoff, hi_pct_cutoff))
    gids = numpy.unique(list(zip(df.granule_id1, df.granule_id2)), axis=0)
    piles = []
    for id1, id2 in gids:
        piles.append(df[(df.granule_id1 == id1) & (df.granule_id2 == id2) & \
                        (df.dem_minus_is2_m >= low) & (df.dem_minus_is2_m <= hi)].dem_minus_is2_m)
    return gids, piles

def plot_stacked_histograms_with_bad_granule_marked(dem_fname,
                                                    bad_gids_list,
                                                    photon_df,
                                                    nbins=200,
                                                    verbose=True):

    granule_hist_plotname = get_photon_histogram_plotname_from_dem_fname(dem_fname)
    gids, piles = collect_granule_level_data(photon_df)

    # - 'gids' is a Nx2 list of granule-id pairs. All the other data corresponds with this list.
    # - 'piles' is a list of pandas.Series objects with the errors from each dataframe.
    # - 'h' is the stacked histogram data returned by plt.hist, consisting of:
        # - an N-tuple of all the histogram totals for each bin of the histogram.
        #     'N' is the number of different granule-id pairs. Each list is 'nbins' long.
        # - a array of the histogram box boundaries. This array is nbins+1 long.
        # - a list of N BarContainer objects, from matplotlib.pyplot.hist (probably not useful but we'll include in case it is).

    # Generate the axes and histogram.
    fig, ax = plt.subplots(dpi=600)
    gid_totals, bin_boundaries, _ = ax.hist(piles, nbins, histtype='bar', stacked=True)

    # for i, (p, gt) in enumerate(zip(piles, gid_totals)):
    #     print("====================", i, "++++++++++++++++++")
    #     print(numpy.mean(p), numpy.std(p))
    #     print(numpy.sum(gt), "of", len(photon_df))
    #     print(numpy.argmax(gt), bin_boundaries[numpy.argmax(gt)])
    #     print(gt)


    # foobar
    # print(len(gids), len(piles), len(gid_totals), len(bin_boundaries))

    # For each bad granule, put its name on the plot.
    for gid1, gid2 in bad_gids_list:
        # gid1, gid2 = atl_granules.granule_id_to_intx2(gid)
        gname = atl_granules.intx2_to_granule_id((gid1, gid2))

        # Find that bad granule in the list of all the granules in this file.
        for i, (tgid1, tgid2) in enumerate(gids):
            if (tgid1 == gid1) and (tgid2 == gid2):
                break

        gid_data = piles[i]
        gid_mean = numpy.mean(gid_data)
        gid_std = numpy.std(gid_data)
        # Since these are "stacked" histograms, I gotta subtract the histogram before
        hist_totals = gid_totals[i] - (0 if i==0 else gid_totals[i-1])
        max_j = numpy.argmax(hist_totals)
        max_y = hist_totals[max_j]
        # max_x = numpy.mean(bin_boundaries[max_j:max_j+2])

        # Plot the GID name hovering over the histogram.
        text_y = max_y * 1.15
        x_buffer = (bin_boundaries[-1] - bin_boundaries[0])*0.02
        if max_j < (len(hist_totals)/2):
            halign = "left"
            text_x = bin_boundaries[0] + x_buffer
        else:
            halign = "right"
            text_x = bin_boundaries[-1] - x_buffer

        if os.path.splitext(gname)[1] == "":
            gname = gname + ".h5"
        # Put the text of the granule name on there.
        ax.text(text_x, text_y, gname, ha=halign, va="bottom", size="small")

        # Add mean & 2-stdev bars
        # Center vertical line
        ax.plot([gid_mean, gid_mean], [0, max_y*0.9], lw=1, color="black")
        # Horizontal line
        ax.plot([gid_mean-2*gid_std, gid_mean+2*gid_std], [max_y*0.9/2, max_y*0.9/2], lw=0.5, linestyle="--", color="black")
        # Left vertical line
        ax.plot([gid_mean-2*gid_std, gid_mean-2*gid_std], [max_y*0.20,max_y*0.70], lw=0.7, color="black")
        # Right vertical line
        ax.plot([gid_mean+2*gid_std, gid_mean+2*gid_std], [max_y*0.20,max_y*0.70], lw=0.7, color="black")

    ax.set_xlabel("Error (m)")
    ax.set_ylabel("Count")
    fig.suptitle(os.path.split(dem_fname)[1])

    fig.tight_layout()
    fig.savefig(granule_hist_plotname)
    if verbose:
        print(os.path.join(*(os.path.normpath(granule_hist_plotname).split(os.sep)[-3:])), "written.")

    return

def find_bad_granules_in_a_dataset(dataset_name_or_object,
                                   # csv_list_fname,
                                   make_histogram_plots_if_bad = True,
                                   verbose = True):
    """Loop through all the validation results of a dataset that has been validated.
    Look for all photon-level validation results produced previously and tag all
    the bad granules found with the '_BAD_GRANULES.csv' file.

    Return a dataframe of all the bad granules found, along with the number of DEMs in which that bad granule was found.
    Save the dataframe to "csv_list_output."""
    if type(dataset_name_or_object) == str:
        dset = datasets.etopo_source_dataset.get_source_dataset_object(dataset_name_or_object)

    elif type(dataset_name_or_object) == datasets.etopo_source_dataset.ETOPO_source_dataset:
        dset = dataset_name_or_object

    else:
        raise TypeError("Unhandled object type given for parameter 'dataset_name_or_object':", type(dataset_name_or_object))

    list_of_datafiles = dset.retrieve_all_datafiles_list()

    list_of_bad_granule_ids = []
    # list_of_filenames_containing_bad_granules = []
    for dem_fname in list_of_datafiles:
        # photon_h5_fname = get_photon_validation_fname_from_dem_fname(dem_fname)
        bad_granules_csv_name = get_bad_granule_csv_name_from_dem_fname(dem_fname)

        # # If we've already created a "BAD_GRANULES.csv" file from this DEM, skip it.
        # if os.path.exists(bad_granules_csv_name):
        #     continue

        # Step through, find the granules that are separated from one another.
        bad_granule_ids, photon_df = find_granules_that_are_separate_from_others(dem_fname,
                                                                                 also_return_photon_df = True,
                                                                                 skip_if_no_photon_results_file = True,
                                                                                 write_bad_granules_csv_if_needed = True,
                                                                                 verbose = verbose)

        if len(bad_granule_ids) > 0:
            assert os.path.exists(bad_granules_csv_name)
            list_of_bad_granule_ids.extend(bad_granule_ids)

            if make_histogram_plots_if_bad:
                plot_stacked_histograms_with_bad_granule_marked(dem_fname, bad_granule_ids, photon_df, verbose=verbose)

    return list(set(list_of_bad_granule_ids))


def get_photon_validation_fname_from_dem_fname(dem_fname,
                                               results_subdir='icesat2_results',
                                               include_results_df_name = False):
    base_path, fname = os.path.split(dem_fname)
    # If the results directory doesn't exist yet, create it.
    if results_subdir is not None:
        results_dir = os.path.join(base_path, results_subdir)
    else:
        results_dir = base_path

    fbase, ext = os.path.splitext(fname)
    results_database_name = os.path.join(results_dir, fbase + "_results.h5")

    base, ext = os.path.splitext(results_database_name)
    photon_results_database_name = base + "_photon_level_results.h5"
    if include_results_df_name:
        return photon_results_database_name, results_database_name
    else:
        return photon_results_database_name

def get_bad_granule_csv_name_from_dem_fname(dem_fname,
                                            results_subdir='icesat2_results'):
    base_path, fname = os.path.split(dem_fname)
    # If the results directory doesn't exist yet, create it.
    if results_subdir is not None:
        results_dir = os.path.join(base_path, results_subdir)
    else:
        results_dir = base_path

    fbase, ext = os.path.splitext(fname)
    bad_granule_csv_name = os.path.join(results_dir, fbase + "_BAD_GRANULES.csv")
    return bad_granule_csv_name

def get_granule_stats_h5_name_from_dem_fname(dem_fname,
                                             results_subdir = 'icesat2_results'):
    base_path, fname = os.path.split(dem_fname)
    # If the results directory doesn't exist yet, create it.
    if results_subdir is not None:
        results_dir = os.path.join(base_path, results_subdir)
    else:
        results_dir = base_path

    fbase, ext = os.path.splitext(fname)
    bad_granule_csv_name = os.path.join(results_dir, fbase + "_granule_stats.h5")
    return bad_granule_csv_name

def get_photon_histogram_plotname_from_dem_fname(dem_fname,
                                                 results_subdir = "icesat2_results"):
    base_path, fname = os.path.split(dem_fname)
    # If the results directory doesn't exist yet, create it.
    if results_subdir is not None:
        results_dir = os.path.join(base_path, results_subdir)
    else:
        results_dir = base_path

    fbase, ext = os.path.splitext(fname)
    bad_granule_png_name = os.path.join(results_dir, fbase + "_bad_granule_histogram.png")
    return bad_granule_png_name


def generate_photon_database_validation(dem_fname,
                                        dem_vdatum,
                                        vdatum_out = "EGM2008",
                                        overwrite = False, # Probably need to add variables for cutoffs here.
                                        verbose = True):
    """Run a validation on the fname and tally up granule-level photon-by-photon
    validation results for each granule within the DEM bounding box.

    Return a dataframe with the results for that tile, delineated by granules within that tile.
    Also, write the dataframe out to an HDF5 to the results dataframe.
    """
    # 1. Get the photon validation dataframe for that tile *on a photon basis*. *Not* aggregated by grid-cell.
    results_subdir = "icesat2_results"
    base_path, fname = os.path.split(dem_fname)
    # If the results directory doesn't exist yet, create it.
    results_dir = os.path.join(base_path, results_subdir)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    photon_results_database_name, results_database_name = get_photon_validation_fname_from_dem_fname(dem_fname, include_results_df_name=True)

    # Check to make sure it's not already in there (skip if it is, unless we've specified to overwrite it.)
    if os.path.exists(photon_results_database_name):
        if overwrite:
            if verbose:
                print("Removing old", os.path.split(photon_results_database_name)[1] + ".")
            os.remove(photon_results_database_name)
        else:
            if verbose:
                print(os.path.split(photon_results_database_name)[1], "already exists.")
            return

    # Check to make sure it's not already in there (skip if it is, unless we've specified to overwrite it.)
    # if os.path.exists(results_database_name):
    #     if overwrite:
    #         if verbose:
    #             print("Removing old", os.path.split(results_database_name)[1] + ".")
    #         os.remove(results_database_name)
    #     else:
    #         if verbose:
    #             print(os.path.split(results_database_name)[1], "already exists.")
    #         return

    else: # not os.path.exists(photon_results_database_name)
        # If we don't have a photon-level validations file, make sure we generate one
        # Key point: ensure "include_photon_level_validation = True".
        validate_dem.validate_dem_parallel(dem_fname,
                                           photon_dataframe_name = None,
                                           use_icesat2_photon_database = True,
                                           icesat2_photon_database_obj = None,
                                           dem_vertical_datum = dem_vdatum,
                                           output_vertical_datum = vdatum_out,
                                           results_dataframe_file = results_database_name,
                                           interim_data_dir = results_dir,
                                           include_photon_level_validation=True,
                                           quiet=not verbose,
                                           )

    return photon_results_database_name

def find_granules_that_are_separate_from_others(dem_fname,
                                                also_return_photon_df = False,
                                                skip_if_no_photon_results_file = False,
                                                dem_vdatum = None,
                                                save_granule_stats_df=False,
                                                remove_granules_with_less_than_N_photons=20,
                                                write_bad_granules_csv_if_needed=True,
                                                verbose=True):
    """For a given fname, perform a validation (if not done already) and compute the photon-level
    validation stats for the DEM. Then, compute a granule-by-granule 2-sided KS stat to
    determine whether these granules come from the same "population". Save the results to a new dataframe."""

    photon_results_database_name = get_photon_validation_fname_from_dem_fname(dem_fname)
    if not os.path.exists(photon_results_database_name):
        if skip_if_no_photon_results_file:
            if also_return_photon_df:
                return [], None
            else:
                return []

        if verbose:
            print("Generating", photon_results_database_name)
        # If the original validation hasn't been done yet, we'd better supply the
        # correct vdatum of the dem here.
        if dem_vdatum is None:
            raise ValueError("If ICESat-2 validation has not been performed yet, " + \
                             "dem_vdatum must be supplied to find_granules_separate_from_others(). " + \
                             "'None' supplied.")
        generate_photon_database_validation(dem_fname,
                                            dem_vdatum=dem_vdatum,
                                            verbose=verbose)

    # 2. Loop through all the unique granules and organize the photons by those.
    # Read the photon_level validations dataframe
    photon_df = pandas.read_hdf(photon_results_database_name, mode='r')
    # A Nx2 array of uniqpe granule-id integer pairs.
    # unique_granule_ids = numpy.unique(list(zip(photon_df.granule_id1, photon_df.granule_id2)))

    granule_ids, granule_data = collect_granule_level_data(photon_df)


    # Remove any granules that have less than the requisite number of photons in them in this database.
    granule_idxs_to_remove = []
    for i, g_data in enumerate(granule_data):
        if len(g_data) < remove_granules_with_less_than_N_photons:
            granule_idxs_to_remove.append(i)
    # Any of the granules that were identified in the last loop, remove them.
    # Go through the list backward to not screw up the indices if there is more than one.
    granule_idxs_to_remove.reverse()
    for i in granule_idxs_to_remove:
        granule_ids = numpy.concatenate((granule_ids[:i,:], granule_ids[i+1:,:]), axis=0)
        granule_data = granule_data[:i] + granule_data[i+1:]

    granule_stats_results_tuples = []

    for i,((gid1, gid2), g_data) in enumerate(zip(granule_ids, granule_data)):
        # In the next loop, only do the granules that come after the first granule ([i+1:]),
        # so that we skip the 2x redundancy of comparing (g1, g2) and later (g2, g1)
        # in reverse order, which will give the exact same answer.
        for (gid1_other, gid2_other), g_data_other in list(zip(granule_ids, granule_data))[i+1:]:
            # Skip if they are the same granule. No need to compare with oneself.
            # This logic shouldn't be needed if the above loopling logic works as intended.
            # But it's harmless to keep it here and would help if it comes to it,
            # So just skip over any granule self-comparisons that happen by accident.
            if gid1 == gid1_other and gid2 == gid2_other:
                continue

            # Apply a 2-sided, 2-sample Kolmogorov-Smirnov test to see if the two
            # sets of DEM errors from different granules come from the same approximate
            # distribution. Most of them are pretty close, perhaps slightly different shapes,
            # but some of they are *way* off, with highly-different means and distributions.
            # Find those. Test on some known samples first.
            # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html for documentation

            # Note: The KS stat isn't working great, because these different granules paths *are* different
            # statistical populations, so it thinks they all are different. Gives high KS scores
            # and low p-values for nearly literally every pair. It's not wrong, but it's too sensitive for
            # our needs. Instead, we just want to know if the 2-s.d. range of one granule is
            # outside the 2-s.d. range of *all* the other granules (perhaps except one?). If so, mark it
            # as a 'bad' granule. We will later check to see if this granule appears as 'bad'
            # in more than one DEM validation, in which case we can probably infer it doesn't have
            # good data.
            # ks_stat, p_value = scipy.stats.ks_2samp(g_data, g_data_other, alternative="two-sided", mode='auto')
            g1_mean = g_data.mean()
            g1_std = g_data.std()
            g1_size = len(g_data)
            g2_mean = g_data_other.mean()
            g2_std = g_data_other.std()
            g2_size = len(g_data_other)

            granule_stats_results_tuples.append( ( gid1,
                                                   gid2,
                                                   g1_size,
                                                   g1_mean,
                                                   g1_std,
                                                   gid1_other,
                                                   gid2_other,
                                                   g2_size,
                                                   g2_mean,
                                                   g2_std)
                                                   # ks_stat,
                                                   # p_value )
                                                )
    # Create a little dataframe from this.
    grdf = pandas.DataFrame(data={"g1_id1"  : [n[0] for n in granule_stats_results_tuples],
                                  "g1_id2"  : [n[1] for n in granule_stats_results_tuples],
                                  "g1_size" : [n[2] for n in granule_stats_results_tuples],
                                  "g1_mean" : [n[3] for n in granule_stats_results_tuples],
                                  "g1_std"  : [n[4] for n in granule_stats_results_tuples],
                                  "g2_id1"  : [n[5] for n in granule_stats_results_tuples],
                                  "g2_id2"  : [n[6] for n in granule_stats_results_tuples],
                                  "g2_size" : [n[7] for n in granule_stats_results_tuples],
                                  "g2_mean" : [n[8] for n in granule_stats_results_tuples],
                                  "g2_std"  : [n[9] for n in granule_stats_results_tuples],
                                  })
                                        # "ks_stat" : [n[10] for n in granule_ks_results_tuples],
                                        # "p_value" : [n[11] for n in granule_ks_results_tuples],

    # A simple test to see if the mean of each granule is outside +/- 2 SD of the mean of the other, for both granules.
    # ks_test_df["is_outside"] = ~(((ks_test_df.g1_mean - (2 * ks_test_df.g1_std)) < ks_test_df.g2_mean) & \
    #                              ((ks_test_df.g1_mean + (2 * ks_test_df.g1_std)) > ks_test_df.g2_mean) & \
    #                              ((ks_test_df.g2_mean - (2 * ks_test_df.g2_std)) < ks_test_df.g1_mean) & \
    #                              ((ks_test_df.g2_mean + (2 * ks_test_df.g2_std)) > ks_test_df.g1_mean))

    # Another simple test to see if the mean +/- 2-SD envelopes of both granules do not at all overlap.
    # (i.e if the "top" of one 2-SD envelop is less than the "bottom" of the other, in either direction.)
    grdf["is_outside"] = ((grdf.g1_mean - (2 * grdf.g1_std)) > (grdf.g2_mean + (2 * grdf.g2_std))) | \
                               ((grdf.g2_mean - (2 * grdf.g2_std)) > (grdf.g1_mean + (2 * grdf.g1_std)))

    # Comment this out later.
    # print(grdf)
    if save_granule_stats_df:
        granule_stats_fname = get_granule_stats_h5_name_from_dem_fname(dem_fname)
        grdf.to_hdf(granule_stats_fname, "icesat2", complevel=3, complib='zlib')
        if verbose:
            print(granule_stats_fname, "written.")

    # Now, identify which granules have been identified as "outside" the 2-s.d. envelopes of literally
    # all (but an allowance of 1) other granules in this dataset.
    LIST_OF_OUTSIDE_GRANULES_IDS = []
    LIST_OF_PHOTON_COUNTS = []
    for (gid1, gid2) in granule_ids:
        # Get all records that have this granule.
        grdf_this_granule = grdf[((grdf["g1_id1"] == gid1) & (grdf["g1_id2"] == gid2)) | \
                                 ((grdf["g2_id1"] == gid1) & (grdf["g2_id2"] == gid2))]
        # There should be 1 less record (all other granules) than the length of the whole list of granules.
        # Just check this logic here.
        assert (len(grdf_this_granule)+1) == granule_ids.shape[0]

        # If the number of granules this granule is "ouside of" is greater than or
        # equal to 1 less than all the other granules in this dataset, and is
        # more than at least 2 other granules, then add it to the list of outside granule IDs.
        if 3 <= grdf_this_granule["is_outside"].sum() >= (len(grdf_this_granule) - 1):
            LIST_OF_OUTSIDE_GRANULES_IDS.append((gid1, gid2))
            photon_count = ((photon_df.granule_id1 == gid1) & (photon_df.granule_id2 == gid2)).sum()
            LIST_OF_PHOTON_COUNTS.append(photon_count)

    if (len(LIST_OF_OUTSIDE_GRANULES_IDS) > 0) and write_bad_granules_csv_if_needed:
        csv_fname = get_bad_granule_csv_name_from_dem_fname(dem_fname)
        outside_granule_df = pandas.DataFrame(data={"granule_name": [atl_granules.intx2_to_granule_id(g)+".h5" for g in LIST_OF_OUTSIDE_GRANULES_IDS],
                                                    "gid1": [g[0] for g in LIST_OF_OUTSIDE_GRANULES_IDS],
                                                    "gid2": [g[1] for g in LIST_OF_OUTSIDE_GRANULES_IDS],
                                                    "photon_count": LIST_OF_PHOTON_COUNTS},
                                             )
        outside_granule_df.to_csv(csv_fname, index=False)
        if verbose:
            print(os.path.join(*(os.path.normpath(csv_fname).split(os.sep)[-3:])), "written with {0} entries.".format(len(LIST_OF_OUTSIDE_GRANULES_IDS)))

    if also_return_photon_df:
        return LIST_OF_OUTSIDE_GRANULES_IDS, photon_df
    else:
        return LIST_OF_OUTSIDE_GRANULES_IDS


def accumulate_bad_granule_dfs(dataset_name_or_obj,
                               bad_granule_regex = "_BAD_GRANULES\.csv\Z",
                               verbose = True):
    """Run through a whole dataset, find all the "_BAD_GRANULES.csv" filenames, accumulate them all into a collective dataframe.
    NOTE: Many may have repeat granule names from their respective DEMs. This is okay.
    Just return a dataframe that includes all the "_BAD_GRANULES.csv" entries, along with the filename they came from.
    Return this dataframe.
    """
    if isinstance(dataset_name_or_obj, datasets.etopo_source_dataset.ETOPO_source_dataset):
        dset_obj = dataset_name_or_obj
    else:
        dset_obj = datasets.etopo_source_dataset.get_source_dataset_object(dataset_name_or_obj)

    bad_granule_file_list = utils.traverse_directory.list_files(dset_obj.config._abspath(dset_obj.config.source_datafiles_directory),
                                                                regex_match=bad_granule_regex,
                                                                include_base_directory=True)

    if len(bad_granule_file_list) > 0:
        bad_granule_dfs = [pandas.read_csv(fn, index_col=False) for fn in bad_granule_file_list]

        # Add a column for csv_filename for where the file came from.
        for fn, df in zip(bad_granule_file_list, bad_granule_dfs):
            df['csv_filename'] = fn

        # Concatenate them into one dataframe
        return pandas.concat(bad_granule_dfs, ignore_index=True)
    else:
        return None

def create_master_list_of_bad_granules(dataset_name,
                                       master_bad_granule_csv = my_config._abspath(my_config.icesat2_bad_granules_csv),
                                       verbose=True):
    """Go through a dataset that has been validated with "photon-level results", and that has already had
    find_bad_granules_in_a_dataset() called on it.

    Pick out all the _BAD_GRANULES.csv files from it, add their records to the master list if they are not already in there.
    """
    bad_granules_df = accumulate_bad_granule_dfs(dataset_name, verbose=verbose)
    if bad_granules_df is None:
        if verbose:
            print(f"No bad granules found in dataset '{dataset_name}'")
        return

    bad_granules_df.to_csv(master_bad_granule_csv, index=False)
    if verbose:
        print(master_bad_granule_csv, "written.")
    return

def get_list_of_granules_to_reject(bad_granule_csv = my_config._abspath(my_config.icesat2_bad_granules_csv),
                                   refind_bad_granules = False,
                                   regenerate_bad_granule_csv = False,
                                   dataset_name_if_regenerating = "CUDEM_CONUS", # TODO: Change this to Copernicus when ready to do the whole world.
                                   files_identified_threshold = 2,
                                   min_photons_threshold = 1000,
                                   return_as_gid_numbers=False,
                                   verbose = True):
    """Reading the 'bad_granules_list.csv' file, return a list of granules that
    have either been identified in 'files_identified_threshold' or more separate DEM
    validations as "bad", or that have a 'min_photons_threshold' minimum number of photons
    in one or more files, and have been identified as 'bad'.

    If return_as_gid_numbers, return a list of 2-tuple (gid1,gid2) granule identifiers.
    Otherwise, return a list of ATL03...h5 granule names."""
    if refind_bad_granules:
        find_bad_granules_in_a_dataset(dataset_name_if_regenerating)

    if regenerate_bad_granule_csv:
        create_master_list_of_bad_granules(dataset_name_if_regenerating,
                                           master_bad_granule_csv = bad_granule_csv,
                                           verbose=verbose)

    if not os.path.exists(bad_granule_csv):
        if verbose:
            print("Error: No bad_granule_list.csv file found. Not filtering out bad granules.")
        return []

    bad_granule_df = pandas.read_csv(bad_granule_csv, index_col=False)
    unique_granule_names = bad_granule_df['granule_name'].unique()

    granules_to_exclude = []
    for uniq_gr in unique_granule_names:
        subset_df = bad_granule_df[bad_granule_df['granule_name'] == uniq_gr]
        num_files_identified = len(subset_df)
        num_photons = subset_df["photon_count"].sum()

        if (num_files_identified >= files_identified_threshold) or (num_photons >= min_photons_threshold):
            granules_to_exclude.append(uniq_gr)

    if return_as_gid_numbers:
        return [atl_granules.granule_id_to_intx2(gid) for gid in granules_to_exclude]
    else:
        return granules_to_exclude


if __name__ == "__main__":
    gr = get_list_of_granules_to_reject(refind_bad_granules = True, regenerate_bad_granule_csv = True)
    if len(gr) > 0:
        print("GRANULES TO EXCLUDE")
        for g in gr:
            print("\t", g)

    # find_bad_granules_in_a_dataset("CUDEM_CONUS")
    # gids = find_granules_that_are_separate_from_others(\
    #        "/home/mmacferrin/Research/DATA/DEMs/CUDEM/data/CONUS/NCEI_ninth_Topobathy_2014_8483/AL_nwFL/ncei19_n30X50_w085X25_2019v1.tif",
    #        dem_vdatum="navd88",
    #        verbose=True)
