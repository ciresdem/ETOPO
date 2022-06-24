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

my_config = utils.configfile.config()

def create_bad_granules_csv(bad_granule_list_fname = my_config._abspath(my_config.icesat2_bad_granules_list),
                            overwrite=False,
                            verbose=True):
    """Create an empty CSV file with a record of the bad granules we've identified."""


def add_file_to_bad_granule_list(granule_fname,
                                 bad_granule_list_fname = my_config._abspath(my_config.icesat2_bad_granules_list)):
    """If we've found a bad granule in the list, add it to the list of bad granules.
    In this case we're using the ATL03..._photons.h5 files."""

# def identify_bad_granules_in_single_validation(dem_fname,
#                                                add_to_list = True,
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

def plot_granule_histograms(granule_ids,
                            granule_data_series,
                            hist_fname = None,
                            bad_granule_ids_to_label = None):
    """Plot a stacked histogram showing the bad granules."""

def collect_granule_level_data(photon_results_df,
                               lo_pct_cutoff = 0.05,
                               hi_pct_cutoff = 99.95):
                               # nbins=100,
                               # omit_hist=False):
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
    # Right now, compute these with matplotlib pyplot. SHould be able to do the
    # same thing in numpy, but I haven't found an easy, elegant way of doing that yet.

    # Maybe I don't want the histograms, actually, just give me the fucking data.
    # if omit_hist:
    return gids, piles

def plot_stacked_histograms_with_bad_granule_marked(dem_fname, photon_df, nbins=200):


    # This is a big me
    # h = plt.hist(piles, nbins, histtype='bar', stacked=True)

    # Return the useful data we've generated here.
    # - 'gids' is a Nx2 list of granule-id pairs. All the other data corresponds with this list.
    # - 'piles' is a list of pandas.Series objects with the errors from each dataframe.
    # - 'h' is the stacked histogram data returned by plt.hist, consisting of:
        # - an N-tuple of all the histogram totals for each bin of the histogram.
        #     'N' is the number of different granule-id pairs. Each list is 'nbins' long.
        # - a array of the histogram box boundaries. This array is nbins+1 long.
        # - a list of N BarContainer objects, from matplotlib.pyplot.hist (probably not useful but we'll include in case it is).
    # - the current figure we just plotted to.
    # - the current axis we just plotted to.
    # return gids, piles, h, plt.gcf(), plt.gca()

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
    for (gid1, gid2) in granule_ids:
        # Get all records that have this granule.
        grdf_this_granule = grdf[((grdf["g1_id1"] == gid1) & (grdf["g1_id2"] == gid2)) | \
                                 ((grdf["g2_id1"] == gid1) & (grdf["g2_id2"] == gid2))]
        # There should be 1 less record (all other granules) than the length of the whole list of granules.
        # Just check this logic here.
        assert (len(grdf_this_granule)+1) == granule_ids.shape[0]

        # If the number of granules this granule is "ouside" of is greater than or
        # equal to 1 less than all the other granules in this dataset, and is
        # more than at least 2 other granules, then add it to the list of outside granule IDs.
        if 3 <= grdf_this_granule["is_outside"].sum() >= (len(grdf_this_granule) - 1):
            LIST_OF_OUTSIDE_GRANULES_IDS.append((gid1, gid2))

    if (len(LIST_OF_OUTSIDE_GRANULES_IDS) > 0) and write_bad_granules_csv_if_needed:
        csv_fname = get_bad_granule_csv_name_from_dem_fname(dem_fname)
        outside_granule_df = pandas.DataFrame(data={"granule_name": [atl_granules.intx2_to_granule_id(g)+".h5" for g in LIST_OF_OUTSIDE_GRANULES_IDS],
                                                    "gid1": [g[0] for g in LIST_OF_OUTSIDE_GRANULES_IDS],
                                                    "gid2": [g[1] for g in LIST_OF_OUTSIDE_GRANULES_IDS]})
        outside_granule_df.to_csv(csv_fname, index=False)
        if verbose:
            print(csv_fname, "written with {0} entries.".format(len(LIST_OF_OUTSIDE_GRANULES_IDS)))

    return LIST_OF_OUTSIDE_GRANULES_IDS

if __name__ == "__main__":
    gids = find_granules_that_are_separate_from_others(\
           "/home/mmacferrin/Research/DATA/DEMs/CUDEM/data/CONUS/NCEI_ninth_Topobathy_2014_8483/AL_nwFL/ncei19_n30X50_w085X25_2019v1.tif",
           dem_vdatum="navd88",
           verbose=True)
