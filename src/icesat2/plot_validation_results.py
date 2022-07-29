# -*- coding: utf-8 -*-

import numpy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import pandas
import os
import collections
import six
import math

####################################3
# Include the base /src/ directory of thie project, to add all the other modules.
import import_parent_dir; import_parent_dir.import_src_dir_via_pythonpath()
####################################3
import utils.configfile
my_config = utils.configfile.config()


def is_iterable(obj):
    """Tell whether an object is a non-string iterable. (list, tuple, etc)."""
    return ( isinstance(obj, collections.Iterable) \
            and not isinstance(obj, six.string_types))

def get_data_from_h5_or_list(h5_name_or_list,
                             empty_val = my_config.etopo_ndv,
                             verbose=True):
    """Return the data either from a single hdf5 results file, or a list of them. Filter out empty (bad data) values."""
    if type(h5_name_or_list) == str:
        data = pandas.read_hdf(h5_name_or_list)
    elif is_iterable(h5_name_or_list):
        data_list = []
        for h5_file in h5_name_or_list:
            if os.path.exists(h5_file):
                data_list.append(pandas.read_hdf(h5_file))
        data = pandas.concat(data_list)
    else:
        raise TypeError("Non-iterable value for parameter 'results_h5_name_or_list':", h5_name_or_list)
    # print(data)

    meddiff         = data['diff_median']
    cellstd         = data['stddev'].astype(float)
    meandiff        = data['diff_mean']
    # numphotons      = data['numphotons']
    numphotons_intd = data['numphotons_intd']
    canopy_fraction = data["canopy_fraction"]
    # Plot some histograms

    # Get rid of data with 3 or less ground photons in the inter-decile range, and all nans.
    # If empty_val means something, return it.
    good_data_mask = (numphotons_intd > 3) & ~numpy.isnan(meandiff) & ~numpy.isnan(meddiff) & ~numpy.isnan(canopy_fraction) \
                     & (meandiff != empty_val) & (meddiff != empty_val) & (canopy_fraction != empty_val) & (cellstd != empty_val)

    data_subset = data[good_data_mask].copy()
    if (len(data_subset) == 0) and verbose:
        print("No reliable results contained in list of results h5 files.")
    return data_subset

def plot_histogram_and_error_stats_4_panels(results_h5_or_list_or_df,
                                            output_figure_name,
                                            empty_val = my_config.etopo_ndv,
                                            place_name=None,
                                            figsize=None,
                                            labels_uppercase = True,
                                            dpi = 600,
                                            verbose=True):
    """Generate a 4-panel figure of error stats.
    1) Histograms of mean errors
    2) 1:1 line of DEM vs ICESat-2 elevations.
    3) Histogram of canopy cover
    4) Histogram of # of photons per grid cell.

    If 'place_name' is provided, use it in the title of the plot.
    """
    if type(results_h5_or_list_or_df) == pandas.DataFrame:
        data = results_h5_or_list_or_df
    else:
        data = get_data_from_h5_or_list(results_h5_or_list_or_df,
                                        empty_val = empty_val)

    # meddiff         = data['diff_median']
    # cellstd         = data['stddev'].astype(float)
    meandiff        = data['diff_mean']
    # numphotons      = data['numphotons']
    numphotons_intd = data['numphotons_intd']
    canopy_fraction = data["canopy_fraction"]
    dem_elev        = data["dem_elev"]
    mean_elev       = data["mean"]

    # Get rid of data with 3 or less ground photons in the inter-decile range, and all nans.
    good_data_mask = numphotons_intd > 3 & ~numpy.isnan(meandiff) & ~numpy.isnan(canopy_fraction)
    meandiff           =        meandiff[good_data_mask]
    # meddiff            =         meddiff[good_data_mask]
    # cellstd            =         cellstd[good_data_mask]
    canopy_fraction    = canopy_fraction[good_data_mask]
    # numphotons         =      numphotons[good_data_mask]
    numphotons_intd    = numphotons_intd[good_data_mask]
    dem_elev           =        dem_elev[good_data_mask]
    mean_elev          =       mean_elev[good_data_mask]


    if len(meandiff) < 3:
        if verbose:
            print("Not enough cells to plot statistics. Aborting.")
        return

    # Generate figure. If a figure size isn't given, use the matplotlib.rcParams default.
    if figsize is None:
        figsize = matplotlib.rcParams['figure.figsize']
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, dpi=dpi, figsize=figsize, tight_layout=True)

    plot_label_margin = [0.015, 0.97]
    plot_label_ha = "left"
    plot_label_va = "top"
    plot_label_size = "large"
    plot_label_weight = "book"

    nbins = 1000
    #############################################################################
    # Plot 1, differences from iceast-2 mean.
    ax1.hist(meandiff, bins=nbins)
    # Unicode "minus" sign is \u2212
    ax1.set_title("DEM " + u"\u2212" + " ICESat-2 mean")
    ax1.set_ylabel("% of data cells")
    ax1.set_xlabel("Elevation difference (m)")
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(len(meandiff), decimals=0))

    # Add the lines for mean +- std
    center = numpy.mean(meandiff)
    std = numpy.std(meandiff)
    ax1.axvline(x=center, color="darkred", linewidth=0.75)
    ax1.axvline(x=center+std, color="darkred", linestyle="--", linewidth=0.5)
    ax1.axvline(x=center-std, color="darkred", linestyle="--", linewidth=0.5)

    # Crop the left & right
    cutoffs = numpy.percentile(meandiff, [1, 99])
    # cutoffs = [-max(numpy.abs(cutoffs)), max(numpy.abs(cutoffs))]
    # Do not crop the photo to make the stddev lines fall outside the plot.
    # If they do, reset the min/max cutoff to be 2 stddev away from the mean on that side.
    if (center + std) >= cutoffs[1]:
        cutoffs[1] = center + (std*2)
    if (center - std) <= cutoffs[0]:
        cutoffs[0] = center - (std*2)
    ax1.set_xlim(cutoffs)

    # Detect whether the mean line is closer to the left or the right (we'll put the text box on the other side)
    text_left = not ((center - cutoffs[0]) < (cutoffs[1] - center))
    txt = ax1.text(0.12 if text_left else 0.97,
                   0.95, # 0.85 if text_left else 0.95,
                   "{0:.2f} $\pm$ {1:.2f} m".format(center, std),
                   ha="left" if text_left else "right",
                   va="top",
                   fontsize="small",
                   transform=ax1.transAxes)
    txt.set_bbox(dict(facecolor="white", alpha=0.7, edgecolor="white", boxstyle="square,pad=0"))

    # Add subplot label "a"
    # Add it on the opposite (horizontal) side of wherever the other text has been placed, in this case.
    # if text_left:
    #     ax1_label_x = plot_label_margin[0] + 0.15
    #     ax1_ha = "right"
    # else:
    #     ax1_label_x = plot_label_margin[0]
    #     ax1_ha = "left"
    ax1.text(*plot_label_margin,
             "A" if labels_uppercase else "a",
             ha = plot_label_ha,
             va = plot_label_va,
             fontsize = plot_label_size,
             fontweight = plot_label_weight,
             transform = ax1.transAxes)
    # lbltxt.set_bbox(dict(facecolor="white", alpha=0.7, edgecolor="white", boxstyle="square,pad=0"))

    # 2) Plot 1:1 line of DEM/ICESat-2 elevations.
    #############################################################################
    # Subplot 2, elev-elev correlation line
    dotsize=2
    alpha = 0.25 * max(0.0025, min(4, (math.log10(100)/math.log10(len(mean_elev)))))
    ax2.scatter(mean_elev, dem_elev, s=dotsize, linewidth=0, alpha=alpha)
    ax2.set_title("DEM vs. ICESat-2")
    ax2.set_ylabel("DEM elevation (m)")
    ax2.set_xlabel("ICESat-2 elevation (m)")
    # ax2.autoscale(False) # Keep the line-plotting from expanding the x,y-axes
    xlim = ax2.get_xlim()
    ylim = ax2.get_ylim()

    plotlim =( min(xlim[0], ylim[0]), max(xlim[1],ylim[1]))
    ax2.set_xlim(plotlim)
    ax2.set_ylim(plotlim)
    ax2.plot(plotlim, plotlim, ls="--", c=".3", lw=0.5, alpha=0.6)
    # ax2.plot(xlim, xlim, ls="--", c=".3", lw=1, alpha=0.6)
    # Set the y-ticks the same as the x-ticks.
    xticks = ax2.get_xticks()
    ax2.set_yticks(xticks)

    # Add subplot label "b"
    ax2.text(*plot_label_margin,
             "B"  if labels_uppercase else "b",
             ha=plot_label_ha,
             va=plot_label_va,
             fontsize=plot_label_size,
             fontweight=plot_label_weight,
             transform=ax2.transAxes)
    # lbltxt.set_bbox(dict(facecolor="white", alpha=0.7, edgecolor="white", boxstyle="square,pad=0"))


    # 3) Plot Histogram of # of ICESat-2 photons per cell.
    #############################################################################
    # Plot 4, number of photons in interdecile range
    ax3.hist(numphotons_intd, bins=(numphotons_intd.max() - numphotons_intd.min() + 1), color="darkred")
    ax3.set_title("Number of photons")
    ax3.set_xlabel("Photon count per DEM cell")
    ax3.set_ylabel("% of data cells")
    ax3.yaxis.set_major_formatter(ticker.PercentFormatter(len(numphotons_intd), decimals=0))

    # Crop the left & right, right at 98 percentile.
    cutoff = numpy.percentile(numphotons_intd, 98)
    xmin = ax3.get_xlim()[0]
    ax3.set_xlim(xmin*0.5, cutoff)

    center = numpy.mean(numphotons_intd)
    std = numpy.std(numphotons_intd)

    ax3.text(0.95, 0.95, "{0:d} $\pm$ {1:d} photons per cell".format(int(numpy.round(center)), int(numpy.round(std))),
             ha="right", va="top",
             fontsize="small",
             transform=ax3.transAxes)

    # Add subplot label "c"
    ax3.text(*plot_label_margin,
             "C"  if labels_uppercase else "c",
             ha=plot_label_ha,
             va=plot_label_va,
             fontsize=plot_label_size,
             fontweight=plot_label_weight,
             transform=ax3.transAxes)
    # lbltxt.set_bbox(dict(facecolor="white", alpha=0.7, edgecolor="white", boxstyle="square,pad=0"))


    # 4) Plot Histogram of canopy cover.
    #############################################################################
    # Plot 3, percent canopy cover
    canopy_fraction = (canopy_fraction * 100).astype(float)
    ax4.hist(canopy_fraction, bins=50,color="darkgreen")
    ax4.set_title("Canopy Cover (%)")
    ax4.set_xlabel("% Canopy Cover")
    ax4.yaxis.set_major_formatter(ticker.PercentFormatter(len(canopy_fraction), decimals=0))

    # Crop the right edge at the 99th percentile
    cutoff = numpy.percentile(canopy_fraction, 99)
    # Add just a bit of padding on the left to make more room for the "D" label.
    xmin = ax4.get_xlim()[0] - 0.025*(cutoff - ax4.get_xlim()[0])
    ax4.set_xlim(xmin, cutoff)

    # center = numpy.mean(canopy_fraction)
    # # std = numpy.std(canopy_fraction)
    # median = numpy.median(canopy_fraction)
    canopy_mask = (canopy_fraction > 0.0) & numpy.isfinite(canopy_fraction) & ~numpy.isnan(canopy_fraction)

    ax4.text(0.95, 0.95, "{0:0.1f} % of cells have >0 cover:\n{1:0.1f} $\pm$ {2:0.1f} % canopy cover\nin non-zero cells".format( \
                         numpy.count_nonzero(canopy_mask) * 100 / canopy_fraction.size,
                         numpy.mean(canopy_fraction[canopy_mask]),
                         numpy.std(canopy_fraction[canopy_mask])),
             ha="right", va="top", transform=ax4.transAxes,
             fontsize="small")

    # Add subplot label "d"
    ax4.text(*plot_label_margin,
             "D"  if labels_uppercase else "d",
             ha=plot_label_ha,
             va=plot_label_va,
             fontsize=plot_label_size,
             fontweight=plot_label_weight,
             transform=ax4.transAxes)
    # lbltxt.set_bbox(dict(facecolor="white", alpha=0.7, edgecolor="white", boxstyle="square,pad=0"))


    # Figure title
    if place_name is None:
        place_name = "DEM"
    fig.suptitle("{0} Errors and Distributions\n(N = {1:,} cells)".format(place_name, len(meandiff)))
    fig.tight_layout()

    # Save the figure to disk.
    fig.savefig(output_figure_name)
    if verbose:
        print(output_figure_name, "written.")

    # Clear the figure and close the plot.
    # If the plot is not "plt.close()"'ed, MatPlotLib keeps it in memory indefinitely
    # even after it's no longer referenced, which is... annoying. Gotta close it explicitly here.
    plt.clf()
    plt.close(fig)

    return


def plot_histograms(results_h5_name_or_list, empty_val = my_config.etopo_ndv):
    data = get_data_from_h5_or_list(results_h5_name_or_list, empty_val = empty_val)

    meddiff         = data['diff_median']
    cellstd         = data['stddev'].astype(float)
    meandiff        = data['diff_mean']
    numphotons      = data['numphotons']
    numphotons_intd = data['numphotons_intd']
    canopy_fraction = data["canopy_fraction"]

    # Get rid of data with 3 or less ground photons in the inter-decile range, and all nans.
    good_data_mask = numphotons_intd > 3 & ~numpy.isnan(meandiff) & ~numpy.isnan(meddiff) & ~numpy.isnan(canopy_fraction) \
                     & (meandiff != empty_val) & (meddiff != empty_val) & (canopy_fraction != empty_val) & (cellstd != empty_val)
    meandiff           =        meandiff[good_data_mask]
    cellstd            =         cellstd[good_data_mask]
    meddiff            =         meddiff[good_data_mask]
    canopy_fraction    = canopy_fraction[good_data_mask]
    numphotons         =      numphotons[good_data_mask]
    numphotons_intd    = numphotons_intd[good_data_mask]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, dpi=600, tight_layout=True) #, sharey=True)

    print("Means:", numpy.mean(meandiff), "+/-", numpy.std(meandiff))
    print("Medians:", numpy.mean(meddiff), "+/-", numpy.std(meddiff))
    print("Canopy:", numpy.mean(canopy_fraction), "+/-", numpy.std(canopy_fraction))
    print("Numphotons:", numpy.mean(numphotons), "+/-", numpy.std(numphotons))
    print("Numphotons in Interdecile Range:", numpy.mean(numphotons_intd), "+/-", numpy.std(numphotons_intd))

    nbins = 300

    #############################################################################
    # Plot 1, differences from iceast-2 mean.
    ax1.hist(meandiff, bins=nbins)
    ax1.set_title("DEM - ICESat-2 mean (m)")
    ax1.set_ylabel("Grid cells")
    ax1.set_xlabel("Elevation diff (m)")

    # Crop the left & right 1% tails.
    cutoffs = numpy.percentile(meandiff, [1, 99])
    # cutoffs = [-max(numpy.abs(cutoffs)), max(numpy.abs(cutoffs))]
    ax1.set_xlim(cutoffs)

    # Add the lines for mean +- std
    center = numpy.mean(meandiff)
    std = numpy.std(meandiff)
    ax1.axvline(x=center, color="darkred", linewidth=1)
    ax1.axvline(x=center+std, color="darkred", linestyle="--", linewidth=0.75)
    ax1.axvline(x=center-std, color="darkred", linestyle="--", linewidth=0.75)

    text_left = (abs(cutoffs[0]) > abs(cutoffs[1]))
    txt = ax1.text(0.03 if text_left else 0.97, 0.95, "{0:.2f} $\pm$ {1:.2f} m".format(center, std), ha="left" if text_left else "right", va="top", transform=ax1.transAxes)
    txt.set_bbox(dict(facecolor="white", alpha=0.7, edgecolor="white", boxstyle="square,pad=0"))
    #############################################################################
    # Plot 2, differences from icesat-2 median
    ax2.hist(cellstd, bins=nbins, color="darkred")
    ax2.set_title("Intra-cell variation (roughness) ")
    ax2.set_xlabel("std.dev. within cells (m)")
    # ax2.set_xlim([-2,4])

    # Crop the left & right
    # print(numpy.count_nonzero(numpy.isnan(cellstd)), "NaNs in cellstd", cellstd.count(), "total.")
    # cutoffs = numpy.nanpercentile(cellstd, [0.5, 99.5])
    # ax2.set_xlim(cutoffs)
    cutoffs = numpy.nanpercentile(cellstd, [0, 99])
    xmin = ax2.get_xlim()[0]
    ax2.set_xlim(cutoffs)


    # Add the lines for mean +- std
    # center = numpy.mean(meddiff)
    # std = numpy.std(meddiff)
    # ax2.axvline(x=center, color="darkred", linewidth=1)
    # ax2.axvline(x=center+std, color="darkred", linestyle="--", linewidth=0.75)
    # ax2.axvline(x=center-std, color="darkred", linestyle="--", linewidth=0.75)

    # ax2.text(0.95, 0.95, "{0:.2f} $\pm$ {1:.2f} m".format(center, std), ha="right", va="top", transform=ax2.transAxes)


    #############################################################################
    # Plot 3, percent canopy cover
    canopy_fraction = (canopy_fraction * 100).astype(float)
    # print(canopy_fraction)
    print("min", min(canopy_fraction), "max", max(canopy_fraction), "median", numpy.median(canopy_fraction), "mean", numpy.mean(canopy_fraction))
    ax3.hist(canopy_fraction, bins=50,color="darkgreen")
    ax3.set_title("Canopy Cover (%)")
    ax3.set_xlabel("% Canopy Cover")
    ax3.set_ylabel("Grid cells")
    # ax4.set_xlim([-3,75])

    # Crop the left & right
    cutoff = numpy.percentile(canopy_fraction, 97)
    xmin = ax3.get_xlim()[0]
    ax4.set_xlim(xmin, cutoff)

    # center = numpy.mean(canopy_fraction)
    # # std = numpy.std(canopy_fraction)
    # median = numpy.median(canopy_fraction)
    canopy_mask = (canopy_fraction > 0.0) & numpy.isfinite(canopy_fraction) & ~numpy.isnan(canopy_fraction)

    ax3.text(0.95, 0.95, "In {0:0.1f} % with >0 cover:\n{1:0.1f} $\pm$ {2:0.1f} %".format( \
                         numpy.count_nonzero(canopy_mask) * 100 / canopy_fraction.size,
                         numpy.mean(canopy_fraction[canopy_mask]),
                         numpy.std(canopy_fraction[canopy_mask])),
             ha="right", va="top", transform=ax3.transAxes)

    #############################################################################
    # Plot 4, number of photons in interdecile range
    ax4.hist(numphotons_intd, bins=nbins, color="darkblue")
    ax4.set_title("# photons (interdecile)")
    ax4.set_xlabel("Photon count per cell")
    # ax3.set_xlim([-20,550])

    # Crop the left & right
    cutoff = numpy.percentile(numphotons_intd, 97)
    xmin = ax4.get_xlim()[0]
    ax4.set_xlim(xmin*0.5, cutoff)

    center = numpy.mean(numphotons_intd)
    std = numpy.std(numphotons_intd)

    ax4.text(0.95, 0.95, "{0:d} $\pm$ {1:d} per cell".format(int(numpy.round(center)), int(numpy.round(std))), ha="right", va="top", transform=ax4.transAxes)

    #############################################################################
    # Figure finishing
    # fig.suptitle(figure_titles_dict[fname] + " Distributions")
    fig.suptitle("NE DEMs Data Distributions\n(N = {0:,} cells)".format(len(data)))
    fig.tight_layout()

    # figname = os.path.splitext(fname)[0] + "_distributions.png"
    figname = os.path.join(os.path.split(results_h5_name_or_list[0])[0], os.path.split(os.path.split(results_h5_name_or_list[0])[0])[1]) + "_distributions.png"
    fig.savefig(figname)
    print(figname, "written.")

    plt.clf()
    # plt.close()
    # fig.show()

def plot_error_stats(results_h5_name_or_list, empty_val = my_config.etopo_ndv):
    # print(fname)
    data = get_data_from_h5_or_list(results_h5_name_or_list, empty_val = empty_val)

    meddiff         = data['diff_median']
    cellstd         = data['stddev'].astype(float)
    meandiff        = data['diff_mean']
    numphotons      = data['numphotons']
    numphotons_intd = data['numphotons_intd']
    canopy_fraction = data["canopy_fraction"]
    dem_elev        = data["dem_elev"]
    mean_elev       = data["mean"]
    # Plot some histograms

    # Get rid of data with 3 or less ground photons in the inter-decile range, and all nans.
    good_data_mask = numphotons_intd > 3 & ~numpy.isnan(meandiff) & ~numpy.isnan(meddiff) & ~numpy.isnan(canopy_fraction)
    meandiff           =        meandiff[good_data_mask]
    meddiff            =         meddiff[good_data_mask]
    cellstd            =         cellstd[good_data_mask]
    canopy_fraction    = canopy_fraction[good_data_mask]
    numphotons         =      numphotons[good_data_mask]
    numphotons_intd    = numphotons_intd[good_data_mask]
    dem_elev           =        dem_elev[good_data_mask]
    mean_elev          =       mean_elev[good_data_mask]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, dpi=600, tight_layout=True) #, sharey=True)

    dotsize=1

    #############################################################################
    # Subplot 1, elev-elev correlation line
    ax1.scatter(mean_elev, dem_elev, s=dotsize)
    ax1.set_ylabel("DEM elevation (m)")
    ax1.set_xlabel("ICESat-2 elevation (m)")
    xlim = ax1.get_xlim()
    ax1.autoscale(False) # Keep the line-plotting from expanding the x,y-axes
    ax1.plot(xlim, xlim, ls="--", c=".3", lw=1)

    #############################################################################
    # Subplot 2, error vs icesat-2 elevation
    ax2.scatter(dem_elev, meandiff, s=dotsize, color="darkred")
    ax2.axhline(y=0, ls="--", c=".3", lw=1)
    ax2.set_xlabel("ICESat-2 elev (m)")
    ax2.set_ylabel("DEM error (m)")

    ax3.scatter(canopy_fraction*100, meandiff, s=dotsize, color="darkgreen")
    ax3.axhline(y=0, ls="--", c=".3", lw=1)
    ax3.set_xlabel("Canopy Cover (%)")
    ax3.set_ylabel("DEM error (m)")

    ax4.scatter(numphotons_intd, meandiff, s=dotsize, color="purple")
    ax4.axhline(y=0, ls="--", c=".3", lw=1)
    ax4.set_xlabel("# photons per cell")
    ax4.set_ylabel("DEM errors (m)")


    # fig.suptitle(figure_titles_dict[fname] + " Error Stats")
    fig.suptitle("NE DEM Error Stats\n(N = {0:,} cells)".format(len(data)))
    fig.tight_layout()


    # figname = os.path.splitext(fname)[0] + "_stats.png"
    figname = os.path.join(os.path.split(results_h5_name_or_list[0])[0], os.path.split(os.path.split(results_h5_name_or_list[0])[0])[1]) + "_stats.png"
    fig.savefig(figname)
    print(figname, "written.")

    plt.clf()

if __name__ == '__main__':

    plot_histogram_and_error_stats_4_panels("/home/mmacferrin/Research/DATA/DEMs/CUDEM/data/CONUS/NCEI_ninth_Topobathy_2014_8483/chesapeake_bay/icesat2_results/ncei19_n39x75_w076x25_2019v1_results.h5",
                                            "/home/mmacferrin/Research/DATA/DEMs/CUDEM/data/CONUS/NCEI_ninth_Topobathy_2014_8483/chesapeake_bay/icesat2_results/ncei19_n39x75_w076x25_2019v1_results_plot.png")

    # plot_histograms(h5_names)
    # plot_error_stats(h5_names)
    # for fname in fnames:
    #     print("\n====", fname)
    #     plot_histograms(fname)
    #     plot_error_stats(fname)
    #     # input("<Press <Enter> to continue>")

    print("Done")
