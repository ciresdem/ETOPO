import os
import geopandas
import pandas
import cartopy
import numpy
import re
import shapely.geometry
import numexpr

####################################3
# Include the base /src/ directory of thie project, to add all the other modules.
import import_parent_dir; import_parent_dir.import_src_dir_via_pythonpath()
####################################3
import utils.progress_bar
import utils.parallel_funcs
import utils.configfile
my_config = utils.configfile.config()


def add_lat_lons(df):
    """From the filename, get the lat/lon of each grid-cell point, from the filename and the i,j position of the pixel."""

    lat = numpy.empty((len(df),), dtype=float)
    lon = numpy.empty((len(df),), dtype=float)

    for idx, ((i,j), row) in enumerate(df.iterrows()):
        fname = row['filename']
        resolution_s = float(re.search('(?<=_)\d{1,2}(?=s_)', fname).group())
        res_deg = resolution_s / (60*60)
        tile_lat_str = re.search('(?<=_)[NS]\d{2}(?=[EW]\d{3}_)', fname).group()
        tile_lon_str = re.search('(?<=_[NS]\d{2})[EW]\d{3}(?=_)', fname).group()
        tile_lat = float(int(tile_lat_str[1:]) * (1 if tile_lat_str[0] == "N" else -1))
        tile_lon = float(int(tile_lon_str[1:]) * (1 if tile_lon_str[0] == "E" else -1))

        # The latitude is calculated from the top of the tile (which is why I'm adding "+1" to tile_lat, since these are
        # all 1-deg subsets of the larger tiles.)
        # Calculating for the center of each grid-cell, which is why I'm adding 0.5 to i & j.
        lat[idx] = (tile_lat + 1) - (res_deg * (i + 0.5))
        lon[idx] = tile_lon + (res_deg * (j + 0.5))

        if ((idx + 1) % 5000 == 0) or (idx == (len(df) - 1)):
            utils.progress_bar.ProgressBar((idx + 1), len(df), suffix="{0:,}/{1:,}".format(idx + 1, len(df)))

    df['latitude'] = lat
    df['longitude'] = lon

    return df


def subset_dfs_and_add_latlons():
    dirname = os.path.join(my_config._abspath(my_config.etopo_validation_results_directory.format(15)),
                            "2022.09.29", "plots")

    h5names = [os.path.join(dirname, fn) for fn in os.listdir(dirname) if re.search(r"totalresults_gte\d{2}\.h5", fn) is not None]

    for hdf_name in h5names:
        print("Reading", os.path.basename(hdf_name), "...")
        results_df = pandas.read_hdf(hdf_name)

        print("Calculating lat/lons, eliminating unneeded data, and writing back to disk.")
        results_df = add_lat_lons(results_df)

        if 'median' in results_df.columns:
            print("Old columns:", results_df.columns)
            results_df = results_df.drop(
                ["median", "stddev", "interdecile_range", "range", "10p", "90p", "diff_median", "min_dist_from_center"],
                axis=1)

            print("New columns:", results_df.columns)

        # Write it back out to disk.
        if os.path.exists(hdf_name):
            os.remove(hdf_name)
        print("Writing back out", os.path.basename(hdf_name))
        results_df.to_hdf(hdf_name, "icesat2", complevel=2, complib="zlib")

        # print(results_df.columns)
        # print(results_df)


def generate_subtile_error_gdf(pct_coverage_cutoff="10%", verbose=True):
    """Go through the resluts (at a various cutoff %age of coverage), and tally up error-stat values for each
    1-degree grid-cell of the ETOPO validation results, so we can plot those in a map."""

    # hdf_name = os.path.join(my_config._abspath(my_config.etopo_validation_results_directory.format(15)),
    #                         "2022.09.29", "plots", f"total_results_gte{pct_coverage_cutoff}.h5")

    results_dir = os.path.join(my_config._abspath(my_config.etopo_validation_results_directory.format(15)),
                               "2022.09.29")
    results_fnames = [os.path.join(results_dir, fn) for fn in os.listdir(results_dir) if (fn.find("_results.h5") > 0)]
    # print(results_fnames[0:20])
    # foobar

    gpkg_fname = os.path.abspath(os.path.join(results_dir, "maps", "results_map_gte{0}.gpkg".format(
                                               str(pct_coverage_cutoff).replace('%', 'pct'))))

    # if verbose:
    # print("Reading", os.path.basename(hdf_name))

    # results_df = pandas.read_hdf(hdf_name)
    # unique_tilenames = sorted(results_df['filename'].unique().tolist())
    if verbose:
        print("Processing", len(results_fnames), "tiles results.")
        # print(len(unique_tilenames), "unique tiles and", len(results_df),
        #       "total records at {0:0.0f}% coverage.".format(pct_coverage_cutoff))

    polygons = [None] * len(results_fnames)
    numcells = numpy.zeros((len(results_fnames),), dtype=int)
    biases = numpy.zeros((len(results_fnames),), dtype=float)
    stddevs = numpy.zeros((len(results_fnames),), dtype=float)
    rmses = numpy.zeros((len(results_fnames),), dtype=float)
    coverage_cutoff_frac = numpy.zeros((len(results_fnames),), dtype=float)

    for i, tile_results_fname in enumerate(results_fnames):
        # df_subset = results_df[results_df["filename"] == tilename]

        tile_df = pandas.read_hdf(tile_results_fname)

        tile_coverage = tile_df["coverage_frac"]
        if isinstance(pct_coverage_cutoff, str) and pct_coverage_cutoff.find("%") > 0:
            # We've used a percentile rather than a straight cutoff value. In this case,
            # take the top X percentile of coverages.
            percentile_value = 100 - int(pct_coverage_cutoff[:pct_coverage_cutoff.find("%")])
            assert 0 <= percentile_value <= 100
            coverage_cutoff_frac[i] = numpy.percentile(tile_coverage, [percentile_value])

        else:
            # Otherwise, we give a value in an integer or floating-point number to subset the coverage.
            assert type(pct_coverage_cutoff) in (int, float)
            coverage_cutoff_frac[i] = float(pct_coverage_cutoff) / 100.

        # Subset the dataframe.
        tile_df = tile_df[tile_coverage >= coverage_cutoff_frac[i]]

        # Find the dataframe file corresponding to this tilename.
        # tile_latlon_str = re.search("(?<=_)[NS]\d{2}[EW]\d{3}(?=_)", tile_results_fname).group()

        tile_lat_str = re.search('(?<=_)[NS]\d{2}(?=[EW]\d{3}_)', tile_results_fname).group()
        tile_lon_str = re.search('(?<=_[NS]\d{2})[EW]\d{3}(?=_)', tile_results_fname).group()
        tile_lat = float(int(tile_lat_str[1:]) * (1 if tile_lat_str[0] == "N" else -1))
        tile_lon = float(int(tile_lon_str[1:]) * (1 if tile_lon_str[0] == "E" else -1))

        polygons[i] = shapely.geometry.box(tile_lon, tile_lat, tile_lon + 1, tile_lat + 1)
        numcells[i] = len(tile_df)

        diff_mean = tile_df['diff_mean']
        biases[i] = diff_mean.mean()
        stddevs[i] = diff_mean.std() if (len(diff_mean) > 1) else 0
        rmses[i] = numpy.sqrt(sum(diff_mean ** 2) / len(diff_mean))

        if verbose:
            # if ((i + 1) % 10) == 0 or ((i + 1) == len(unique_tilenames)):
            utils.progress_bar.ProgressBar((i + 1), len(results_fnames),
                                           suffix="{0}/{1}".format(i + 1, len(results_fnames)))


    tile_basenames = [os.path.basename(tfn) for tfn in results_fnames]
    gdf = geopandas.GeoDataFrame(data={"filename": tile_basenames,
                                       "numcells": numcells,
                                       "bias": biases,
                                       "stddev": stddevs,
                                       "rmse": rmses,
                                       "coverage_cutoff": coverage_cutoff_frac,
                                       "geometry": polygons},
                                 geometry="geometry",
                                 crs="EPSG:4326")

    gdf.to_file(gpkg_fname, layer="tile_results", driver='GPKG')
    if verbose:
        print(os.path.basename(gpkg_fname), "written.")
        print()

# TODO: Finish this.
def create_etopo_error_map(results_df,
                           map_filename=os.path.join(
                               my_config._abspath(my_config.etopo_validation_results_directory.format(15)),
                               "plots", "ETOPO_RMSE_fig.png"),
                           map_title="ETOPO RMSE",
                           cmap='jet',
                           alpha=0.6,
                           verbose=True):
    """Generate an image of a world map showing areas that have been downloaded."""
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    fig = plt.figure(figsize=(12, 7))
    prj = cartopy.crs.PlateCarree()
    ax = plt.axes(projection=prj)
    # fig, ax = plt.subplots(1,1, figsize=(12,7)) #, projection=4326)
    # ax.set_xlim(-180, 180)
    # ax.set_ylim(-90, 90)
    ax.set_extent([-180, 180, -90, 90], crs=prj)
    # Add the background world in light grey
    world.plot(color="lightgrey", ax=ax)
    # Add lat/lon graticules every 10 degrees, with labels
    ax.gridlines(xlocs=numpy.arange(-180, 180, 10),
                 ylocs=numpy.arange(-90, 90, 10),
                 draw_labels=True,
                 x_inline=False, y_inline=False,
                 color='k',
                 linestyle="dotted",
                 linewidth=0.5)
    # Plot the boxes, colored by their downloaded status
    # dmin = min(download_gdf[progress_fieldname])
    # dmax = max(download_gdf[progress_fieldname])
    cm_obj = plt.get_cmap(cmap, lut=2)
    # color_dmin = tuple(list(cm_obj(dmin)[0:3]) + [alpha])
    # color_dmax = tuple(list(cm_obj(dmax)[0:3]) + [alpha])

    # TODO: Edit from here. I need a value for EACH 1-deg subset tile, in a geodataframe.

    # If "is_downloaded" or "is_populated" field is all true, create an empty row at the end with a non-existent polygon.
    # Just need to give at least one "false" reading in order for the colormap to
    # still show correctly on a fully-completed map where 'is_downloaded' are all True.
    if numpy.all(download_gdf[progress_fieldname]):
        added_row = True
        if progress_fieldname == "is_downloaded":
            download_gdf.loc[len(download_gdf)] = [0, 0, 0, 0, 0, False,
                                                   shapely.geometry.Polygon(([0, 0], [0.00001, 0], [0, 0]))]
        elif progress_fieldname == "is_populated":
            download_gdf.loc[len(download_gdf)] = ["", 0, 0, 0, 0, 0, 0, 0, False,
                                                   shapely.geometry.Polygon(([0, 0], [0.00001, 0], [0, 0]))]
    else:
        added_row = False

    # Add the data to the plot.
    download_gdf.plot(column=progress_fieldname, ax=ax, alpha=alpha, cmap=cm_obj)

    # Get rid of that temporary extra row if we added it.
    if added_row:
        # After it's plotted, drop the extra row we just added.
        download_gdf.drop(download_gdf.tail(1).index, inplace=True)

    # Add some helpful text to the figure.
    plt.text(0.5, 0.9, map_title,
             ha="center", va="center",
             fontsize="x-large",
             fontweight="bold",
             transform=fig.transFigure)

    total_tiles = len(download_gdf)
    tiles_downloaded = numpy.count_nonzero(download_gdf[progress_fieldname])
    pct_download = 100.0 * tiles_downloaded / total_tiles

    # Key for the complete color
    plt.text(0.13, 0.81, "■ = complete",
             ha="left", va="center",
             color=color_dmax,
             backgroundcolor="white",
             fontsize="medium",
             fontweight="bold",
             bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'),
             transform=fig.transFigure)

    plt.text(0.5, 0.10, "{0:,} of {1:,} tiles complete, {2:.1f}%".format(tiles_downloaded, total_tiles, pct_download),
             ha="center", va="center",
             fontsize="large",
             fontweight="bold",
             transform=fig.transFigure)
    now_str = datetime.datetime.now().astimezone().strftime("%a %Y-%m-%d %H:%M:%S %Z")
    plt.text(0.5, 0.06, "Last updated " + now_str,
             ha="center", va="center",
             fontsize="large",
             transform=fig.transFigure)

    # Save the figure.
    fig.savefig(map_filename, dpi=300)
    if verbose:
        print(os.path.split(map_filename)[1], "written.")
    # Make sure to close the figure to free up the memory.
    plt.close(fig)

    return


if __name__ == "__main__":
    # for i in range(40, 51):
    # i=0
    generate_subtile_error_gdf(pct_coverage_cutoff="5%", verbose=True)

    # args = [[p] for p in list(range(20, 40))]
    # print("Processing", len(args), "ETOPO tile maps from {0} to {1} % coverage.".format(args[0][0], args[-1][0]))
    # utils.parallel_funcs.process_parallel(generate_subtile_error_gdf,
    #                                       args_lists=args,
    #                                       max_nprocs=len(args),
    #                                       use_progress_bar_only=True,
    #                                       kwargs_list={'verbose': False},
    #                                       verbose=True)