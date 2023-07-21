
import os
import pandas
from osgeo import gdal
# import subprocess
import re
import numpy

import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()

import utils.traverse_directory
import utils.parallel_funcs
import etopo.coastline_mask

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "finished_tiles",
                                        "15s", "2022.09.29")
                           )
stats_dir = os.path.join(data_dir, "stats")
coastline_dir = os.path.join(stats_dir, "coastline_mask_simple")

# print(data_dir)
# print(coastline_dir)

etopo_files = utils.traverse_directory.list_files(data_dir,
                                                  regex_match=r"ETOPO_2022_v1_15s_(\w{7})(_2022\.09\.((29)|(30)))?\.tif\Z",
                                                  depth=0)
# print(len(etopo_files), "tiles.")

coastline_masks = [os.path.join(coastline_dir, os.path.splitext(os.path.basename(fn))[0] + "_coastline_mask_simple.tif")
                    for fn in etopo_files]

# Generate all the coastline masks. Comment this out after completion.
def create_etopo_coastline_mask_simple(etopo_tile, outfile):
    etopo.coastline_mask.create_coastline_mask(etopo_tile,
                                               return_bounds_step_epsg=False,
                                               mask_out_lakes=False,
                                               include_gmrt=False,
                                               mask_out_buildings=False,
                                               mask_out_urban=False,
                                               mask_out_nhd=False,
                                               use_osm_planet=False,
                                               output_file=outfile,
                                               run_in_tempdir=True,
                                               horizontal_datum_only=True,
                                               verbose=False)


# args_list = list(zip(etopo_files, coastline_masks))
# utils.parallel_funcs.process_parallel(create_etopo_coastline_mask_simple,
#                                       args_lists=args_list,
#                                       outfiles=coastline_masks,
#                                       max_nprocs=10,
#                                       )

stats_csvs = [os.path.join(stats_dir, os.path.splitext(os.path.basename(fn))[0] + "_stats.csv") for fn in etopo_files]

# for i, (etile, cmask, stats_csv) in enumerate(zip(etopo_files, coastline_masks, stats_csvs)):
#

def compute_etopo_tile_stats(etopo_tile, coastline_mask, stats_file, verbose=True):
    """Compute the elevation distributions for tiles """
    # Get the thresholds set up for the rows in the data table.
    thresholds_m = numpy.arange(-11000,9000,100)
    thresholds_ft = numpy.arange(-36000,29250,200)
    thresholds_ft_in_m = thresholds_ft / 3.2808399
    thresh_m_all = numpy.concatenate((thresholds_m, thresholds_ft_in_m))
    # print(thresh_m_all)

    thresh_all_for_table = numpy.concatenate((thresholds_m, thresholds_ft))
    thresh_units = (["m"] * len(thresholds_m)) + (["ft"] * len(thresholds_ft))
    nrows = len(thresh_all_for_table)

    # Open the ETOPO tile, get the elev data
    if verbose:
        print("Reading", os.path.basename(etopo_tile))
    ds = gdal.Open(etopo_tile, gdal.GA_ReadOnly)
    elevs_array = ds.GetRasterBand(1).ReadAsArray()
    del ds

    # Open the coastline tile, get the coastline_mask
    if verbose:
        print("Reading", os.path.basename(coastline_mask))
    cds = gdal.Open(coastline_mask, gdal.GA_ReadOnly)
    land_mask = cds.GetRasterBand(1).ReadAsArray().astype(bool)
    # Mask out any land beneath the red sea. This would just be masking errors at the shoreline.
    land_mask[elevs_array < -420] = False
    del cds

    n_total = [elevs_array.size] * nrows
    n_land = [numpy.count_nonzero(land_mask)] * nrows
    elevs_land = elevs_array[land_mask]
    n_total_above_threshold = numpy.zeros(thresh_all_for_table.shape, dtype=int)
    pct_total_above_threshold = numpy.zeros(thresh_all_for_table.shape, dtype=float)
    n_land_above_threshold = numpy.zeros(thresh_all_for_table.shape, dtype=int)
    pct_land_above_threshold = numpy.zeros(thresh_all_for_table.shape, dtype=float)

    for i, tr in enumerate(thresh_m_all):
        n_total_above_threshold[i] = numpy.count_nonzero(elevs_array >= tr)
        pct_total_above_threshold[i] = (n_total_above_threshold[i] * 100.) / n_total[i]
        if n_land[i] > 0:
            n_land_above_threshold[i] = numpy.count_nonzero(elevs_land >= tr)
            pct_land_above_threshold[i] = (n_land_above_threshold[i] * 100.) / n_land[i]
        else:
            n_land_above_threshold[i] = 0
            pct_land_above_threshold[i] = 0

    table_dict = {"elev_threshold": thresh_all_for_table,
                  "elev_unit": thresh_units,
                  "n_total": n_total,
                  "n_total_above_threshold": n_total_above_threshold,
                  "pct_total_above_threshold": pct_total_above_threshold,
                  "n_land": n_land,
                  "n_land_above_threshold": n_land_above_threshold,
                  "pct_land_above_threshold": pct_land_above_threshold}

    df = pandas.DataFrame(data=table_dict)
    df.to_csv(stats_file, index=False)
    if verbose:
        print(os.path.basename(stats_file), "written.")

args_list = list(zip(etopo_files, coastline_masks, stats_csvs))
utils.parallel_funcs.process_parallel(compute_etopo_tile_stats,
                                      args_lists=args_list,
                                      outfiles=stats_csvs,
                                      max_nprocs=10,
                                      kwargs_list={"verbose": False}
                                      )

# NOW, let's tally up all the files.
stats_file_all = os.path.join(stats_dir, "ETOPO_2022_stats_all.csv")

for i, csv in enumerate(stats_csvs):
    df = pandas.read_csv(csv, index_col=False, header=0)
    if i == 0:
        master_df = df
        continue

    # Tally up all the columns in each dataframe into this one, starting with the first one.
    master_df["n_total"] = master_df["n_total"] + df["n_total"]
    master_df["n_total_above_threshold"] = master_df["n_total_above_threshold"] + df["n_total_above_threshold"]
    master_df["n_land"] = master_df["n_land"] + df["n_land"]
    master_df["n_land_above_threshold"] = master_df["n_land_above_threshold"] + df["n_land_above_threshold"]

# Then, recalculate all the percentage stats
master_df["pct_total_above_threshold"] = master_df["n_total_above_threshold"] * 100. / master_df["n_total"]
master_df["pct_land_above_threshold"] = master_df["n_land_above_threshold"] * 100. / master_df["n_land"]

master_df.to_csv(stats_file_all, index=False)
print(os.path.basename(stats_file_all), "written.")


################################################################################################
# Do that again, but tally it up with a cosine correction for latitude to scale for land areas to
# not bias toward the poles (especially Antarctica).
stats_file_scaled = os.path.join(stats_dir, "ETOPO_2022_stats_scaled.csv")

for i, csv in enumerate(stats_csvs):
    df = pandas.read_csv(csv, index_col=False, header=0)
    if i == 0:
        master_df = df
        continue

    # Get the mean latitude of that 15s tile.
    min_lat_str = re.search(r"(?<=_)[NS]\d{2}(?=[EW]\d{3}_)", os.path.basename(csv)).group()
    mean_lat_deg = ((-1 if (min_lat_str[0] == "S") else 1) * int(min_lat_str[1:])) + (15./2)
    mean_lat_cos_correction = numpy.cos(numpy.radians(mean_lat_deg))

    # Tally up all the columns in each dataframe into this one, starting with the first one.
    master_df["n_total"] = master_df["n_total"] + (df["n_total"] * mean_lat_cos_correction)
    master_df["n_total_above_threshold"] = master_df["n_total_above_threshold"] + (df["n_total_above_threshold"] * mean_lat_cos_correction)
    master_df["n_land"] = master_df["n_land"] + (df["n_land"] * mean_lat_cos_correction)
    master_df["n_land_above_threshold"] = master_df["n_land_above_threshold"] + (df["n_land_above_threshold"] * mean_lat_cos_correction)

# Then, recalculate all the percentage stats
master_df["pct_total_above_threshold"] = master_df["n_total_above_threshold"] * 100. / master_df["n_total"]
master_df["pct_land_above_threshold"] = master_df["n_land_above_threshold"] * 100. / master_df["n_land"]

master_df.to_csv(stats_file_scaled, index=False)
print(os.path.basename(stats_file_scaled), "written.")