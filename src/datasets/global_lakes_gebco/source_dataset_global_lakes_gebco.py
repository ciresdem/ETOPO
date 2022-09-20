# -*- coding: utf-8 -*-

"""Source code for the global_lakes ETOPO source dataset class."""

import os
# import geopandas
import pandas
from osgeo import gdal
import subprocess
import numpy
import scipy.stats
import skimage.measure, skimage.color
import matplotlib.pyplot as plt
import re
import argparse
import multiprocessing

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset
import datasets.CopernicusDEM.source_dataset_CopernicusDEM as source_dataset_CopernicusDEM
import datasets.global_lakes_globathy.source_dataset_global_lakes_globathy as source_dataset_global_lakes_globathy
import datasets.GEBCO.source_dataset_GEBCO as source_dataset_GEBCO
import etopo.coastline_mask as coastline_mask
import utils.configfile
# etopo_config = configfile.config()

class source_dataset_global_lakes_gebco(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "global_lakes_gebco_config.ini" )):
        """Initialize the global_lakes source dataset object."""

        super(source_dataset_global_lakes_gebco, self).__init__("global_lakes_gebco", configfile)

        # Small GDF of
        self.lake_exceptions_df = None

    def get_array_from_gtiff(self, gtif):
        """Get the raw array from a geotiff. Make sure the dataset is closed before returning."""
        ds = gdal.Open(gtif, gdal.GA_ReadOnly)
        if ds is None:
            return None
        band = ds.GetRasterBand(1)
        array = band.ReadAsArray()
        band = None
        ds = None
        del band, ds
        return array

    def create_subdata_tempgrid(self, grid_datafile, datalist_fname, output_fname, ndv=None, resample_alg="near", verbose=True):
        """Use the waffles -M stacks module to create a resampled grid using the datalist provided,
        in the same grid as the grid_datafile.
        Save to output_fname.

        Used for creating grids of copernicus data as the equal projection & grid of the grid we're surveying.
        """
        # If it already exists, work is done. Just exit da fuck outta here.
        if os.path.exists(output_fname):
            ds = gdal.Open(output_fname, gdal.GA_ReadOnly)
            # Check whether the file was validly read. If it's a corrupted or incomplete file, just delete it.
            if ds is None:
                os.remove(output_fname)
            else:
                return

        assert os.path.exists(grid_datafile)
        ds = gdal.Open(grid_datafile, gdal.GA_ReadOnly)
        epsg = coastline_mask.get_dataset_epsg(ds)
        assert type(epsg) == int
        bbox, xy_step = coastline_mask.get_bounding_box_and_step(ds, invert_for_waffles=True)
        # Free the memory and close the file.
        ds = None

        # Waffles adds a .tif, so strip it off the output name.
        outf_base, outf_ext = os.path.splitext(output_fname)
        assert outf_ext.lower() == ".tif"

        stacks_args = ["waffles", "-M", "stacks:supercede=True",
                       "-R", "/".join([str(x) for x in bbox]),
                       "-E", "/".join([str(abs(x)) for x in xy_step]),
                       "-S", resample_alg,
                       "-N", str(ndv),
                       "-P", "epsg:" + str(epsg),
                       "-F", "GTiff",
                       "-O", outf_base,
                       "-w", # use weights (necessary for 'stacks')
                       datalist_fname
                       ]

        # If a nodata value is provided, include it in the args, just behind the final datalist argument.
        if ndv is not None:
            stacks_args = stacks_args[:-1] + ["-N", str(ndv)] + [stacks_args[-1]]

        if verbose:
            print(" ".join(stacks_args))

        subprocess.run(stacks_args,
                       stdin = None,
                       stdout = None if verbose else subprocess.PIPE,
                       stderr = None if verbose else subprocess.STDOUT,
                       encoding = "utf-8")

        assert os.path.exists(output_fname)

        return


    def get_data_directory(self, resolution_s):
        if int(resolution_s) == 15:
            return self.config._abspath(self.config.source_datafiles_directory_15s)
        elif int(resolution_s) == 1:
            return self.config._abspath(self.config.source_datafiles_directory_1s)
        else:
            return self.config._abspath(self.config.source_datafiles_directory)

    def is_tile_included_in_lake_exceptions(self, tilename, resolution_s):
        if self.lake_exceptions_df is None:
            # Read the little csv file.
            etopo_config = utils.configfile.config()
            csvname = etopo_config.lake_exceptions_csv.format(resolution_s)
            df = pandas.read_csv(csvname, index_col = False)
            self.lake_exceptions_df = df
        else:
            df = self.lake_exceptions_df

        tilename_marker = re.search("[NS](\d{2})[EW](\d{3})", os.path.split(tilename)[1]).group()
        sub_df = df[df.filename == tilename_marker]
        # If this tiles isn't mentioned anywhere in the file, return True.
        return len(sub_df) > 0

    def check_lake_exceptions(self, tilename, lake_array, resolution_s):
        """Check whether the lake in question has an exception listed. If nothing is listed here for this lake,
        return None.
        If we are supposed to use GEBCO, return True.
        If we are supposed to use Globathy, return False.

        Note, the CSV has a "Globathy" column, the answer is the opposite of that, basically.
        Look at etopo_config.ini::lake_exceptions_csv to get the file path of this CSV."""
        if self.lake_exceptions_df is None:
            # Read the little csv file.
            etopo_config = utils.configfile.config()
            csvname = etopo_config.lake_exceptions_csv.format(resolution_s)
            df = pandas.read_csv(csvname, index_col = False)
            self.lake_exceptions_df = df
        else:
            df = self.lake_exceptions_df

        tilename_marker = re.search("[NS](\d{2})[EW](\d{3})", os.path.split(tilename)[1]).group()
        sub_df = df[df.filename == tilename_marker]
        # If this tiles isn't mentioned anywhere in the file, just ignore and move on.
        # This will be the case for the majority of tiles.
        if len(sub_df) == 0:
            return None

        for i,df_row in sub_df.iterrows():

            # if df_row.filename == "N30E105":
            #     print("\t", "N30E105", df_row.row, df_row.col, lake_array[df_row.row, df_row.col])
            # Check if the [row.row,row.col] value of the lake_array is True. That means this lake has been flagged.
            if (df_row.row == -1 and df_row.col == -1) or lake_array[df_row.row, df_row.col]:
                # Then check the value of row.Globathy, that'll tell us whether we need to use GEBCO or not.
                if df_row.Globathy:
                    # Note: The return value is whether or not we're forcing it to use GEBCO. The dataframe
                    # df.Globathy field states whether or not we're forcing it to use Globathy. So the dataframe
                    # column value and the return value are opposites here.
                    return False
                else:
                    assert df_row.Globathy == False
                    return True

            elif df_row.globathy_flags == "ALL_OTHER_GLOBATHY":
                # This is a flag we've used to designate one or a few lakes to use GEBCO, but all the others in the tile as Globathy.
                return False

        return None

    def create_gebco_global_lakes(self, resolution_s=15, numprocs=1, overwrite=False, verbose=True):
        """Create a global lakes outline dataset with GEBCO elevations in it.

        Test to see where this is valid and where it isn't.
        Run the create_globathy_global_lakes() method first to get all the lake outlines."""

        # Get dataset for Copernicus & GEBCO, as well as global_lakes_globathy
        copernicus_ds = source_dataset_CopernicusDEM.source_dataset_CopernicusDEM()
        gebco_ds = source_dataset_GEBCO.source_dataset_GEBCO()
        globathy_ds = source_dataset_global_lakes_globathy.source_dataset_global_lakes_globathy()

        # Get the datalist fnames for each of the source datasets for lakes.
        copernicus_datalist = copernicus_ds.get_datalist_fname(resolution_s = resolution_s)
        gebco_datalist = gebco_ds.get_datalist_fname(resolution_s = resolution_s)

        # Directory to put copernicus and gebco temp grids into.
        temp_dir = self.get_data_temp_dir(resolution_s = resolution_s)

        # Loop through all the Globathy lake tiles.
        glob_gdf = globathy_ds.get_geodataframe(resolution_s = resolution_s, verbose=verbose)

        running_procs = []
        running_fnames = []
        total_finished = 0

        N = len(glob_gdf.index)

        for i, row in enumerate(glob_gdf.iterrows()):
            # iterrows() returns an "(index, row)" tuple. Don't care about the incides here, so ignore them.
            _, row = row
            # The Globathy file name.
            gb_fname = row.filename
            process_started = False

            # if verbose:
            #     print("{0}/{1}".format(i + 1, len(glob_gdf)), os.path.split(gb_fname)[1])

            if not os.path.exists(gb_fname):
                if verbose:
                    print(gb_fname, "does not exist.")
                total_finished += 1
                if i < (N-1):
                    continue
                else:
                    process_started = True

            ymin = numpy.round(row.ytop + (row.ysize * row.yres))

            gebco_lakes_fname = self.config.global_lakes_fname_template.format("N" if (numpy.round(ymin) >= 0) else "S",
                                                                               int(abs(numpy.round(ymin))),
                                                                               "E" if (numpy.round(row.xleft) >= 0) else "W",
                                                                               int(abs(numpy.round(row.xleft))))
            gebco_lakes_fpath = os.path.join(self.get_data_directory(resolution_s), gebco_lakes_fname)
            if os.path.exists(gebco_lakes_fpath):
                # Test to see if it's a valid file.
                ds = gdal.Open(gebco_lakes_fpath, gdal.GA_ReadOnly)
                if ds is None or overwrite:
                    os.remove(gebco_lakes_fpath)
                else:
                    # if verbose:
                    #     print("\tAlready exists. Moving on.")
                    total_finished += 1
                    if i < (N-1):
                        continue
                    else:
                        process_started = True

            # If there still hasn't been room in the process queue for this proc, *or* we're on the very last
            # process and some others are still running, keep looping and monitor them.
            while (not process_started) or ((i==(N-1)) and len(running_procs) > 0):
            # First, loop to find any running procs that are done. If so, we'll add another, then wrap them up.
            # while len(running_procs) > 0:
            #     found_a_dead_proc = False
            #     for proc in running_procs:
            #         if not proc.is_alive():
            #             found_a_dead_proc = True
            #             break
            #     if found_a_dead_proc:
            #         break
            #     else:
            #         time.sleep(0.01)

                # If there's room for a new process, start it up and add it to the queue.
                if (not process_started) and (len(running_procs) < numprocs):

                    p = multiprocessing.Process(target=create_one_gebco_lakes_tile,
                                                args=(self,
                                                      gb_fname,
                                                      gebco_lakes_fpath,
                                                      temp_dir,
                                                      copernicus_datalist,
                                                      gebco_datalist,
                                                      resolution_s,
                                                      ),
                                                kwargs={"check_exceptions": True,
                                                        "output_csv" : True,
                                                        "verbose" : False
                                                        }
                                                )
                    p.start()
                    # Mark that this process has been started.
                    process_started = True
                    running_procs.append(p)
                    running_fnames.append(gebco_lakes_fpath)

                procs_to_remove = []
                fnames_to_remove = []
                for proc, fname in zip(running_procs, running_fnames):
                    if not proc.is_alive():
                        proc.join()
                        proc.close()

                        procs_to_remove.append(proc)
                        fnames_to_remove.append(fname)
                        total_finished += 1
                        if verbose:
                            print("{0}/{1}".format(total_finished, N), fname, ("written." if os.path.exists(fname) else ""))

                for proc, fname in zip(procs_to_remove, fnames_to_remove):
                    running_procs.remove(proc)
                    running_fnames.remove(fname)

        # If we'd set the lake exceptions GDF, re-set it to none, in case we call this function again at a different resolution.
        self.lake_exceptions_gdf = None

        if overwrite:
            gdf_fname = self.get_geopkg_object(verbose=verbose).get_gdf_filename(resolution_s=resolution_s)
            if os.path.exists(gdf_fname):
                os.remove(gdf_fname)
        return self.get_geodataframe(resolution_s=resolution_s, verbose=verbose)


    def get_empty_lake_stats_df(self):
        """Create a dataset with named colunms and 2 rows empty data.

        We'll strip it out later to return an empty dataframe with these columns.
        """
        columns_data = {"tile_lon_min": [0, 0],
                        "tile_lat_min": [0, 0],
                        "lake_num_n"  : [0, 0],
                        "lake_size_pix": [0, 0],
                        "lake_xj_min": [0, 0],
                        "lake_xj_max": [0, 0],
                        "lake_yi_min": [0, 0],
                        "lake_yi_max": [0, 0],
                        "copernicus_lake_elev": [0.0, 0.0],
                        "copernicus_lake_elev_frac_same": [0.0, 0.0],
                        "gebco_elev_mean": [0.0, 0.0],
                        "gebco_elev_std": [0.0, 0.0],
                        "gebco_elev_min": [0.0, 0.0],
                        "gebco_elev_max": [0.0, 0.0],
                        "globathy_elev_mean": [0.0, 0.0],
                        "globathy_elev_std": [0.0, 0.0],
                        "globathy_elev_min": [0.0, 0.0],
                        "globathy_elev_max": [0.0, 0.0],
                        "use_gebco_elevs": [False, False]
                        }

        # Get rid of the two rows, copy it to make it its own dataframe (not a slice of another), and return it.
        return pandas.DataFrame(data=columns_data)[[False, False]].copy()

    def get_lake_stats_df(self, resolution_s=15, overwrite=False, verbose=True):
        """Create the CSV to keep track of lake statistics as we're generating them.
        Useful for debugging our lake-selection criteria.
        """
        csv_fname = self.config.gebco_lakes_stats_csv.format(resolution_s)

        if (not overwrite) and os.path.exists(csv_fname):
            return pandas.read_csv(csv_fname, index_col = False)

        df = get_empty_lake_stats_df()
        self.write_lakes_csv(df, csv_fname, verbose=verbose)
        return df

    def write_lakes_csv(self, df, csv_fname, verbose=True):
        """Write out the lakes stats dataframe to disk."""
        df.to_csv(csv_fname, index=False)
        if verbose:
            print(os.path.split(csv_fname)[1], "written with {0} entries.".format(len(df.index)))

    def get_data_dir(self, resolution_s = 15):
        if resolution_s == 15:
            return self.config._abspath(self.config.source_datafiles_directory_15s)
        else:
            assert resolution_s == 1
            return self.config._abspath(self.config.source_datafiles_directory_1s)

    def get_data_temp_dir(self, resolution_s = 15, create_if_not_present=True):
        tempdir = os.path.join(self.get_data_dir(resolution_s = resolution_s), "temp")
        if not os.path.exists(tempdir) and create_if_not_present:
            os.mkdir(tempdir)
        assert os.path.isdir(tempdir)
        return tempdir


def create_one_gebco_lakes_tile(gc_object : source_dataset_global_lakes_gebco,
                                globathy_fname : str,
                                gebco_fname : str,
                                temp_dir : str,
                                copernicus_datalist : str,
                                gebco_datalist : str,
                                resolution_s : int,
                                check_exceptions: bool = True,
                                output_csv : bool = True,
                                verbose : bool = True) -> int:
    """For multiprocessing, here is a single method to create one GEBCO tile.
    Return the number of distinct lakes included from GEBCO in this tile."""
    gebco_csv_fpath = os.path.splitext(gebco_fname)[0] + "_stats.csv"

    # Use skimage.label to outline & label each lake in the globathy tile.
    gb_ds = gdal.Open(globathy_fname, gdal.GA_ReadOnly)
    gb_band = gb_ds.GetRasterBand(1)
    array = gb_band.ReadAsArray()
    ndv = gb_band.GetNoDataValue()
    # if verbose:
    #     print(" Done.")

    # If no lakes appear in this dataset, simply move along.
    if numpy.all(array == ndv):
        if verbose:
            print("\t0 lakes found. Moving on.")
        return 0

    # Label the image with skimage.label to find all the lakes
    # connectivity = 1 indicates only use non-diagonal (bordering) connectivity. Pixels connected only diagonally do not count.
    lab_array, nlabels = skimage.measure.label(array != ndv, background=0, return_num=True, connectivity=1)
    # If we did our check correctly up top there should be at least one damned lake in here.
    assert nlabels >= 1
    # print(nlabels, "lakes.")
    # print(lab_array)

    # A boolean flag to see if we should bother checking any lakes in this tile against the exceptions.
    # If not, skip it.
    if check_exceptions is True:
        do_we_check_exceptions = gc_object.is_tile_included_in_lake_exceptions(globathy_fname, resolution_s)
    else:
        do_we_check_exceptions = False
    # image_label_overlay = skimage.color.label2rgb(lab_array, bg_label=0)

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.imshow(image_label_overlay)

    # foobar
    # Create a temporary grid of both Copernicus and GEBCO elevations at the same grid as each Globathy tile.
    # Use waffles and the datalists of each dataset for this.
    # Create an copy of the Globathy file for our GEBCO file. Create an empty array to write into it.

    # Copernicus files for getting the surface elevation of the lake surface
    copernicus_grid_fname = os.path.join(temp_dir,
                                         os.path.split(globathy_fname)[1].replace("lakes_globathy_", "copernicus_temp_"))
    # GEBCO "nearest neighbor" grid for getting the lake bed elevation without interpolation.
    gebco_grid_nearest_fname = os.path.join(temp_dir, os.path.split(globathy_fname)[1].replace("lakes_globathy_",
                                                                                         "gebco_nearest_temp_"))

    # Only create this if we need to write any lakes out.
    # GEBCO "bilinear sample" grid for putting better lakebed elevations into the final grid (if using GEBCO)
    gebco_grid_bilinear_fname = gebco_grid_nearest_fname.replace("_nearest_", "_bilinear_")

    # Generate an empty grid to start copying lakes into.
    gebco_lakes_grid = array.copy()
    gebco_lakes_grid[:, :] = ndv

    gbfn = os.path.split(globathy_fname)[1]

    # Create copernicus temp file
    # Create GEBCO "nearest neighbor" temp file.
    # Create GEBCO "bilinear" temp file.
    if verbose:
        print("  Creating source grids... ", end="")

    got_all_arrays = False
    while not got_all_arrays:
        for j, (dlist, fname, algorithm) in enumerate(zip([copernicus_datalist, gebco_datalist, gebco_datalist],
                                                          [copernicus_grid_fname, gebco_grid_nearest_fname,
                                                           gebco_grid_bilinear_fname],
                                                          ["near", "near", "bilinear"])):

            if verbose:
                print("Copernicus... " if j == 0 else ("GEBCO nearest... " if j == 1 else "GEBCO bilinear... "), end="")
            # print("\tCreating", os.path.split(fname)[1])
            gc_object.create_subdata_tempgrid(globathy_fname,
                                              dlist,
                                              fname,
                                              ndv=0 if (j == 0) else ndv,
                                              # Use zero for NDV of copernicus, the default NDV otherwise.
                                              resample_alg=algorithm,
                                              verbose=False)
        if verbose:
            print("Done.")

        copernicus_array = gc_object.get_array_from_gtiff(copernicus_grid_fname)
        gebco_nearest_array = gc_object.get_array_from_gtiff(gebco_grid_nearest_fname)
        gebco_bilinear_array = gc_object.get_array_from_gtiff(gebco_grid_bilinear_fname)
        # For some fucking reason, we seem to occasionally not be able to read some of these files. If so, delete them
        # and keep looping until we get them. Hopefully this doesn't enter an infinite loop here.
        if (copernicus_array is not None) and (gebco_nearest_array is not None) and (gebco_bilinear_array is not None):
            got_all_arrays = True
        else:
            if copernicus_array is None:
                os.remove(copernicus_grid_fname)
            if gebco_nearest_array is None:
                os.remove(gebco_grid_nearest_fname)
            if gebco_bilinear_array is None:
                os.remove(gebco_grid_bilinear_fname)


    used_any_gebco_elevs = False
    numlakes_included = 0

    # Get the dataframe of lakes stats
    lks_df = gc_object.get_empty_lake_stats_df()

    # Get the tile edge bounds from the file name. Maybe not the best way to do it, but
    # given our file naming conventions containing N**E***, it works in this case.
    tile_ymin = (-1 if (re.search("[NS](?=\d{2}[EW]\d{3})", gbfn).group() == "S") else 1) * \
                int(re.search("(?<=[NS])\d{2}(?=[EW](\d{3}))", gbfn).group())
    tile_xmin = (-1 if (re.search("(?<=[NS](\d{2}))[EW](?=\d{3})", gbfn).group() == "W") else 1) * \
                int(re.search("(?<=[NS](\d{2})[EW])\d{3}", gbfn).group())

    # if re.search(r"[NS]\d{2}[EW]\d{3}", gbfn).group() == "N30E105":
    #     print(nlabels, "lakes in tile", os.path.basename(gbfn))

    # Loop through all the lakes in the dataset, starting with lake # 1 (0 is background).
    for LN in range(1, nlabels+1):
        # Get the copernicus lake heights -- figure out the copernicus elevaion of the lake.
        lake_mask = (lab_array == LN)
        lake_size = numpy.count_nonzero(lake_mask)
        copernicus_elevs = copernicus_array[lake_mask]
        # gebco_nearest_elevs = gebco_nearest_array[lake_mask]
        gebco_bilinear_elevs = gebco_bilinear_array[lake_mask]

        # if re.search(r"[NS]\d{2}[EW]\d{3}", gbfn).group() == "N30E105":
        #     print("\t", LN, numpy.count_nonzero(lake_mask))

        if lake_size < 5:
            # # If it's a tiny lake, 1-2 grid cells, just use the fucking gebco data *if* it's below the lake surface threshold.
            # if gebco_bilinear_elevs[0] < copernicus_elevs[0]:
            #     gebco_lakes_grid[lake_mask] = gebco_bilinear_elevs
            #     used_gebco_elevs = True
            #     used_any_gebco_elevs = True
            #     numlakes_included += 1
            #     gebco_data = gebco_bilinear_elevs
            # elif gebco_nearest_elevs[0] < copernicus_elevs[0]:
            #     gebco_lakes_grid[lake_mask] = gebco_nearest_elevs
            #     used_gebco_elevs = True
            #     used_any_gebco_elevs = True
            #     numlakes_included += 1
            #     gebco_data = gebco_nearest_elevs
            # # Otherwise, just don't skip that lake and don't use the GEBCO elevations (use globathy)
            # else:
            used_gebco_elevs = False
            gebco_data = gebco_bilinear_elevs

            copernicus_surface_elev = numpy.mean(copernicus_elevs)
            copernicus_mode_fraction = 1.

        # elif lake_size == 2:
        #     if numpy.all(gebco_bilinear_elevs < copernicus_elevs):
        #         gebco_lakes_grid[lake_mask] = gebco_bilinear_elevs
        #         used_gebco_elevs = True
        #         used_any_gebco_elevs = True
        #         numlakes_included += 1
        #         gebco_data = gebco_bilinear_elevs
        #
        #     elif numpy.all(gebco_nearest_elevs < copernicus_elevs):
        #         gebco_lakes_grid[lake_mask] = gebco_nearest_elevs
        #         used_gebco_elevs = True
        #         used_any_gebco_elevs = True
        #         numlakes_included += 1
        #         gebco_data = gebco_nearest_elevs
        #     else:
        #         used_gebco_elevs = False
        #         gebco_data = gebco_bilinear_elevs
        #
        #     copernicus_surface_elev, copernicus_mode_count = scipy.stats.mode(copernicus_elevs)
        #     copernicus_surface_elev = copernicus_surface_elev[0]
        #     copernicus_mode_count = copernicus_mode_count[0]
        #     copernicus_mode_fraction = copernicus_mode_count / lake_size

        else:
            assert lake_size >= 5
            copernicus_surface_elev, copernicus_mode_count = scipy.stats.mode(copernicus_elevs)
            copernicus_surface_elev = copernicus_surface_elev[0]
            copernicus_mode_count = copernicus_mode_count[0]
            copernicus_mode_fraction = copernicus_mode_count / lake_size
            # copernicus_surface_elev = scipy.stats.mode(copernicus_elevs).mode[0]

            # force_gebco will be True (to use GEBCO), False (to use Globathy), or None (just use the default algorithm).
            force_gebco = None
            if do_we_check_exceptions:
                force_gebco = gc_object.check_lake_exceptions(globathy_fname, lake_mask, resolution_s)

            gebco_data = gebco_bilinear_elevs

            num_gebco_below = numpy.count_nonzero(gebco_bilinear_elevs < copernicus_surface_elev)
            # If 80% or more of the lake bed elevations are below the lake surface, use the GEBCO
            # elevations, and just manually lower the few that are above the lake surface.
            # This (I think) seems to take care of skipping most the small lakes where GEBCO sucks.
            if (((num_gebco_below / float(lake_size)) >= 0.80) and (force_gebco != False)) or (force_gebco == True):
                # Find the next shallowest pixel and set all GEBCO elevations that are *above* the lake surface to that.
                try:
                    next_shallowest_depth = numpy.max(gebco_bilinear_elevs[gebco_bilinear_elevs < copernicus_surface_elev])
                except ValueError:
                    next_shallowest_depth = copernicus_surface_elev - 1.0
                gebco_bilinear_elevs[gebco_bilinear_elevs >= copernicus_surface_elev] = next_shallowest_depth
                gebco_lakes_grid[lake_mask] = gebco_bilinear_elevs
                used_gebco_elevs = True
                used_any_gebco_elevs = True
                numlakes_included += 1
            else:
                used_gebco_elevs = False

        # Calculate where the lake is in the image.
        B = numpy.argwhere(lake_mask)
        (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1

        # Get the globathy data for this lake.
        globathy_lake_data = array[lake_mask]

        # Add this metadata to the lake record.
        lks_df.loc[0 if (len(lks_df) == 0) else (lks_df.index.max() + 1)] = \
                                             [tile_xmin,
                                              tile_ymin,
                                              LN,
                                              lake_size,
                                              xstart,
                                              xstop,
                                              ystart,
                                              ystop,
                                              copernicus_surface_elev,
                                              copernicus_mode_fraction,
                                              numpy.mean(gebco_data) if lake_size > 1 else gebco_data[0],
                                              numpy.std(gebco_data) if lake_size > 1 else 0,
                                              numpy.min(gebco_data),
                                              numpy.max(gebco_data),
                                              numpy.mean(globathy_lake_data) if lake_size > 1 else globathy_lake_data[0],
                                              numpy.std(globathy_lake_data) if lake_size > 1 else 0,
                                              numpy.min(globathy_lake_data),
                                              numpy.max(globathy_lake_data),
                                              used_gebco_elevs]

    if used_any_gebco_elevs:
        # Save out the data file.
        gdal.GetDriverByName("GTiff").CreateCopy(gebco_fname, gb_ds)
        if verbose:
            print(os.path.split(gebco_fname)[1], "created with {0} lakes, ".format(numlakes_included), end="")

        gc_ds = gdal.Open(gebco_fname, gdal.GA_Update)
        band = gc_ds.GetRasterBand(1)
        band.WriteArray(gebco_lakes_grid)
        band.SetNoDataValue(ndv)
        band.ComputeStatistics(0)

        band = None
        gc_ds = None
        gb_band = None
        gb_ds = None
        del band, gc_ds, gb_band, gb_ds
        if verbose:
            print("and updated.")

    # Write lake stats out to a master CSV df.
    if output_csv and len(lks_df.index) > 0:
        gc_object.write_lakes_csv(lks_df, gebco_csv_fpath, verbose=verbose)

    return numlakes_included


def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Generate all the GEBCO lakes tiles. The global_lakes_globathy layer should be generated first.")
    parser.add_argument("-resolution", "-r", default=None, help="Resolution (in s) of tiles to produce. 1 or 15. Default both.")
    parser.add_argument("-numprocs", "-np", type=int, default=utils.parallel_funcs.physical_cpu_count(),
                        help="Number of processes to run simultaneously. Defaults to the number of physical cores on the machine. (Note: NOT YET IMPLEMENTED.)")
    parser.add_argument("--overwrite", "-o", default=False, action="store_true", help="Overwrite existing files. Default False")
    parser.add_argument("--quiet", "-q", default=False, action="store_true",
                        help="Run in quiet mode.")

    return parser.parse_args()


if __name__ == "__main__":
    args = define_and_parse_args()

    if args.resolution is None:
        resolutions = (15,1)
    else:
        resolutions = (int(args.resolution),)

    gl = source_dataset_global_lakes_gebco()

    for res in resolutions:
        gl.create_gebco_global_lakes(resolution_s=res,
                                     numprocs=args.numprocs,
                                     overwrite=args.overwrite,
                                     verbose=not args.quiet)