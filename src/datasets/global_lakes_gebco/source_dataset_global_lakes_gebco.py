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
# import utils.configfile
# import utils.configfile
# etopo_config = configfile.config()

class source_dataset_global_lakes_gebco(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "global_lakes_gebco_config.ini" )):
        """Initialize the global_lakes source dataset object."""

        super(source_dataset_global_lakes_gebco, self).__init__("global_lakes_gebco", configfile)

    def get_array_from_gtiff(self, gtif):
        """Get the raw array from a geotiff. Make sure the dataset is closed before returning."""
        ds = gdal.Open(gtif, gdal.GA_ReadOnly)
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
        """
        # If it already exists, work is done. Just exit da fuck outta here.
        if os.path.exists(output_fname):
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
        if ndv != None:
            stacks_args = stacks_args[:-1] + ["-N", str(ndv)] + [stacks_args[-1]]

        if verbose:
            print(" ".join(stacks_args))

        subprocess.run(stacks_args,
                       stdin = None,
                       stdout = None if verbose else subprocess.PIPE,
                       stderr = None if verbose else subprocess.STDOUT,
                       encoding = "utf-8")

        return


    def get_data_directory(self, resolution_s):
        if int(resolution_s) == 15:
            return self.config.source_datafiles_directory_15s
        elif int(resolution_s) == 1:
            return self.config.source_datafiles_directory_1s
        else:
            return self.config.source_datafiles_directory

    def create_gebco_global_lakes(self, resolution_s=15, verbose=True):
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

        # Get the dataframe of lakes stats
        lks_df = self.get_lake_stats_df(resolution_s = resolution_s,
                                        verbose = verbose)

        # Directory to put copernicus and gebco temp grids into.
        temp_dir = self.get_data_temp_dir(resolution_s = resolution_s)

        # Loop through all the Globathy lake tiles.
        glob_gdf = globathy_ds.get_geodataframe(resolution_s = resolution_s, verbose=verbose)
        for row in glob_gdf.iterrows():
            i, row = row
            # print(row)
            # foobar
            # print(i, row.filename)
            # The Globathy file name.
            gb_fname = row.filename

            # Use skimage.label to outline & label each lake in the globathy tile.
            if verbose:
                print("{0}/{1}".format(i+1, len(glob_gdf)), "Opening", os.path.split(gb_fname)[1])
            gb_ds = gdal.Open(gb_fname, gdal.GA_ReadOnly)
            gb_band = gb_ds.GetRasterBand(1)
            array = gb_band.ReadAsArray()
            ndv = gb_band.GetNoDataValue()
            # if verbose:
            #     print(" Done.")

            # If no lakes appear in this dataset, simply move along.
            if numpy.all(array == ndv):
                if verbose:
                    print("\t0 lakes found. Moving on.")
                continue

            # Label the image with skimage.label to find all the lakes
            # connectivity = 1 indicates only use non-diagonal (bordering) connectivity. Pixels connected only diagonally do not count.
            lab_array, nlabels = skimage.measure.label(array != ndv, background=0, return_num = True, connectivity = 1)
            # If we did our check correctly up top there should be at least one damned lake in here.
            assert nlabels >= 1
            # print(nlabels, "lakes.")
            # print(lab_array)

            # image_label_overlay = skimage.color.label2rgb(lab_array, bg_label=0)

            # fig, ax = plt.subplots(figsize=(10, 6))
            # ax.imshow(image_label_overlay)

            # foobar
            # Create a temporary grid of both Copernicus and GEBCO elevations at the same grid as each Globathy tile.
            # Use waffles and the datalists of each dataset for this.
            # Create an copy of the Globathy file for our GEBCO file. Create an empty array to write into it.

            # Copernicus files for getting the surface elevation of the lake surface
            copernicus_grid_fname = os.path.join(temp_dir, os.path.split(gb_fname)[1].replace("lakes_globathy_", "copernicus_temp_"))
            # GEBCO "nearest neighbor" grid for getting the lake bed elevation without interpolation.
            gebco_grid_nearest_fname = os.path.join(temp_dir, os.path.split(gb_fname)[1].replace("lakes_globathy_", "gebco_nearest_temp_"))

            # Only create this if we need to write any lakes out.
            # GEBCO "bilinear sample" grid for putting better lakebed elevations into the final grid (if using GEBCO)
            gebco_grid_bilinear_fname = gebco_grid_nearest_fname.replace("_nearest_", "_bilinear_")

            # Generate an empty grid to start copying lakes into.
            gebco_lakes_grid = array.copy()
            gebco_lakes_grid[:,:] = ndv

            ymin = row.ytop + (row.yres * row.ysize)

            gebco_lakes_fname = self.config.global_lakes_fname_template.format("N" if (ymin >= 0) else "S",
                                                                               int(abs(ymin)),
                                                                               "E" if (row.xleft >= 0) else "W",
                                                                               int(abs(row.xleft)))
            gebco_lakes_fpath = os.path.join(self.get_data_directory(resolution_s), gebco_lakes_fname)
            if os.path.exists(gebco_lakes_fpath):
                if verbose:
                    print("\tAlready exists. Moving on.")
                continue

            # Create copernicus temp file
            # Create GEBCO "nearest neighbor" temp file.
            # Create GEBCO "bilinear" temp file.
            if verbose:
                print("  Creating source grids... ", end="")
            for j, (dlist, fname, algorithm) in enumerate(zip([copernicus_datalist, gebco_datalist, gebco_datalist],
                                                              [copernicus_grid_fname, gebco_grid_nearest_fname, gebco_grid_bilinear_fname],
                                                              ["near", "near", "bilinear"])):

                if verbose:
                    print("Copernicus... " if j == 0 else ("GEBCO nearest... " if j == 1 else "GEBCO bilinear... "), end="")
                # print("\tCreating", os.path.split(fname)[1])
                self.create_subdata_tempgrid(gb_fname,
                                             dlist,
                                             fname,
                                             ndv=0 if (j==0) else ndv, # Use zero for NDV of copernicus, the default NDV otherwise.
                                             resample_alg=algorithm,
                                             verbose=False)
            if verbose:
                print("Done.")


            copernicus_array = self.get_array_from_gtiff(copernicus_grid_fname)
            gebco_nearest_array = self.get_array_from_gtiff(gebco_grid_nearest_fname)
            gebco_bilinear_array = self.get_array_from_gtiff(gebco_grid_bilinear_fname)

            used_any_gebco_elevs = False
            numlakes_included = 0

            # Loop through all the lakes in the dataset, starting with lake # 1 (0 is background).
            for LN in range(1,nlabels):
                # Get the copernicus lake heights -- figure out the copernicus elevaion of the lake.
                lake_mask = (lab_array == LN)
                lake_size = numpy.count_nonzero(lake_mask)
                copernicus_elevs = copernicus_array[lake_mask]
                gebco_nearest_elevs = gebco_nearest_array[lake_mask]
                gebco_bilinear_elevs = gebco_bilinear_array[lake_mask]

                if lake_size == 1:
                    # If it's a tiny lake, 1-2 grid cells, just use the fucking gebco data *if* it's below the lake surface threshold.
                    if gebco_bilinear_elevs[0] < copernicus_elevs[0]:
                        gebco_lakes_grid[lake_mask] = gebco_bilinear_elevs
                        used_gebco_elevs = True
                        used_any_gebco_elevs = True
                        numlakes_included += 1
                        gebco_data = gebco_bilinear_elevs
                    elif gebco_nearest_elevs[0] < copernicus_elevs[0]:
                        gebco_lakes_grid[lake_mask] = gebco_nearest_elevs
                        used_gebco_elevs = True
                        used_any_gebco_elevs = True
                        numlakes_included += 1
                        gebco_data = gebco_nearest_elevs
                    # Otherwise, just don't skip that lake and don't use the GEBCO elevations (use globathy)
                    else:
                        used_gebco_elevs = False
                        gebco_data = gebco_bilinear_elevs

                    copernicus_surface_elev = copernicus_elevs[0]
                    copernicus_mode_fraction = 1.

                elif lake_size == 2:
                    if numpy.all(gebco_bilinear_elevs < copernicus_elevs):
                        gebco_lakes_grid[lake_mask] = gebco_bilinear_elevs
                        used_gebco_elevs = True
                        used_any_gebco_elevs = True
                        numlakes_included += 1
                        gebco_data = gebco_bilinear_elevs

                    elif numpy.all(gebco_nearest_elevs < copernicus_elevs):
                        gebco_lakes_grid[lake_mask] = gebco_nearest_elevs
                        used_gebco_elevs = True
                        used_any_gebco_elevs = True
                        numlakes_included += 1
                        gebco_data = gebco_nearest_elevs
                    else:
                        used_gebco_elevs = False
                        gebco_data = gebco_bilinear_elevs

                    copernicus_surface_elev, copernicus_mode_count = scipy.stats.mode(copernicus_elevs)
                    copernicus_surface_elev = copernicus_surface_elev[0]
                    copernicus_mode_count = copernicus_mode_count[0]
                    copernicus_mode_fraction = copernicus_mode_count / lake_size

                else:
                    assert lake_size >= 3
                    copernicus_surface_elev, copernicus_mode_count = scipy.stats.mode(copernicus_elevs)
                    copernicus_surface_elev = copernicus_surface_elev[0]
                    copernicus_mode_count = copernicus_mode_count[0]
                    copernicus_mode_fraction = copernicus_mode_count / lake_size
                    # copernicus_surface_elev = scipy.stats.mode(copernicus_elevs).mode[0]

                    gebco_data = gebco_bilinear_elevs

                    num_gebco_below = numpy.count_nonzero(gebco_bilinear_elevs < copernicus_surface_elev)
                    # If 80% or more of the lake bed elevations are below the lake surface, use the GEBCO
                    # elevations, and just manually lower the few that are above the lake surface.
                    # This (I think) seems to take care of skipping most the small lakes where GEBCO sucks.
                    if (num_gebco_below / lake_size) >= 0.80:
                        # Find the next shallowest pixel and set all GEBCO elevations that are *above* the lake surface to that.
                        next_shallowest_depth = numpy.max(gebco_bilinear_elevs[gebco_bilinear_elevs < copernicus_surface_elev])
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
                lks_df.loc[lks_df.index.max()+1] = [row.xleft,
                                                    ymin,
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
                gdal.GetDriverByName("GTiff").CreateCopy(gebco_lakes_fpath, gb_ds)
                if verbose:
                    print(os.path.split(gebco_lakes_fpath)[1], "created with {0} lakes, ".format(numlakes_included), end="")

                gc_ds = gdal.Open(gebco_lakes_fpath, gdal.GA_Update)
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

            # Strip off any zero rows if they exist.
            if (lks_df.lake_size_pix == 0).sum() > 0:
                lks_df = lks_df[lks_df.lake_size_pix > 0]
                # Reset the index number, so that the append operation up top is still inserting at the end.
                lks_df.reset_index(drop=True, inplace=True)

            if len(lks_df.index) > 0:
                self.write_lakes_csv(lks_df, self.config.gebco_lakes_stats_csv.format(resolution_s), verbose=verbose)

        # If lake elevation is less than Copernicus height, and (1-2 pixels, *or* of varied heights in GEBCO),
        # Flag it as a probable GEBCO accurate lakebed elevation.
        # Then copy it into GEBCO lakebed array.
        # Otherwise, skip that lake.

        # Write lake stats out to a master CSV df.

    def get_lake_stats_df(self, resolution_s=15, overwrite=False, verbose=True):
        """Create the CSV to keep track of lake statistics as we're generating them.
        Useful for debugging our lake-selection criteria.
        """
        csv_fname = self.config.gebco_lakes_stats_csv.format(resolution_s)

        if (not overwrite) and os.path.exists(csv_fname):
            return pandas.read_csv(csv_fname, index_col = False)

        # Create a dataset with named colunms and 2 rows empty data. We'll save it with that but strip it out later.
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

        df = pandas.DataFrame(data=columns_data)

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
        if not os.path.exists(tempdir):
            os.mkdir(tempdir)
        assert os.path.isdir(tempdir)
        return tempdir


if __name__ == "__main__":
    gl = source_dataset_global_lakes_gebco()
    # gl.create_gebco_global_lakes(resolution_s = 1)
    gl.create_gebco_global_lakes(resolution_s = 15)
