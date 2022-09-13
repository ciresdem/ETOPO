# -*- coding: utf-8 -*-

"""Source code for the global_lakes_flat ETOPO source dataset class."""

import os
import subprocess
from osgeo import gdal
import multiprocessing
import time
import argparse
import re

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset
import datasets.dataset_geopackage
import utils.progress_bar
import utils.parallel_funcs
import utils.configfile

etopo_config = utils.configfile.config()

class source_dataset_global_lakes_flat(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "global_lakes_flat_config.ini" )):
        """Initialize the global_lakes_flat source dataset object."""

        super(source_dataset_global_lakes_flat, self).__init__("global_lakes_flat", configfile)

    def create_global_lakes_flat_tiles(self,
                                       resolution_s = 15,
                                       crm_only_if_1s = True,
                                       depth_source=1,
                                       max_id = None,
                                       min_id = None,
                                       min_area = None,
                                       max_area = None,
                                       numprocs = 1,
                                       overwrite = False,
                                       verbose = True):
        """Possible values of depth_source:
            - 'hydrolakes' uses Hydrolakes "Depth_avg" field (tends to be shallower)
            - 'globathy'
            - a number (i.e. 1, 5, etc) in meters depth for all lakes in that tile..
        min_id and max_id can subset by ID number. Largest lakes are first, going down the list.
        min_area and max_area subset by lake area, in km2.
        Can combine min & max to get a range, but should probably only do one of "area" or "id" filtering.
        """
        resolution_s = int(resolution_s)
        assert resolution_s in (1,15)

        # 1) Get the tile outlines and resolutions of each grid to create. (Use the etopo empty tiles to do this.)
        etopo_gpkg = datasets.dataset_geopackage.ETOPO_Geopackage(resolution_s)
        etopo_gdf = etopo_gpkg.get_gdf(crm_only_if_1s = crm_only_if_1s,
                                 verbose = verbose)

        # 2) Get the directory of the output files and filenames for each output
        data_dir = self.config._abspath(self.config.source_datafiles_directory_1s if (resolution_s == 1) else self.config.source_datafiles_directory_15s)

        # print(data_dir)
        # print(gdf)
        # print(gdf.columns)
        # return gdf

        # 3) Generate all the output grids. Lakes everywhere!

        running_procs = []
        running_fnames = []
        running_tdirs = []
        files_finished = 0
        tempdirs_total = []

        for i, row in etopo_gdf.iterrows():
            # This only works because for ETOPO, the resolution is the same as the grid size in degrees.
            ybottom = int(row.ytop - resolution_s)
            # Generate the file name and the path it'll be written to.
            fname_out = self.config.global_lakes_fname_template.format("N" if ybottom >= 0 else "S",
                                                                       int(abs(ybottom)),
                                                                       "E" if row.xleft >= 0 else "W",
                                                                       int(abs(row.xleft)))
            fpath_out = os.path.join(data_dir, fname_out)

            # If the file already exists and we're not overwriting, just skip & move along.
            if os.path.exists(fpath_out):
                if overwrite:
                    os.remove(fpath_out)
                else:
                    files_finished += 1

        if not overwrite and verbose:
            print("{0} of {1} tiles finished, {2} remaining.".format(files_finished, len(etopo_gdf), len(etopo_gdf) - files_finished))

        files_finished = 0
        try:

            for i, row in enumerate(etopo_gdf.iterrows()):
                # iterrows() returns a an (index, row) tuple. Just get the row.
                row = row[1]
                # print(i)
                # print(type(row))
                # for j in range(len(row)):
                #     print(j, type(row[j]), row[j])
                # # print(row)
                # foobar
                # NOTE: This ONLY works because in ETOPO's case, the resolution in seconds (1,15) is equal to the tile-size in degrees (1/15)
                ybottom = int(row.ytop - resolution_s)
                xright = int(row.xleft + resolution_s)
                # Generate the file name and the path it'll be written to.
                fname_out = self.config.global_lakes_fname_template.format("N" if ybottom >= 0 else "S",
                                                                           int(abs(ybottom)),
                                                                           "E" if row.xleft >= 0 else "W",
                                                                           int(abs(row.xleft)))
                fpath_out = os.path.join(data_dir, fname_out)

                # If the file already exists and we're not overwriting, just skip & move along.
                if not overwrite and os.path.exists(fpath_out):
                    files_finished += 1
                    continue

                while len(running_procs) >= numprocs:
                    procs_to_remove = []
                    fnames_to_remove = []
                    tdirs_to_remove = []
                    # Find processes that are done running, add them to procs to be removed.
                    for proc, fname, tempdir in zip(running_procs, running_fnames, running_tdirs):
                        if not proc.is_alive():
                            # If a process has finished, label it to be removed from the queue
                            proc.join()
                            proc.close()
                            procs_to_remove.append(proc)
                            fnames_to_remove.append(fname)
                            tdirs_to_remove.append(tempdir)

                    # Remove the process from the list of running processes.
                    for proc, fname, tempdir in zip(procs_to_remove, fnames_to_remove, tdirs_to_remove):
                        files_finished += 1
                        if verbose:
                            print("{0}/{1}".format(files_finished, len(etopo_gdf)), fname)
                        running_procs.remove(proc)
                        running_fnames.remove(fname)
                        # Get rid of the tempdir, including anything in there.
                        if os.path.exists(tempdir):
                            tfns = os.listdir(tempdir)
                            for tf in tfns:
                                os.remove(tf)
                            os.rmdir(tempdir)

                    time.sleep(0.01)

                # Waffles lakes command looks like:
                # waffles -M lakes -R -86/-85/30/31 -E 1s -P epsg:4326 -D ~/Research/DATA/ETOPO/scratch_data/ -k -O lakes_test_N30W086
                # Create the waffles command arguments there.
                lakes_flags = ""
                if depth_source is not None:
                    assert type(depth_source) in (int, float) or depth_source in ("globathy", "hydrolakes")
                    lakes_flags = lakes_flags + ":depth=" + str(depth_source)
                if max_id is not None:
                    lakes_flags = lakes_flags + ":max_id={0:d}".format(max_id)
                if min_id is not None:
                    lakes_flags = lakes_flags + ":min_id={0:d}".format(min_id)
                if min_area is not None:
                    lakes_flags = lakes_flags + ":min_area={0}".format(min_area)
                if max_area is not None:
                    lakes_flags = lakes_flags + ":max_area={0}".format(max_area)

                args = ["waffles",
                        "-M", "lakes" + lakes_flags,
                        "-R", "{0}/{1}/{2}/{3}".format(int(row.xleft),
                                                       xright,
                                                       ybottom,
                                                       int(row.ytop)),
                        "-E", "{0}s".format(resolution_s), # This makes the assumption that the xres and yres are the same. This should be true.
                        "-P", "epsg:4326",
                        "-D", etopo_config.etopo_cudem_cache_directory,
                        "-k",
                        "-O", os.path.splitext(fpath_out)[0]
                        ]

                # Execute the command.
                # if verbose:
                #     print("{0}/{1}".format(i+1, len(etopo_gdf)),
                #           fname_out + "...", end="" , flush=True)
                # print("")
                # continue

                # For debugging purposes, print out the commands I'm issuing.
                # TODO: Comment this out later.
                # print("\n" + " ".join(args))
                # return

                # Get the name of the temporary dir to work within. The subprocess will create it.
                tempdir_name = os.path.join(os.path.dirname(fpath_out), "temp" + str(i))

                # try:
                    # Run the lakes module command.
                p = multiprocessing.Process(target = generate_one_tile,
                                            args = (args,fpath_out),
                                            kwargs = {"tempdir": tempdir_name,
                                                      "verbose": verbose}
                                            )

                                                      # 'encoding': "utf-8",
                                                      # "stdin"   : None,
                                                      # "stdout"  : None,
                                                      # "stderr"  : subprocess.PIPE}
                p.start()
                running_procs.append(p)
                running_fnames.append(fpath_out)
                running_tdirs.append(tempdir_name)
                tempdirs_total.append(tempdir_name)
                        # subprocess.run(args,
                        #                encoding="utf-8",
                        #                stdin=None,
                        #                stdout=None,
                        #                stderr=subprocess.PIPE)
                # except KeyboardInterrupt as e:
                #     # If the process gets interrupted, delete any partially-written file.
                #     if os.path.exists(fpath_out):
                #         os.remove(fpath_out)
                #     raise e

                # if p.returncode == 0:
                #     # Write out statistics to the file if they are needed. I *think* this does it.
                #     assert os.path.exists(fpath_out)
                #     if verbose:
                #         print("Setting stats...", end='')
                #     ds = gdal.Open(fpath_out)
                #     band = ds.GetRasterBand(1)
                #     band.ComputeStatistics(0)
                #     band = None
                #     ds = None
                #
                #     if verbose:
                #         print(" Done.", flush=True)

                # else:
                #     if verbose:
                #         print(" ERROR:")
                #         print(" ".join(args))
                #         print(p.stderr, flush=True)
                #     if os.path.exists(fpath_out):
                #         if verbose:
                #             print("Removing existing (possibly bad)", fname_out, flush=True)
                #         os.remove(fpath_out)

            # Clean up the remaining child processes.
            while len(running_procs) > 0:
                procs_to_remove = []
                fnames_to_remove = []
                tdirs_to_remove = []
                # Find processes that are done running, add them to procs to be removed.
                for proc, fname, tempdir in zip(running_procs, running_fnames, running_tdirs):
                    if not proc.is_alive():
                        # If a process has finished, label it to be removed from the queue
                        proc.join()
                        proc.close()
                        procs_to_remove.append(proc)
                        fnames_to_remove.append(fname)
                        tdirs_to_remove.append(tempdir)
                # Remove the process from the list of running processes.
                for proc, fname, tempdir in zip(procs_to_remove, fnames_to_remove, tdirs_to_remove):
                    files_finished += 1
                    if verbose:
                        print("{0}/{1}".format(files_finished, len(etopo_gdf)), fname)
                    running_procs.remove(proc)
                    running_fnames.remove(fname)
                    # Also clean up & remove the temporary directory.
                    if os.path.exists(tempdir):
                        tfns = [os.path.join(tempdir, tf) for tf in os.listdir(tempdir)]
                        for tf in tfns:
                            os.remove(tf)
                        os.rmdir(tempdir)
                    running_tdirs.remove(tempdir)

                time.sleep(0.01)

        except KeyboardInterrupt as e:
            # Clean up the running processes.
            for proc,fname in zip(running_procs, running_fnames):
                proc.terminate()
                proc.join()
                proc.close()
                if os.path.exists(fname):
                    os.remove(fname)
                raise e

        if overwrite:
            # Create a gdf, first delete the old one.
            gdf_fname = self.get_geopkg_object(verbose=verbose).get_gdf_filename(resolution_s=resolution_s)
            if os.path.exists(gdf_fname):
                os.remove(gdf_fname)
        return self.get_geodataframe(resolution_s=resolution_s, verbose=verbose)

        # For some reason above, the code doesn't always eliminate tempdirs that have been created.
        # At the end, try to clean up any missed ones.
        for tempdir in tempdirs_total:
            if os.path.exists(tempdir):
                os.rmdir(tempdir)

def generate_one_tile(args, fname, tempdir=None, verbose=True):
    """Subprocess for multiprocessing.Process above in main function. Run the waffles command."""
    # Change the child proceess to work within the tempdir given to it. Create it if necessary.
    if tempdir:
        if not os.path.exists(tempdir):
            os.mkdir(tempdir)
        os.chdir(tempdir)

    p = subprocess.run(args, capture_output=True)
    if p.returncode != 0 and verbose:
        print("process '{0}' completed with Error code {1}.".format(" ".join(args), p.returncode),
              ("{0} exists." if os.path.exists(fname) else "{0} not written.").format(fname))
    return


if __name__ == "__main__":
    lakes = source_dataset_global_lakes_flat()
    for res in (15,1):
        lakes.get_geodataframe(res)
        # lakes.create_global_lakes_flat_tiles(resolution_s = res,
        #                                      depth_source = lakes.config.flat_lake_depth,
        #                                      numprocs=15,
        #                                      overwrite=True,
        #                                      verbose=True)

