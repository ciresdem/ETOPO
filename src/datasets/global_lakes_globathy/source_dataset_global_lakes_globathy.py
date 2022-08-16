# -*- coding: utf-8 -*-

"""Source code for the global_lakes_globathy ETOPO source dataset class."""

import os
import subprocess
from osgeo import gdal

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset
import datasets.dataset_geopackage
import utils.configfile
etopo_config = utils.configfile.config()

class source_dataset_global_lakes_globathy(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "global_lakes_globathy_config.ini" )):
        """Initialize the global_lakes_globathy source dataset object."""

        super(source_dataset_global_lakes_globathy, self).__init__("global_lakes_globathy", configfile)

    def create_global_lakes_globathy_tiles(self,
                                           resolution_s = 15,
                                           crm_only_if_1s = True,
                                           overwrite = False,
                                           verbose = True):

        resolution_s = int(resolution_s)
        assert resolution_s in (1,15)

        # 1) Get the tile outlines and resolutions of each grid to create. (Use the etopo empty tiles to do this.)
        etopo_gpkg = datasets.dataset_geopackage.ETOPO_Geopackage(resolution_s)
        gdf = etopo_gpkg.get_gdf(crm_only_if_1s = crm_only_if_1s,
                                 verbose = verbose)

        # 2) Get the directory of the output files and filenames for each output
        data_dir = self.config._abspath(self.config.source_datafiles_directory_1s if (resolution_s == 1) else self.config.source_datafiles_directory_15s)

        # print(data_dir)
        # print(gdf)
        # print(gdf.columns)
        # return gdf

        # 3) Generate all the output grids. Lakes everywhere!

        for i,row in enumerate(gdf.iterrows()):
            # print(i)
            # print(type(row))
            # for j in range(len(row)):
            #     print(j, type(row[j]), row[j])
            # # print(row)
            # foobar
            row = row[1]
            # NOTE: This ONLY works because in ETOPO's case, the resolution in seconds (1,15) is equal to the tile-size in degrees (1/15)
            xright = int(row.xleft + resolution_s)
            ybottom = int(row.ytop - resolution_s)
            # Generate the file name and the path it'll be written to.
            fname_out = self.config.global_lakes_fname_template.format("N" if ybottom >= 0 else "S",
                                                                       int(abs(ybottom)),
                                                                       "E" if row.xleft >= 0 else "W",
                                                                       int(abs(row.xleft)))
            fpath_out = os.path.join(data_dir, fname_out)

            # If the file already exists and we're not overwriting, just skip & move along.
            if not overwrite and os.path.exists(fpath_out):
                continue

            # Waffles lakes command looks like:
            # waffles -M lakes -R -86/-85/30/31 -E 1s -P epsg:4326 -D ~/Research/DATA/ETOPO/scratch_data/ -k -O lakes_test_N30W086
            # Create the waffles command arguments there.
            args = ["waffles",
                    "-M", "lakes",
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
            if verbose:
                print("{0}/{1}".format(i+1, len(gdf)),
                      fname_out + "...", end="" , flush=True)
            # print("")
            # continue

            # print("\n" + " ".join(args))
            # return

            try:
                # Run the lakes module command.
                p = subprocess.run(args,
                                   encoding="utf-8",
                                   stdin=None,
                                   stdout=None,
                                   stderr=subprocess.PIPE)
            except KeyboardInterrupt as e:
                # If the process gets interrupted, delete any partially-written file.
                if os.path.exists(fpath_out):
                    os.remove(fpath_out)
                raise e

            if p.returncode == 0:
                # Write out statistics to the file if they are needed. I *think* this does it.
                assert os.path.exists(fpath_out)
                if verbose:
                    print("Setting stats...", end='')
                ds = gdal.Open(fpath_out)
                band = ds.GetRasterBand(1)
                band.ComputeStatistics(0)
                band = None
                ds = None

                if verbose:
                    print(" Done.", flush=True)

            else:
                if verbose:
                    print(" ERROR:")
                    print(" ".join(args))
                    print(p.stderr, flush=True)
                if os.path.exists(fpath_out):
                    if verbose:
                        print("Removing existing (possibly bad)", fname_out, flush=True)
                    os.remove(fpath_out)

        return gdf

    # def (resolution_s=15)

if __name__ == "__main__":
    sdt = source_dataset_global_lakes_globathy()

    gdf = sdt.create_global_lakes_globathy_tiles(resolution_s = 15, verbose=True)
