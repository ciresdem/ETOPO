# -*- coding: utf-8 -*-

"""Source code for the GMRT ETOPO source dataset class."""

import os
import numpy
import subprocess

#####################
# Don't actually need the CUDEM package here, but it's an indicator whether CUDEM is installed on this machine.
# If not, toss an error (we need it).
try:
    import cudem
except ImportError as e:
    import sys
    print('CUDEM must be installed on this machine.', file=sys.stderr)
    raise e
#####################

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset
import datasets.dataset_geopackage as dataset_geopackage

class source_dataset_GMRT(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "GMRT_config.ini" )):
        """Initialize the GMRT source dataset object."""

        super(source_dataset_GMRT, self).__init__("GMRT", configfile)

    def fetch_tiles(self, resolution_s=15, crm_only_if_1s=True, verbose=True):
        """Using the CUDEM 'fetches' module, create all the tiles needed for this dataset."""
        etopo_gpkg = dataset_geopackage.ETOPO_Geopackage(resolution = resolution_s)
        etopo_gdf = etopo_gpkg.get_gdf(crm_only_if_1s=crm_only_if_1s,
                                       resolution_s=resolution_s,
                                       verbose=verbose).copy()

        # Loop through all the ETOPO files and create an identical tile in this dataset.
        for i, row in etopo_gdf.iterrows():
            xleft = row.xleft
            xright = numpy.round(xleft + (row.xsize*row.xres))
            ytop = row.ytop
            ybottom = numpy.round(ytop + (row.ysize*row.yres))

            gmrt_tile_fname = os.path.join(self.config._abspath(self.config.source_datafiles_directory),
                                           self.config.datafiles_name_template.format(resolution_s,
                                                                                      "N" if ybottom >= 0 else "S",
                                                                                      abs(int(numpy.round(ybottom))),
                                                                                      "E" if xleft >= 0 else "W",
                                                                                      abs(int(numpy.round(xleft)))))

            fetches_command = ["waffles", "-M", "stacks",
                               "-w",
                               "-R", "{0}/{1}/{2}/{3}".format(xleft,xright,ybottom,ytop),
                               "-E", "{0}s".format(resolution_s),
                               "--t_srs", "EPSG:4326",
                               "-D", etopo_gpkg.config.etopo_cudem_cache_directory,
                               "-k",
                               "-O", os.path.splitext(gmrt_tile_fname)[0],
                               "-f",
                               "-F", "GTiff",
                               "gmrt:layer=topo-mask"]

            # print(" ".join(fetches_command))

            p = subprocess.run(fetches_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            # p = subprocess.run(fetches_command)
            if (not ((p.returncode == 0) or os.path.exists(gmrt_tile_fname))) and verbose:
                print("ERROR: '{0}' sent return code {1}:".format(" ".join(fetches_command), p.returncode))
                print(p.stdout)

            if verbose:
                print("{0}/{1} {2} {3}written.".format(i+1, len(etopo_gdf), os.path.split(gmrt_tile_fname)[1], "" if os.path.exists(gmrt_tile_fname) else "NOT "))

        return


if __name__ == "__main__":
    gmrt = source_dataset_GMRT()
    gmrt.fetch_tiles(resolution_s=15)
    gmrt.fetch_tiles(resolution_s=1)
