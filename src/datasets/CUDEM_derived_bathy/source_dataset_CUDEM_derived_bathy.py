# -*- coding: utf-8 -*-

"""Source code for the CUDEM_derived_bathy ETOPO source dataset class."""

import os
import subprocess

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset
import utils.configfile
etopo_config = utils.configfile.config()
import utils.parallel_funcs

class source_dataset_CUDEM_derived_bathy(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "CUDEM_derived_bathy_config.ini" )):
        """Initialize the CUDEM_derived_bathy source dataset object."""

        super(source_dataset_CUDEM_derived_bathy, self).__init__("CUDEM_derived_bathy", configfile)

    def generate_tiles(self, resolution_s = 1, file_prefix="gebcop", large=False, max_nprocs=10, overwrite=False, verbose=True):
        """Create the tiles using the CUDEM script including GEBCO and MAR_Gravimetry."""
        tile_xmins, tile_xmaxs, tile_ymins, tile_ymaxs = self.get_tile_bounds(resolution_s=resolution_s, large=large)
        fnames_out = [None] * len(tile_xmins)

        for i, (xmin, ymin) in enumerate(zip(tile_xmins, tile_ymins)):
            fnames_out[i] = os.path.join(self.get_source_datafiles_dir(resolution_s=resolution_s),
                                         "{0}_{1}{2:02d}{3}{4:03d}.tif".format(file_prefix,
                                                                           "N" if ymin >= 0 else "S",
                                                                           abs(ymin),
                                                                           "E" if xmin >= 0 else "W",
                                                                           abs(xmin))
                                         )

        tempdirs = [os.path.join(etopo_config._abspath(etopo_config.etopo_cudem_cache_directory), "tempproc{0:04d}".format(i)) for i in range(len(fnames_out))]
        # If tempdirs are still in there, remove their contents before beginning this execution. This is useful if we'd
        # previously killed a process and temporary files are still in there.
        for tempdir in tempdirs:
            if os.path.exists(tempdir):
                rm_cmd = ["rm", "-rf", tempdir + "/*"]
                subprocess.run(rm_cmd, capture_output=True)

        args_lists = list(zip(fnames_out, tile_xmins, tile_xmaxs, tile_ymins, tile_ymaxs))
        kwargs_init = {"resolution_s": resolution_s,
                        "overwrite": overwrite,
                        "verbose": False if (len(fnames_out)>1 and max_nprocs>1) else verbose} # Only make it verbose if we're doing 1 file and verobse==True
        kwargs_list = [kwargs_init.copy() for i in range(len(args_lists))]

        # Include MAR gravity over the tiles where we want it.
        mar_grav_list = self.get_include_mar_grav_list(list(zip(tile_xmins, tile_xmaxs, tile_ymins, tile_ymaxs)))

        for i in range(len(kwargs_list)):
        # for kw, mar_grav in zip(kwargs_list, mar_grav_list):
            kwargs_list[i]["include_mar_grav"] = mar_grav_list[i]
            if mar_grav_list[i] is False:
                kwargs_list[i]["additional_options"] = ":pre_upper_limit=-1" # Also, for these two tiles, we want to set the upper limit
                                             # of bathy elevations to -1 m, rather than the default -0.1 m used previously.

        # print("\n".join([str(args) + " " + str(kwargs) for args, kwargs in zip(args_lists, kwargs_list)]))
        # foobar

        # for args, kwargs in zip(args_lists, kwargs_list):
        #     # print(args, kwargs)
        #     self.create_single_tile(*args, **kwargs)
        #
        # return

        utils.parallel_funcs.process_parallel(self.create_single_tile,
                                              args_lists,
                                              overwrite_outfiles=overwrite,
                                              kwargs_list=kwargs_list,
                                              outfiles=fnames_out,
                                              temp_working_dirs=tempdirs,
                                              max_nprocs=max_nprocs,
                                              verbose=verbose)

        return

    @staticmethod
    def get_include_mar_grav_list(tile_bounds_list):
        """For each tile, indicate whether we want to include MAR Gravity measurements in the bathy DEM."""
        mar_grav = [True] * len(tile_bounds_list)

        # Specifically for the two tiles near the shore in TX, omit the MAR Gravity readings because it gives shit data there.
        # Keep it in everywhere else.
        for i, tbounds in enumerate(tile_bounds_list):
            if tbounds[2] in (24,25) and tbounds[0] in (-98,):
                mar_grav[i] = False

        return mar_grav

    @staticmethod
    def get_tile_bounds(resolution_s=1, large=False):
        """Define the bounding boxes of the tiles we need to generate, at a given resolution."""
        if resolution_s != 1:
            raise NotImplementedError("Haven't yet gotten a list of non-1s tiles to generate.")

        assert resolution_s == 1

        tile_x_mins = []
        tile_y_mins = []
        tile_x_maxs = []
        tile_y_maxs = []

        # We've identified 3 areas (boxes) of 1-degree cells to generate. Get a list of bounding intervals here.
        # First box, Gulf of Mexico, off-shore
        if large:
            # Try doing just one really large box, to maybe avoid the seams?
            tile_x_mins.append(-98)
            tile_x_maxs.append(-96)
            tile_y_mins.append(24)
            tile_y_maxs.append(26)
        else:
            for lat in range(24, 25 + 1):
                for lon in range(-98, -97 + 1):
                    tile_x_mins.append(lon)
                    tile_x_maxs.append(lon + 1)
                    tile_y_mins.append(lat)
                    tile_y_maxs.append(lat + 1)

        # Second box, Bahamas
        if large:
            tile_x_mins.append(-80)
            tile_x_maxs.append(-76)
            tile_y_mins.append(24)
            tile_y_maxs.append(28)
        else:
            for lat in range(24, 27 + 1):
                for lon in range(-80, -77 + 1):
                    tile_x_mins.append(lon)
                    tile_x_maxs.append(lon + 1)
                    tile_y_mins.append(lat)
                    tile_y_maxs.append(lat + 1)

        # Third box, coastal Canada
        if large:
            tile_x_mins.append(-67)
            tile_x_maxs.append(-65)
            tile_y_mins.append(43)
            tile_y_maxs.append(46)
        else:
            for lat in range(43, 45 + 1):
                for lon in range(-67, -66 + 1):
                    tile_x_mins.append(lon)
                    tile_x_maxs.append(lon + 1)
                    tile_y_mins.append(lat)
                    tile_y_maxs.append(lat + 1)

        return tile_x_mins, tile_x_maxs, tile_y_mins, tile_y_maxs

    @staticmethod
    def create_single_tile(output_file,
                           xmin,
                           xmax,
                           ymin,
                           ymax,
                           resolution_s=1,
                           include_mar_grav=True,
                           additional_options="",
                           overwrite=False,
                           verbose=True):
        """Create a single tile using the waffles command with config options."""
        #  include_mar_grav=True, # omitted, not needed here.

        if os.path.exists(output_file):
            if overwrite:
                os.remove(output_file)
            else:
                return

        # Command for generating the JSON file.
        cmd_args = ["waffles",
                    "-R", "{0}/{1}/{2}/{3}".format(xmin, xmax, ymin, ymax),
                    "-E", "{0}s".format(resolution_s),
                    "-O", os.path.splitext(output_file)[0],
                    "-w", "-m",
                    "-k", "-D", etopo_config._abspath(etopo_config.etopo_cudem_cache_directory),
                    "-P", "epsg:{0}".format(4326),
                    "-M", "cudem:min_weight=.15:landmask=coastline:pre_count=1" + additional_options, # Can add "pre_upper_limit=-1" if we want to set upper limit from -0.1 to -1. Test this out and see how it does.
                    "gebco:exclude_tid=0/40,-11,1",
                    "fabdem,-11,2"]

        if include_mar_grav:
            cmd_args = cmd_args + ["mar_grav:raster=True:upper_limit=-0.1,-11,.01"]

        if verbose:
            print(" ".join(cmd_args))

        subprocess.run(cmd_args, capture_output=not verbose)

        return

if __name__ == "__main__":
    cdm = source_dataset_CUDEM_derived_bathy()
    gdf = cdm.get_geodataframe(resolution_s=1)
    # cdm.create_single_tile("/home/mmacferrin/Research/DATA/DEMs/CUDEM_derived_bathy/1s/gebcop_N24W098.tif",
    #                        -98, -97, 24, 25,
    #                        resolution_s=1,
    #                        overwrite=True,
    #                        include_mar_grav=False,
    #                        additional_options=":pre_upper_limit=-1",
    #                        verbose=True)
    # cdm.generate_tiles(file_prefix="gebcop_large", resolution_s=1, max_nprocs=1, verbose=True, large=True, overwrite=False)
    # cdm.generate_tiles(file_prefix="gebcop", resolution_s=1, max_nprocs=1, verbose=True, large=False, overwrite=False)
