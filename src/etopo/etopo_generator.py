# -*- coding: utf-8 -*-

"""etopo_dataset.py -- parent code for managing, building and assessing/validating
   the ETOPO global topo-bathy DEM."""

import os
import imp
# import shutil
import shapely.geometry
import subprocess
import multiprocessing
import numpy
import time
import re
import shutil

try:
    import cudem
except ModuleNotFoundError:
    print("""Warning, CUDEM code does not appear to be installed in this python installation.
          Some ETOPO_Generator functionality may not work.""")

#####################################################
# Code snippet to import the base directory into
# PYTHONPATH to aid in importing from all the other
# modules in other subdirs.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
#####################################################

import generate_empty_grids
import utils.configfile
import utils.progress_bar
import utils.parallel_funcs
import datasets.dataset_geopackage as dataset_geopackage

class ETOPO_Generator:
    # Tile naming conventions. ETOPO_2022_v1_N15E175.tif, e.g.
    fname_template_tif = r"ETOPO_2022_v1_{0:d}s_{1:s}{2:02d}{3:s}{4:03d}.tif"
    fname_template_netcdf = r"ETOPO_2022_v1_{0:d}s_{1:s}{2:02d}{3:s}{4:03d}.nc"

    # The base directory of the project is two levels up. Retrieve the absolute path to it on this machine.
    # This file resides in [project_basedir]/src/etopo
    project_basedir = os.path.abspath(os.path.join(os.path.split(__file__)[0], "..", ".."))

    etopo_config = utils.configfile.config(os.path.join(project_basedir, "etopo_config.ini"), )

    # Note, still need to add resolution "1s" or "15s" directory.
    empty_tiles_directory = etopo_config.etopo_empty_tiles_directory
    # intermediate_tiles_directory = etopo_config.etopo_intermediate_tiles_directory # Same as above
    finished_tiles_directory = etopo_config.etopo_finished_tiles_directory # Same as above
    etopo_gpkg_1s  = etopo_config.etopo_tile_geopackage_1s
    etopo_gpkg_15s = etopo_config.etopo_tile_geopackage_15s

    def __init__(self):
        """Read the configuration file, get the info about our grid locations."""
        self.datasets_dict = None

    def create_empty_grids(self, resolution_s=(1,15), verbose=True):
        """Create empty grid-cells for the ETOPO data product.
        resolution_s can be 1, 15, or a list/tuple with both values (1,15).
        """
        assert resolution_s in (1,15,(1,15),(15,1),[1,15],[15,1])

        # If it's a single int, make it a list so that the next block (looping over the list) universally works.
        if type(resolution_s) == int:
            resolution_s = [resolution_s]

        for res in resolution_s:
            resdir = os.path.join(ETOPO_Generator.empty_tiles_directory, str(res) + "s")
            if not os.path.exists(resdir):
                print("Directory", resdir, "does not exist.")
                assert os.path.exists(os.path.split(resdir)[0])
                os.mkdir(resdir)

            # In this case, the "tile_width_deg" is the same as the "resolution_s".
            # The 1-deg tiles are 1s resolution, 15-deg tiles are 15s resolution.
            # Makes each tile a 3600x3600 file.
            generate_empty_grids.create_empty_tiles(resdir,
                                                    fname_template_tif=ETOPO_Generator.fname_template_tif,
                                                    fname_template_netcdf=None, # TODO: Fill this in once the Netcdf outputs are finalized.
                                                    tile_width_deg=res,
                                                    resolution_s=res,
                                                    also_write_geopackage=True,
                                                    ndv = ETOPO_Generator.etopo_config.etopo_ndv,
                                                    verbose=verbose)

    def create_etopo_geopackages(self, verbose=True):
        """Create geopackages for all the tiles of the ETOPO dataset, at 1s and 15s
        resolutions. SHould run the "create_empty_grids() routine first."""
        # TODO: This code is out-of-date and out-of-sync with the DatasetGeopackage class definition.
        # Change this function to use the ETOPO_Geopackage API instead.
        for res in ("1s", "15s"):
            gtifs_directory = os.path.join(ETOPO_Generator.empty_tiles_directory, res)
            gpkg_fname = ETOPO_Generator.etopo_gpkg_1s if (res=="1s") \
                         else ETOPO_Generator.etopo_gpkg_15s


            dataset_geopackage.DatasetGeopackage(gpkg_fname).create_dataset_geopackage(\
                                           dir_or_list_of_files = gtifs_directory,
                                           geopackage_to_write = gpkg_fname,
                                           recurse_directory = False,
                                           file_filter_regex=r"\.tif\Z",
                                           verbose=verbose)

    def generate_tile_source_dlists(self, source = "all",
                                          active_only=True,
                                          # resolution=1,
                                          verbose=True):
        """For each of the ETOPO dataset tiles, generate a DLIST of all the
        source datasets that overlap that tile, along with their relative ranking.

        This DLIST will be used with the CUDEM waffles command to create the final
        ETOPO dataset from all available sources over each tile.

        If active_only (default), use only datasets listed as "active" in their configfiles.

        Resolution can be 1 or 15 (seconds).
        """
        # Get the dictionary of source datasets
        datasets_dict = self.fetch_etopo_source_datasets(active_only = active_only,
                                                         return_type=dict,
                                                         verbose=verbose)

        if source.strip().lower() == "all":
            datasets_list = sorted(datasets_dict.keys())
        else:
            datasets_list = [source]

        for dset_name in datasets_list:
            print("-------", dset_name, ":")
            dset_obj = datasets_dict[dset_name]
            dset_obj.create_waffles_datalist(verbose=verbose)

    def generate_etopo_tile_datalist(self, resolution=1,
                                           etopo_tile_fname = None,
                                           active_sources_only = True,
                                           verbose=True):
        """For each ETOPO tile (or just the one given), produce a waffles datalist of
        all the source tiles that overlap it (and thus would be included with it),

        Resolution can be 1 or 15.

        Put the datalist in the etopo_config.etopo_datalist_directory directory, under the
        '1s' or '15s' subdir, respectively, depending upon the resolution.

        Provide etopo_tile_fname if we only want one file done. Otherwise, do them all
        (for that resolution).
        """
        # Retrieve the dataset_geopackage object for this ETOPO grid.
        etopo_geopkg_obj = dataset_geopackage.ETOPO_Geopackage(resolution)
        etopo_gdf = etopo_geopkg_obj.get_gdf(verbose=verbose)
        config_obj = self.etopo_config
        etopo_crs = etopo_gdf.crs

        # 0. Get the ETOPO datalist folder, in the "1s" or "15s" directory
        datalist_folder = os.path.join(config_obj._abspath(config_obj.etopo_datalist_directory),
                                       str(resolution) + "s")
        # 1. If etopo_tile_fname is being used, get the one row corresponding to that fname.
        # Also get the polygons for all the etopo tiles.
        if etopo_tile_fname == None:
            etopo_fnames = etopo_gdf['filename'].tolist()
            etopo_polygons = etopo_gdf['geometry']
        else:
            # If we only have one file, make it a one-item list for the 'for' loop.
            etopo_fnames = [etopo_tile_fname]
            etopo_polygons = etopo_gdf[etopo_gdf.filename == etopo_tile_fname]['geometry'].tolist()
            assert(len(etopo_polygons) == len(etopo_fnames))

        # 2. Get all the ETOPO source dataset objects.
        datasets_list = self.fetch_etopo_source_datasets(verbose=verbose, return_type=list)

        # 3. For each tile in the gdf, create a datalist name for that tile in the datalist folder.
        for etopo_fname, etopo_poly in zip(etopo_fnames, etopo_polygons):
            fname_base = os.path.splitext(os.path.split(etopo_fname)[1])[0]
            etopo_tile_datalist_fname = os.path.join(datalist_folder, fname_base + ".datalist")

            # 5. Get all the datalist entries from that dataset object. Use the function for that.
            datalist_lines = []

            for dset_obj in datasets_list:
                this_dlist_entries = dset_obj.generate_tile_datalist_entries(etopo_poly,
                                                                             polygon_crs = etopo_crs,
                                                                             verbose = verbose)
                if len(this_dlist_entries) > 0:
                    datalist_lines.extend(this_dlist_entries)

            # 6. Put all the datalist entries together, write as lines to the datalist.
            datalist_text = '\n'.join(datalist_lines)
            with open(etopo_tile_datalist_fname, 'w') as f:
                f.write(datalist_text)
                f.close()
                if verbose:
                    print(etopo_tile_datalist_fname, "written with", len(datalist_lines), "entries.")

        return
        # 7. Test out the waffles -M stacks command to see if it runs faster now.

    def flag_datalist_mismatches(self, resolution=1,
                                       verbose=True):
        """Right now some of the waffles commands are acting strangely, prioritizing
        lower-ranking datasets instead of taking just the higher-ranking ones. Detect
        those here and list them out.
        """
        # TODO: Finish
        pass

    def generate_all_etopo_tiles(self, resolution=1,
                                       numprocs=utils.parallel_funcs.physical_cpu_count(),
                                       overwrite=False,
                                       verbose=True):
        """Geenerate all of the ETOPO tiles at a given resolution."""
        # Get the ETOPO_geopackage object, with the datalist filenames in it
        # (if they're not in there already, they may be)
        etopo_gdf = dataset_geopackage.ETOPO_Geopackage(resolution).add_dlist_paths_to_gdf()

        # Sort the lists, just 'cuz.
        etopo_tiles  = etopo_gdf['filename'].to_numpy()
        sort_mask    = numpy.argsort(etopo_tiles)

        etopo_tiles  = numpy.array(etopo_tiles) [sort_mask]
        etopo_polys  = etopo_gdf['geometry'].to_numpy()[sort_mask]
        etopo_dlists = etopo_gdf['dlist'].to_numpy()[sort_mask]
        etopo_xres   = etopo_gdf['xres'].to_numpy()[sort_mask]
        etopo_yres   = etopo_gdf['yres'].to_numpy()[sort_mask]

        if verbose:
            print("Generating", len(etopo_tiles), "ETOPO tiles at", str(resolution) + "s resolution:")

        # Append new active processes to this queue until they're all done.
        active_procs = []
        active_tempdirs = []
        total_finished_procs = 0
        # N = 6
        N = len(etopo_tiles)
        waiting_procs = [None] * N
        temp_dirnames = [None] * N
        current_max_running_procs = numprocs # "max_running_procs" can change depending how many tiles are left.

        # First, generate a whole list of child processes waiting to be started.
        for i,(tile, poly, dlist, xres, yres) in enumerate(zip(etopo_tiles, etopo_polys, etopo_dlists, etopo_xres, etopo_yres)):

            # This typically does nothing, but handy if we're wanting to do only a small subset of the data and set N to an artificially low number.
            if i >= N:
                break

            dest_tile = tile.replace("empty_tiles", "finished_tiles")
            # If the destination tile already exists, just leave it and put a None in the process' place
            if (not overwrite) and os.path.exists(dest_tile):
                proc = None
                temp_dirname = None
            else:
                temp_dirname = os.path.join(self.etopo_config.project_base_directory, "scratch_data", "temp" + str(i))
                proc = multiprocessing.Process(target=generate_single_etopo_tile,
                                               args=(dest_tile,
                                                     poly,
                                                     dlist,
                                                     xres,
                                                     yres,
                                                     self.etopo_config.etopo_ndv,
                                                     self.etopo_config.etopo_cudem_cache_directory,
                                                     temp_dirname),
                                               kwargs = {'verbose': False})


            waiting_procs[i] = proc
            temp_dirnames[i] = temp_dirname

        # Tally up how many were already written to disk.
        waiting_procs = [proc for proc in waiting_procs if proc != None]
        temp_dirnames = [tdir for tdir in temp_dirnames if tdir != None]

        total_finished_procs = N - len(waiting_procs)

        try:

            while total_finished_procs < N:
                procs_waiting_to_be_removed = []
                tmpdirs_waiting_to_be_deleted = []
                for aproc,tmpdir in zip(active_procs, active_tempdirs):
                    if not aproc.is_alive():
                        aproc.join()
                        aproc.close()
                        procs_waiting_to_be_removed.append(aproc)
                        tmpdirs_waiting_to_be_deleted.append(tmpdir)
                        total_finished_procs += 1

                for dproc,tmpdir in zip(procs_waiting_to_be_removed,tmpdirs_waiting_to_be_deleted):
                    active_procs.remove(dproc)
                    os.rmdir(tmpdir)
                    active_tempdirs.remove(tmpdir)

                # Reset the maximum number of running procs to be either "nprocs" or
                # the number of procs we actually have left, whichever is smaller.
                current_max_running_procs = min(numprocs, (N - total_finished_procs))

                # Populate new processes
                while len(active_procs) < current_max_running_procs:
                    if len(waiting_procs) == 0:
                        # Shouldn't ever get here if I've done my math right, but
                        # in case there are not processing still waiting, just exit.
                        break
                    proc_to_start = waiting_procs.pop(0)
                    tmpdir = temp_dirnames.pop(0)
                    os.mkdir(tmpdir)
                    proc_to_start.start()
                    active_procs.append(proc_to_start)
                    active_tempdirs.append(tmpdir)

                if verbose:
                    utils.progress_bar.ProgressBar(total_finished_procs, N, suffix="{0:,}/{1:,}".format(total_finished_procs, N))

                time.sleep(0.01)

        except:
            # If things shut down or crash, clean up the open temp_dirs:
            for tmpdir in active_tempdirs:
                os.rmdir(tmpdir)

            # while len(active_procs) >= min(numprocs, N - total_finished_procs):
            #     finished_procs = []
            #     for running_proc in active_procs:
            #         if not running_proc.is_active():
            #             running_proc.join()
            #             # Add it to the list of finished procs to remove from the queue.
            #             finished_procs.append(running_proc)
            #             total_finished_procs += 1

            #     for f_proc in finished_procs:
            #         active_procs.remove(f_proc)

            # if overwrite or not os.path.exists(dest_tile):
            #     retval, stdout = generate_single_etopo_tile(dest_tile,
            #                                                 poly,
            #                                                 dlist,
            #                                                 xres,
            #                                                 yres,
            #                                                 self.etopo_config.etopo_ndv,
            #                                                 self.etopo_config.etopo_cudem_cache_directory,
            #                                                 verbose=False)
            # else:
            #     retval = 0

            # if verbose:
            #     if retval == 0:
            #         utils.progress_bar.ProgressBar(i+1, len(etopo_tiles), suffix="{0:,}/{1:,}".format(i+1, len(etopo_tiles)))
            #     else:
            #         print("\r\nERROR: waffles return code", retval)
            #         print(stdout)
            # # Get rid of this once I confirm it's working.
            # # break

        return

    def generate_etopo_source_master_dlist(self,
                                           active_only=True,
                                           verbose=True):
        """Create a master datalist that lists all the datasets from which ETOPO is derived.
        It will be a datalist of datalists from each of the etopo_source_dataset datalists."""
        datasets_dict = self.fetch_etopo_source_datasets(active_only = active_only,
                                                         return_type=dict,
                                                         verbose=verbose)

        dlist_lines = []
        DATALIST_DTYPE = -1 # The Waffles code for a datalist used recursively.
        for (dset_name, dset_obj) in datasets_dict.items():
            dset_dlist_fname = dset_obj.get_datalist_fname()
            dset_ranking_number = dset_obj.get_dataset_ranking_score()
            text_line = "{0} {1} {2}".format(dset_dlist_fname, DATALIST_DTYPE, dset_ranking_number)
            dlist_lines.append(text_line)

        dlist_text = "\n".join(dlist_lines)

        with open(self.etopo_config.etopo_sources_datalist, 'w') as f:
            f.write(dlist_text)
            f.close()
            if verbose:
                print(self.etopo_config.etopo_sources_datalist, "written.")

        return self.etopo_config.etopo_sources_datalist

    def fetch_etopo_source_datasets(self, active_only=True, verbose=True, return_type=dict):
        """Look through the /src/datasets/ directory, and get a list of the input datasets to use.
        This list will be saved to self.source_datasets, and will be instances of
        the datasets.source_dataset class, or sub-classes thereof.
        Each subclass must be called source_dataset_XXXX.py where XXX refers to the name
        of the dataset specified by the folder it's in. For instance, the CopernicusDEM
        is in the folder /src/datasets/CopernicusDEM/source_dataset_CopernicusDEM.py file.

        Any dataset that does not have a sub-class defined in such a file, will simply
        be instantiated with the parent class defined in source_dataset.py. Also if
        the dataset's 'is_active()' routeine is False, the dataset will be skipped.

        type_out can  be list, tuple, or dict. If dict, the keys will be the dataset names, the value the objects.
        For list and tuple, just a list of the objects is fine.
        """
        # The the list of dataset objects already exists, just return it.
        if self.datasets_dict != None:
            if return_type == dict:
                return self.datasets_dict
            else:
                return list(self.datasets_dict.values())

        # Get this directory.
        datasets_dir = os.path.join(ETOPO_Generator.project_basedir, 'src', 'datasets')
        assert os.path.exists(datasets_dir) and os.path.isdir(datasets_dir)

        # Get all the subdirs within this directory.
        subdirs_list = sorted([fn for fn in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, fn))])

        dataset_objects_dict = dict()
        # Loop through all the sub-directories, check requirements for an active dataset class.
        for dataset_name in subdirs_list:
            subdir = os.path.join(datasets_dir, dataset_name)
            dataset_class_name = "source_dataset_{0}".format(dataset_name)
            module_fname = os.path.join(subdir, dataset_class_name + ".py")

            if os.path.exists(module_fname):
                # Create the import path of the module, relative to the /src
                # directory (which is imported above)
                file, pathname, desc = imp.find_module(dataset_class_name, [subdir])
                module = imp.load_module(dataset_class_name, file, pathname, desc)

                # Create an object from the file.
                dataset_object = getattr(module, dataset_class_name)()
                if (not active_only) or dataset_object.is_active():
                    dataset_objects_dict[dataset_name] = dataset_object
                else:
                    # If the dataset is not currently active and we're only fetching the acive ones, just forget it.
                    continue

            # The the dataset doesn't have a class module, forget it.
            else:
                continue

        self.datasets_dict = dataset_objects_dict

        if verbose:
            print("Datasets active:", ", ".join([d for d in sorted(self.datasets_dict.keys())]))

        if return_type == dict:
            return self.datasets_dict
        else:
            return return_type(self.datasets_dict.values())

    def move_weights_to_subdir(self, resolution="both",
                                     weights_regex = r"_w\.tif\Z",
                                     verbose=True):
        """The waffles -M stacks module creates a "_w.tif" 'weights' file with each
        ETOPO data tile. Move these into the "weights" subdir in the same directory.
        """
        if resolution.lower() == "both":
            resolutions = (1,15)
        else:
            assert type(resolution) == int
            resolutions = (resolution,)

        for res in resolutions:
            tiles_dir = os.path.join(self.etopo_config.etopo_finished_tiles_directory, str(res) + "s")
            weights_dir = os.path.join(tiles_dir, "weights")

            assert os.path.exists(tiles_dir)
            if not os.path.exists(weights_dir):
                os.mkdir(weights_dir)

            weights_fnames_in_tiles_dir = [fn for fn in os.listdir(tiles_dir) if re.search(weights_regex, fn) != None]

            for fname in weights_fnames_in_tiles_dir:
                old_file = os.path.join(tiles_dir, fname)
                new_file = os.path.join(weights_dir, fname)
                shutil.move(old_file, new_file)

            if verbose:
                print(len(weights_fnames_in_tiles_dir), "files moved from", tiles_dir, "to", weights_dir)

    def copy_finished_tiles_to_onedrive(self):
        """I share the finished tiles with the team through my OneDrive directory.
        Copy the whole finished_tiles directory tree over there.

        If files already exist in OneDrive, overwrite them."""
        src_dir  = os.path.abspath(self.etopo_config.etopo_finished_tiles_directory)
        dest_dir = os.path.abspath(self.etopo_config.etopo_onedrive_directory)

        shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)


def generate_single_etopo_tile(dest_tile_fname,
                               dest_tile_polygon,
                               datalist_fname,
                               tile_xres,
                               tile_yres,
                               etopo_ndv,
                               etopo_cache_dir,
                               temp_dir_for_cwd,
                               verbose = True):
    """Use waffles to generate an ETOPO tiles from a datalist.

    Defined as a standalone process to more easilyi enable multi-processing."""

    # Get the bounds of the polygon (which we already know is a vertically-aligned square,
    # so I don't need to do anything special here.)
    dest_tile_bounds = shapely.geometry.MultiPoint(dest_tile_polygon.exterior.coords).bounds

    waffles_args = ["waffles",
                    "-M", "stacks:supercede=True:keep_weights=True",
                    "-R", "{0}/{1}/{2}/{3}".format(dest_tile_bounds[0], dest_tile_bounds[2], dest_tile_bounds[1], dest_tile_bounds[3]),
                    "-N", str(etopo_ndv),
                    "-E", "{0}/{1}".format(abs(tile_xres), abs(tile_yres)),
                    "-w", # Use the datalist weights.
                    # "-a", # Archive the datalist (why again?)
                    "-k", # Hold onto any cached files.
                    "-P", "EPSG:4326",
                    "-D", etopo_cache_dir,
                    "-O", os.path.splitext(dest_tile_fname)[0], # Get rid of the .tif at the end. Waffles adds that automatically.
                    datalist_fname]

    if verbose:
        print(" ".join(waffles_args))

    # Run it. Make the "cwd" a temp dir to avoid conflicts.
    proc = subprocess.run(waffles_args, cwd=temp_dir_for_cwd, capture_output=subprocess.PIPE, text=True)
    # proc = subprocess.run(waffles_args, cwd=os.path.dirname(dest_tile_fname), capture_output=subprocess.PIPE, text=True)
    if verbose:
        print(dest_tile_fname, "written.")

    return proc.returncode, proc.stdout

    # Sample (complete) waffles command.
    """waffles -M stacks:supercede=True:keep_weights=True
    -R -67.0000000/-66.0000000/18.0000000/19.0000000
    -E 0.000277777777778
    -N -99999
    -w
    -a
    -k
    -P EPSG:4326
    -D /home/mmacferrin/Research/DATA/ETOPO/scratch_data/
    -O /home/mmacferrin/Research/DATA/ETOPO/data/finished_tiles/1s/ETOPO_2022_v1_1s_N18W067
    /home/mmacferrin/Research/DATA/ETOPO/data/etopo_sources.datalist"""
    # ^ Except, use the source_tile_datalist created here for that entry.


if __name__ == "__main__":
    EG = ETOPO_Generator()
    # EG.generate_tile_source_dlists(source="all",active_only=True,verbose=True)
    # EG.generate_etopo_source_master_dlist()
    # EG.create_empty_grids()
    # EG.create_etopo_geopackages()
    # EG.create_intermediate_grids(source="FABDEM", resolution_s=1)
    # EG.generate_etopo_tile_datalist(resolution=1)
    # EG.generate_all_etopo_tiles(resolution=15, numprocs=15)
    EG.move_weights_to_subdir()
