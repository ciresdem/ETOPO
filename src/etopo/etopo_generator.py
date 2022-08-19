# -*- coding: utf-8 -*-

"""etopo_dataset.py -- parent code for managing, building and assessing/validating
   the ETOPO global topo-bathy DEM."""

import os
import imp # TODO: Use an implementation using importlib (imp has been deprecated)
import shapely.geometry
import subprocess
import multiprocessing
import numpy
import time
import re
import shutil
import pathlib
import sys
import pandas
import datetime

try:
    import cudem
    cudem
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

import etopo.generate_empty_grids
import utils.configfile
import utils.progress_bar
import utils.parallel_funcs
import utils.traverse_directory
import datasets.dataset_geopackage as dataset_geopackage
import datasets.etopo_source_dataset

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
    etopo_gpkg_60s = etopo_config.etopo_tile_geopackage_60s

    def __init__(self):
        """Read the configuration file, get the info about our grid locations."""
        self.datasets_dict = None

    def create_empty_grids(self, resolution_s=None, verbose=True):
        """Create empty grid-cells for the ETOPO data product.
        resolution_s can be 1, 15, or a list/tuple with both values (1,15).
        """
        if resolution_s is None:
            resolution_s = (1,15,60)
        elif type(resolution_s) in (list,tuple):
            assert numpy.all([(type(r) == int) and (r in (1,15,60)) for r in resolution_s])
        elif type(resolution_s) == int:
            assert resolution_s in (1,15,60)
            # If it's a single integer, put it in an iterable tuple to iterate over.
            resolution_s = (resolution_s,)
        else:
            raise TypeError("Uknown type for 'resolution' in create_empty_grids(): " + str(type(resolution_s)) + " " + str(resolution_s))

        for res in resolution_s:
            resdir = os.path.join(ETOPO_Generator.empty_tiles_directory, str(res) + "s")
            if not os.path.exists(resdir):
                print("Directory", resdir, "does not exist.")
                assert os.path.exists(os.path.split(resdir)[0])
                os.mkdir(resdir)

            # In this case, the "tile_width_deg" is the same as the "resolution_s".
            # The 1-deg tiles are 1s resolution, 15-deg tiles are 15s resolution.
            # Makes each tile a 3600x3600 file.
            etopo.generate_empty_grids.create_empty_tiles(resdir,
                                                    fname_template_tif=ETOPO_Generator.fname_template_tif,
                                                    fname_template_netcdf=None, # TODO: Fill this in once the Netcdf outputs are finalized.
                                                    tile_width_deg=(360 if (res==60) else res),
                                                    resolution_s=res,
                                                    also_write_geopackage=True,
                                                    ndv = ETOPO_Generator.etopo_config.etopo_ndv,
                                                    verbose=verbose)

    # def create_etopo_geopackages(self, verbose=True):
    #     """Create geopackages for all the tiles of the ETOPO dataset, at 1s and 15s
    #     resolutions. SHould run the "create_empty_grids() routine first."""
    #     # TODO: This code is out-of-date and out-of-sync with the DatasetGeopackage class definition.
    #     # Change this function to use the ETOPO_Geopackage API instead.
    #     for res in ("1s", "15s"):
    #         gtifs_directory = os.path.join(ETOPO_Generator.empty_tiles_directory, res)
    #         gpkg_fname = ETOPO_Generator.etopo_gpkg_1s if (res=="1s") \
    #                      else ETOPO_Generator.etopo_gpkg_15s


    #         dataset_geopackage.DatasetGeopackage(gpkg_fname).create_dataset_geopackage(\
    #                                        dir_or_list_of_files = gtifs_directory,
    #                                        geopackage_to_write = gpkg_fname,
    #                                        recurse_directory = False,
    #                                        file_filter_regex=r"\.tif\Z",
    #                                        verbose=verbose)

    # def generate_tile_source_dlists(self, source = "all",
    #                                       active_only=True,
    #                                       # resolution=1,
    #                                       verbose=True):
    #     """For each of the ETOPO dataset tiles, generate a DLIST of all the
    #     source datasets that overlap that tile, along with their relative ranking.

    #     This DLIST will be used with the CUDEM waffles command to create the final
    #     ETOPO dataset from all available sources over each tile.

    #     If active_only (default), use only datasets listed as "active" in their configfiles.

    #     Resolution can be 1 or 15 (seconds).
    #     """
    #     # Get the dictionary of source datasets
    #     datasets_dict = self.fetch_etopo_source_datasets(active_only = active_only,
    #                                                      return_type=dict,
    #                                                      verbose=verbose)

    #     if source.strip().lower() == "all":
    #         datasets_list = sorted(datasets_dict.keys())
    #     else:
    #         datasets_list = [source]

    #     for dset_name in datasets_list:
    #         print("-------", dset_name, ":")
    #         dset_obj = datasets_dict[dset_name]
    #         dset_obj.create_waffles_datalist(verbose=verbose)

    def replace_config_value(self, all_txt, configfilename, fieldname, replacement_value):
        """Quick little helper function to replace the text of a value with another value in a configfile text string."""
        # Find the fieldname with either whitespace, newline, or \A (start of string) in front of it.
        # First find the fieldname

        try:
            fieldname_span = re.search(r"\s" + fieldname.strip(), all_txt).span()
        except AttributeError:
            try:
                fieldname_span = re.search(r"\A" + fieldname.strip(), all_txt).span()
            except AttributeError:
                raise AttributeError("Configfile '{0}' has no attribute '{1}'.".format(os.path.split(configfilename)[1], fieldname))

        txt_after_fieldname = all_txt[fieldname_span[1]:]
        # Find the equal sign surrounded by any whitespace.
        equal_span = re.search("(\s*)=(\s*)", txt_after_fieldname).span()
        # This span should start at 0. Check that.
        assert equal_span[0] == 0
        txt_after_equal = txt_after_fieldname[equal_span[1]:]

        # Get the right pattern for the dtype involved, for int, float, str
        # NOTE: This assumes the input value is the same type as the value replacing it.
        # This may break if that isn't true, but it should be true in this case.
        if type(replacement_value) == int:
            pstr = "\A(\-?)(\d+)"
        elif type(replacement_value) == float:
            # if numpy.isnan(replacement_value):
                # If it's a Nan value, could just be an empty string. Treat it as such.
            pstr = "\A((\-?)((\d+\.\d*)|(\d*\.\d+))|(\w+)|((?P<quote>[\'\"]).*?(?P=quote)))"
            # Floating points must have a decimal place, with integers either preceding it or after it.
            # pstr = "\A((\d+\.\d*)|(\d*\.\d+))"
        elif type(replacement_value) in (str, type(None)):
            # Look for either single words alone or groups of words surrounded by a single-or-double quote.
            # Same if we're replacing it with None, we'll just do an empty string.
            pstr = r'\A((\w+)|((?P<quote>[\'\"]).*?(?P=quote)))'
        elif type(replacement_value) == bool:
            # Look for "True" or "False"
            pstr = r'((True)|(False))'
        else:
            raise TypeError("Uknown type of value for '{0}': {1}".format(replacement_value, type(replacement_value)))

        try:
            value_span = re.search(pstr, txt_after_equal, flags=re.MULTILINE).span()
        except AttributeError:
            raise ValueError('Regex call:\n\tre.search(r"{0}", r\'{1}\') returned None.'.format(
                pstr, txt_after_equal))

        if type(replacement_value) == str:
            # If this string has whitespace in it, surround it with quotes if it isn't already.
            if re.search("\s", replacement_value) != None:
                if replacement_value[0] not in ('"', "'"):
                    replacement_value = '"' + replacement_value
                if replacement_value[-1] != replacement_value[0]:
                    replacement_value = replacement_value + replacement_value[0]
        elif type(replacement_value) == type(None):
            replacement_value = '""'
        elif type(replacement_value) == float and numpy.isnan(replacement_value):
            replacement_value = '""'

        # If the existing value is the same as what we're replacing it with, just return the original text unchanged.
        if txt_after_equal[value_span[0]:value_span[1]] == str(replacement_value):
            return all_txt

        # Return the entire string before the previous value,
        # the string of the replacement value,
        # and all the text after the previous value.
        return all_txt[:(fieldname_span[1] + equal_span[1])] + \
            str(replacement_value) + \
            all_txt[(fieldname_span[1] + equal_span[1] + value_span[1]):]


    def write_ranks_and_ids_from_csv(self, save_old = True, verbose=True):
        """Write out any changes to the "ranks_and_ids" csv back to the respective
        config.ini files of the objects. This speeds up editing significantly.

        If 'save_old', then save the old configfile to an _old.ini file to preserve it."""
        csv_name = self.etopo_config._abspath(self.etopo_config.etopo_dataset_ranks_and_ids_csv)
        if not os.path.exists(csv_name):
            raise FileNotFoundError(csv_name + " does not exist.")

        df = pandas.read_csv(csv_name, index_col=False)

        # a small dictionary for translating CSV field names to config.ini field names.
        fieldnames_lookup = {"name_long"    : "dataset_name",
                             "ranking_score": "default_ranking_score",
                             "id_number"   : "dataset_id_number",
                             "vdatum_name"  : "dataset_vdatum_name",
                             "vdatum_epsg"  : "dataset_vdatum_epsg",
                             "is_active"    : "is_active"}

        for _, row in df.iterrows():
            dset_obj = datasets.etopo_source_dataset.get_source_dataset_object(row["dset_name"])
            configfile = dset_obj.configfilename

            if save_old:
                base, ext = os.path.splitext(configfile)
                old_configfile = base + "_old" + ext
                # Copy the current configfile over to the "_old" version. Overwrite the 'old' old if it already exists.
                shutil.copy(configfile, old_configfile)
                if verbose:
                    print(os.path.split(old_configfile)[1], "written.")

            with open(configfile, 'r') as f:
                c_txt = f.read()

            for fieldname in row.keys():
                # Skip the name of the dataset, which doesn't appear in the configfile.
                if fieldname == "dset_name":
                    continue
                # Replace the text of the configfile with the value given in the CSV.
                fieldname_config = fieldnames_lookup[fieldname]
                c_txt = self.replace_config_value(c_txt, configfile, fieldname_config, row[fieldname])

            with open(configfile, 'w') as f:
                f.write(c_txt)
                if verbose:
                    print(os.path.split(configfile)[1], "written.")

        return

    def export_ranks_and_ids_csv(self, verbose=True):
        """This is not used by the code, but output a list of all the ETOPO datasets,
        with their ranks and ID numbers, and "is_active" status, and whether or not the
        datalist for it exists yet. This is useful for tweaking all the ranks
        and id #'s in the respective config.ini files and avoiding conflicts between
        datasets."""
        dsets_dict = self.fetch_etopo_source_datasets(active_only=False, verbose=verbose, return_type=dict)

        dset_names = []
        dset_names_long = []
        dset_ranks = []
        dset_ids = []
        dset_vdatum_names = []
        dset_vdatum_epsgs = []
        dset_is_actives = []

        for dset_name in sorted(dsets_dict.keys()):
            config = dsets_dict[dset_name].config

            try:
                dset_names.append(dset_name)
                dset_names_long.append(config.dataset_name)
                dset_ranks.append(config.default_ranking_score)
                dset_ids.append(config.dataset_id_number)
                dset_vdatum_names.append(config.dataset_vdatum_name)
                dset_vdatum_epsgs.append(config.dataset_vdatum_epsg)
                dset_is_actives.append(config.is_active)
            except AttributeError as e:
                print("ERROR: Dataset {0} doesn't have a '{1}' attribute in {2}".format(
                       dset_name,
                       re.search("(?<=has no attribute \')\w+(?=\')", str(e)).group(),
                       config._configfile),
                            file=sys.stderr)
                raise e

        df = pandas.DataFrame(data = {"dset_name": dset_names,
                                      "name_long": dset_names_long,
                                      "ranking_score": dset_ranks,
                                      "id_number": dset_ids,
                                      "vdatum_name": dset_vdatum_names,
                                      "vdatum_epsg": dset_vdatum_epsgs,
                                      "is_active": dset_is_actives})

        csv_name = self.etopo_config._abspath(self.etopo_config.etopo_dataset_ranks_and_ids_csv)
        df.to_csv(csv_name, index=False)
        if verbose:
            print(os.path.split(csv_name)[1], "written with {0} entries.".format(len(df)))

        return df

    def generate_etopo_tile_datalist(self, resolution=1,
                                           etopo_tile_fname = None,
                                           active_sources_only = True,
                                           verbose=True):
        """For each ETOPO tile (or just the one given), produce a waffles datalist of
        all the source tiles that overlap it (and thus would be included with it),

        Resolution can be 1 or 15, or 60 for the whole world.

        'active_sources_only' only applies to 1 or 15s tiles. For the 60s global tile,
        it just uses the ETOPO 15s tiles.

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

        if resolution == 60:
            # If the resolution is 60, just use all the ETOPO 15s tiles to generate the 60s
            # global tile.
            etopo_15s_gpkg = dataset_geopackage.ETOPO_Geopackage(15)
            etopo_15s_gdf = etopo_15s_gpkg.get_gdf(verbose=verbose)

            # We want to query the finished tiles of the dataset, not the empty tiles.
            dlist_filenames = [fn.replace("empty_tiles", "finished_tiles") for fn in etopo_15s_gdf['filename'].tolist()]
            assert numpy.all([os.path.exists(fn) for fn in dlist_filenames])
            datalist_lines = ["{0} 200 1".format(fn) for fn in dlist_filenames]
            datalist_text = "\n".join(datalist_lines)

            etopo_60s_fname = etopo_gdf['filename'].tolist()[0]
            fname_base = os.path.splitext(os.path.split(etopo_60s_fname)[1])[0]
            etopo_60s_datalist_fname = os.path.join(datalist_folder, fname_base + ".datalist")

            with open(etopo_60s_datalist_fname, 'w') as f:
                f.write(datalist_text)
                f.close()
                if verbose:
                    print(etopo_60s_datalist_fname, "written with {0} lines.".format(len(datalist_lines)))

        else:
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
            N = len(etopo_fnames)
            for i,(etopo_fname, etopo_poly) in enumerate(zip(etopo_fnames, etopo_polygons)):
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
                        print("\r" + " "*120, end="")
                        print(etopo_tile_datalist_fname, "written with", len(datalist_lines), "entries.")

                # Get rid of any previous .inf or .json files with the same name.
                # These are auto-generated by CUDEM and need to be rebuilt.
                inf_fname = etopo_tile_datalist_fname + ".inf"
                json_fname = etopo_tile_datalist_fname + ".json"
                if os.path.exists(inf_fname):
                    os.remove(inf_fname)
                if os.path.exists(json_fname):
                    os.remove(json_fname)

                if verbose:
                    utils.progress_bar.ProgressBar(i+1, N, suffix = "{0}/{1}".format(i+1,N))

        return
        # 7. Test out the waffles -M stacks command to see if it runs faster now.

    # def flag_datalist_mismatches(self, resolution=1,
    #                                    verbose=True):
    #     """Right now some of the waffles commands are acting strangely, prioritizing
    #     lower-ranking datasets instead of taking just the higher-ranking ones. Detect
    #     those here and list them out.
    #     """
    #     # TODO: Finish
    #     pass


    def add_datestamp_to_filename(self, tilename):
        """Add a _YYYY.MM.DD stamp to the end of the ETOPO filename.

        Useful when debugging the ETOPO tiles."""
        base, ext = os.path.splitext(tilename)
        nowstr = datetime.datetime.now().strftime("%Y.%m.%d")
        assert len(nowstr) == 10
        return base + "_" + nowstr + ext

    def generate_all_etopo_tiles(self, resolution=1,
                                       numprocs=utils.parallel_funcs.physical_cpu_count(),
                                       add_datestamp_to_files = False,
                                       overwrite=False,
                                       verbose=True):
        """Geenerate all of the ETOPO tiles at a given resolution."""
        # Get the ETOPO_geopackage object, with the datalist filenames in it
        # (if they're not in there already, they may be)
        etopo_gdf = dataset_geopackage.ETOPO_Geopackage(resolution).add_dlist_paths_to_gdf()

        # Sort the lists, just 'cuz.
        etopo_tiles  = etopo_gdf['filename'].to_numpy()
        sort_mask    = numpy.argsort(etopo_tiles)

        etopo_tiles  = etopo_tiles[sort_mask]
        etopo_polys  = etopo_gdf['geometry'].to_numpy()[sort_mask]
        etopo_dlists = etopo_gdf['dlist'].to_numpy()[sort_mask]
        etopo_xres   = etopo_gdf['xres'].to_numpy()[sort_mask]
        etopo_yres   = etopo_gdf['yres'].to_numpy()[sort_mask]

        # Look through all the dlists, (re-)generate if necessary.
        if verbose:
            print("Generating tile datalists:")
        self.generate_etopo_tile_datalist(resolution=resolution,
                                          etopo_tile_fname=None,
                                          active_sources_only=True,
                                          verbose=verbose)

        if verbose:
            print("Generating", len(etopo_tiles), "ETOPO tiles at", str(resolution) + "s resolution:")

        # Append new active processes to this queue until they're all done.
        active_procs = []
        active_tempdirs = []
        total_finished_procs = 0
        N = 6
        # N = len(etopo_tiles)
        waiting_procs = [None] * N
        temp_dirnames = [None] * N
        current_max_running_procs = numprocs # "max_running_procs" can change depending how many tiles are left.


        # First, generate a whole list of child processes waiting to be started.
        for i,(tile, poly, dlist, xres, yres) in enumerate(zip(etopo_tiles, etopo_polys, etopo_dlists, etopo_xres, etopo_yres)):

            # This typically does nothing, but handy if we're wanting to do only a small subset of the data and set N to an artificially low number.
            if i >= N:
                break

            dest_tile = tile.replace("empty_tiles", "finished_tiles")
            # If we're debugging, add the datestamp to the end of the filename.
            if add_datestamp_to_files:
                dest_tile = self.add_datestamp_to_filename(dest_tile)

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
                                               kwargs = {'verbose': False,
                                                         'algorithm': 'bilinear' if resolution==1 else 'average'})


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
                    # pass
                    utils.progress_bar.ProgressBar(total_finished_procs, N, suffix="{0:,}/{1:,}".format(total_finished_procs, N))

                time.sleep(0.01)

            # Move all the generated _w.tif "weights" files into the "weights" subdir.
            self.move_weights_to_subdir(resolution=resolution, verbose=verbose)

        except:
            # If things shut down or crash, clean up the open temp_dirs:
            for tmpdir in active_tempdirs:
                os.rmdir(tmpdir)

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
            print("Datasets ", "(Active) " if active_only else "", ":" + ", ".join([d for d in sorted(self.datasets_dict.keys())]))

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
        if type(resolution) == str and resolution.lower() == "both":
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

    def copy_finished_tiles_to_onedrive(self, resolution=None,
                                              use_symlinks = True,
                                              tile_regex = "\.tif\Z",
                                              also_synchronize = False,
                                              verbose=True):
        """I share the finished tiles with the team through my OneDrive directory.
        Copy the whole finished_tiles directory tree over there.

        If files already exist in OneDrive, overwrite them."""
        src_dir  = os.path.abspath(self.etopo_config.etopo_finished_tiles_directory)
        dest_dir = os.path.abspath(self.etopo_config.etopo_onedrive_directory)

        if resolution != None:
            src_dir = os.path.join(src_dir, str(resolution) + "s")
            dest_dir = os.path.join(dest_dir, str(resolution) + "s")

        if use_symlinks:
            # Just create symlinks so that we don't actually have to copy over all the data.
            # This returns relative paths to the files, which is perfect.
            list_of_matching_files = utils.traverse_directory.list_files(src_dir,
                                                                         regex_match=tile_regex,
                                                                         include_base_directory = False)


            # print(len(list_of_matching_files), list_of_matching_files[-3:])
            if verbose:
                print("Creating", len(list_of_matching_files), "links from", src_dir, "to", dest_dir, "...", end="")
            for fname in list_of_matching_files:
                src_f = os.path.join(src_dir, fname)
                dst_f = os.path.join(dest_dir, fname)
                # Remove the original file if it already exists.
                if os.path.exists(dst_f):
                    os.remove(dst_f)
                # Create a link to the source file.
                pathlib.Path(dst_f).symlink_to(src_f)
            if verbose:
                print("Done.")

        else:
            if verbose:
                print("Copying", src_dir, "-->", dest_dir)

            shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)

        if also_synchronize:
            self.upload_to_onedrive(resolution=resolution)

        return

    def upload_to_onedrive(self, resolution=None):
        """Take the files generated and put into our onedrive directory, and synchronize them up to the web.

        This may take a while.
        Should call after calling "copy_finished_tiles_to_onedrive()".
        """

        # If we also want to synchronize everything to OneDrive online, initiate the command here.
        onedrive_args = ["onedrive", "--synchronize", "--upload_only", "--single_directory"]
        if resolution is None:
            dirname = "ETOPO_Release"
        elif int(resolution) == 1:
            dirname = "ETOPO_Release/1s"
        elif int(resolution) == 15:
            dirname = "ETOPO_Release/15s"
        else:
            raise ValueError("Unknown resolution in ETOPO_Generator::upload_to_onedrive():", resolution)

        subprocess.run(onedrive_args + [dirname])

        return


def generate_single_etopo_tile(dest_tile_fname,
                               dest_tile_polygon,
                               datalist_fname,
                               tile_xres,
                               tile_yres,
                               etopo_ndv,
                               etopo_cache_dir,
                               temp_dir_for_cwd,
                               algorithm = "bilinear",
                               verbose = True):
    """Use waffles to generate an ETOPO tiles from a datalist.

    Defined as a standalone process to more easily enable multi-processing."""

    # Get the bounds of the polygon (which we already know is a vertically-aligned square,
    # so I don't need to do anything special here.)
    dest_tile_bounds = shapely.geometry.MultiPoint(dest_tile_polygon.exterior.coords).bounds

    # NOTE: The datalist must be present (already generated) for this.
    # For effiency sake we don't pass the etopo generator object to this method to create it.
    # Just assume it's already there. It's up to the calling function to create it.
    if not os.path.exists(datalist_fname):
        raise FileNotFoundError(datalist_fname + " not found.")

    waffles_args = ["waffles",
                    "-M", "stacks:supercede=True:keep_weights=True",
                    "-R", "{0}/{1}/{2}/{3}".format(dest_tile_bounds[0], dest_tile_bounds[2], dest_tile_bounds[1], dest_tile_bounds[3]),
                    "-N", str(etopo_ndv),
                    "-E", "{0}/{1}".format(abs(tile_xres), abs(tile_yres)),
                    "-w", # Use the datalist weights.
                    # "-a", # Archive the datalist (why again?)
                    "-k", # Hold onto any cached files.
                    "-P", "EPSG:4326",
                    "-S", algorithm,
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
    -k
    -P EPSG:4326
    -S bilinear
    -D /home/mmacferrin/Research/DATA/ETOPO/scratch_data/
    -O /home/mmacferrin/Research/DATA/ETOPO/data/finished_tiles/1s/ETOPO_2022_v1_1s_N18W067
    /home/mmacferrin/Research/DATA/ETOPO/data/etopo_sources.datalist"""
    # ^ Except, use the source_tile_datalist created here for that entry.


if __name__ == "__main__":
    EG = ETOPO_Generator()
    # EG.generate_etopo_tile_datalist(resolution=1, overwrite=True)
    # EG.generate_tile_source_dlists(source="all",active_only=True,verbose=True)
    # EG.generate_etopo_source_master_dlist()
    # EG.create_empty_grids(resolution_s=60)
    # EG.create_etopo_geopackages()


    # EG.generate_all_etopo_tiles(resolution=15, overwrite=False)
    # EG.copy_finished_tiles_to_onedrive(resolution=15, use_symlinks=True)

    # EG.generate_all_etopo_tiles(resolution=60, overwrite=True)
    # EG.copy_finished_tiles_to_onedrive(resolution=60, use_symlinks=True)


    # EG.generate_all_etopo_tiles(resolution=1, overwrite=False)
    # EG.copy_finished_tiles_to_onedrive(resolution=1, use_symlinks=True)


    # EG.generate_all_etopo_tiles(resolution=1, overwrite=False)
    # EG.copy_finished_tiles_to_onedrive(resolution=1, use_symlinks=True)

    # EG.export_ranks_and_ids_csv()
    EG.write_ranks_and_ids_from_csv()
