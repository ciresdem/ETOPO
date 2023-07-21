# -*- coding: utf-8 -*-

"""etopo_generator.py -- parent code for managing, building and assessing/validating
   the ETOPO global topo-bathy DEM."""

import os
import importlib
from typing import List

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
import argparse
from osgeo import gdal

try:
    import cudem
    # noinspection PyStatementEffect
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
import utils.gtif_to_netcdf
import datasets.dataset_geopackage as dataset_geopackage
import datasets.etopo_source_dataset
import etopo.map_finished_tiles_to_release_directory

etopo_config = utils.configfile.config()

class ETOPO_Generator:
    # The base directory of the project is two levels up. Retrieve the absolute path to it on this machine.
    # This file resides in [project_basedir]/src/etopo
    project_basedir = os.path.abspath(os.path.join(os.path.split(__file__)[0], "..", ".."))

    etopo_config = etopo_config

    # Note, still need to add resolution "1s" or "15s" directory.
    empty_tiles_directory = etopo_config.etopo_empty_tiles_directory
    # intermediate_tiles_directory = etopo_config.etopo_intermediate_tiles_directory # Same as above
    finished_tiles_directory = etopo_config.etopo_finished_tiles_directory # Same as above
    etopo_gpkg_1s  = etopo_config.etopo_tile_geopackage_1s
    etopo_gpkg_15s = etopo_config.etopo_tile_geopackage_15s
    etopo_gpkg_60s = etopo_config.etopo_tile_geopackage_60s

    # Tile naming conventions. ETOPO_2022_v1_N15E175.tif, e.g.
    fname_template_15s_tif = etopo_config.etopo_15_filename_template_tif
    fname_template_15s_netcdf = etopo_config.etopo_15_filename_template_netcdf

    fname_template_1s_tif = etopo_config.crm_1deg_filename_template

    def __init__(self):
        """Read the configuration file, get the info about our grid locations."""
        self.datasets_dict = None

    def create_empty_grids(self, resolution_s=None, verbose=True):
        """Create empty grid-cells for the ETOPO data product.
        resolution_s can be any of 1, 15, 30, 60 or a list/tuple with multiple values (1,15).
        """
        if resolution_s is None:
            resolution_s = (1,15,30,60)
        elif type(resolution_s) in (list,tuple):
            assert numpy.all([(type(r) == int) and (r in (1,15,30,60)) for r in resolution_s])
        elif type(resolution_s) == int:
            assert resolution_s in (1,15,30,60)
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

            # TODO: Refactor this code to natively use the templates for each resolution. Right now it'll default to...
            # TODO: ...the 15s template and then translate it later.
            #
            # tif_template = ETOPO_Generator.fname_template_1s_tif if resolution_s == 1 else ETOPO_Generator.fname_template_15s_tif
            tif_template = ETOPO_Generator.fname_template_15s_tif
            netcdf_template = None

            # In this case, the "tile_width_deg" is the same as the "resolution_s".
            # The 1-deg tiles are 1s resolution, 15-deg tiles are 15s resolution.
            # Makes each tile a 3600x3600 file.
            etopo.generate_empty_grids.create_empty_tiles(resdir,
                                                          fname_template_tif=tif_template,
                                                          fname_template_netcdf=netcdf_template,
                                                          tile_width_deg=(360 if (res in (30,60)) else res),
                                                          resolution_s=res,
                                                          also_write_geopackage=True,
                                                          ndv = ETOPO_Generator.etopo_config.etopo_ndv,
                                                          verbose=verbose)

    def replace_config_value(self, all_txt, configfilename, fieldname, replacement_value):
        """Quick little helper function to replace the text of a value with another value in a configfile text string.

        Used by write_ranks_and_ids_from_csv."""
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


    def write_ranks_and_ids_csv_summary(self, orig_csv_name, df, verbose=True):
        """Given the dataframe of CSV ranks_and_ids values, and the name of the original file, write a "_summary" file."""
        base, ext = os.path.splitext(orig_csv_name)
        summary_csv_name = base + "_summary" + ext

        # Get rid of inactive layers.
        df_subset = df[df.is_active]
        # Sort layers by ranking_score
        df_sorted = df_subset.sort_values("ranking_score")

        if os.path.exists(summary_csv_name):
            os.remove(summary_csv_name)

        df_sorted.to_csv(summary_csv_name, index=False)
        if verbose:
            print("\n" + os.path.basename(summary_csv_name), "written with {0} entries.".format(len(df_sorted)))

        return df_sorted

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

        self.write_ranks_and_ids_csv_summary(csv_name, df, verbose=verbose)

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
            print("\n" + os.path.basename(csv_name), "written with {0} entries.".format(len(df)))

        self.write_ranks_and_ids_csv_summary(csv_name, df, verbose=verbose)

        return df

    def clear_tiles(self, resolution=None, dirname=None, regex_filter = "\.((tif)|(ini))", verbose=True):
        """Clear out all the old ETOPO tiles to make room for new ones."""
        if resolution is None:
            resolution = (1, 15, 60)
        elif type(resolution) == int:
            resolution = (resolution,)

        # Default to clearing out the "finished_tiles" and "datalists" directories in data/
        if dirname is None:
            dirnames = []
            for res in resolution:
                dirnames.extend([os.path.join(self.etopo_config._abspath(self.etopo_config.etopo_datalist_directory),
                                              "{0:d}s".format(res)),
                                 os.path.join(self.etopo_config._abspath(self.etopo_config.etopo_finished_tiles_directory),
                                              "{0:d}s".format(res))])
        elif type(dirname) in (list, tuple):
            dirnames = dirname
        else:
            dirnames = [dirname]

        for dname in dirnames:
            files = utils.traverse_directory.list_files(dname, regex_match=regex_filter)
            for fn in files:
                os.remove(fn)
            if verbose:
                print("{0} files removed from {1}".format(len(files), dname))

        return

    def generate_etopo_tile_datalists(self, gdf = None,
                                            resolution=1,
                                            etopo_tile_fname = None,
                                            bed = False,
                                            subdir = None,
                                            tile_id = None,
                                            crm_only_if_1s = True,
                                            omit_copernicus_if_bed_south = True,
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
        if gdf is None:
            etopo_geopkg_obj = dataset_geopackage.ETOPO_Geopackage(resolution)
            etopo_gdf = etopo_geopkg_obj.get_gdf(crm_only_if_1s=crm_only_if_1s, bed=bed, verbose=verbose)
        else:
            etopo_gdf = gdf
        config_obj = self.etopo_config
        etopo_crs = etopo_gdf.crs

        # print(len(etopo_gdf))
        # print(etopo_gdf)
        # print(etopo_gdf.columns)
        # foobar

        # 0. Get the ETOPO datalist folder, in the "1s" or "15s" or "60s" directory
        datalist_folder = os.path.join(config_obj._abspath(config_obj.etopo_datalist_directory),
                                       str(resolution) + "s")

        resolution = int(resolution)

        if resolution in (30, 60):
            # If the resolution is 30 or 60, just use all the ETOPO 15s tiles to generate the 60s
            # global tile.
            etopo_15s_gpkg = dataset_geopackage.ETOPO_Geopackage(15)
            # Don't just get the bed, look for all the tiles, we'll sort out which ones we should get just the bed.
            etopo_15s_gdf = etopo_15s_gpkg.get_gdf(bed=False, verbose=verbose)

            # We want to query the finished tiles of the dataset, not the empty tiles.
            dlist_filenames = [fn.replace("/empty_tiles/", "/finished_tiles/") for fn in etopo_15s_gdf['filename'].tolist()]

            tilenum_regex_str = r"[NS](\d{2})[EW](\d{3})"

            # If files were date-tagged, then fist get the list of all the files that *are* in the finished_tiles_directory
            finished_tiles_dir = os.path.dirname(dlist_filenames[0])

            if subdir:
                dlist_filenames = [os.path.join(os.path.dirname(fn), subdir, os.path.basename(fn)) for fn in dlist_filenames]
                finished_tiles_dir = os.path.join(finished_tiles_dir, subdir)

            try:
                # This breaks if we've been date-tagging the files.
                assert numpy.all([os.path.exists(fn) for fn in dlist_filenames])
            except AssertionError as e:
                # Find the tiles that do *not* have "_bed" in them.
                finished_tiles_in_dir = [os.path.join(finished_tiles_dir, fn) for fn in os.listdir(finished_tiles_dir) if \
                                         ((re.search(r"ETOPO_2022_([\w\.]+)\.tif\Z", fn) is not None)
                                          and (fn.find("_bed") == -1)
                                          and (fn.find("_w.tif") == -1))]
                # Then search for the tile that matches the base of the tile name we're looking for.
                finished_tiles_lookup_dict = dict([(re.search(tilenum_regex_str, fn).group(), fn) for fn in finished_tiles_in_dir])

                # Replace the list of tiles with ones that have the same tilenum string even if the name's aren't a 100% match.
                # This handles date-tagging errors.
                try:
                    dlist_filenames = [finished_tiles_lookup_dict[re.search(tilenum_regex_str, fn).group()] for fn in dlist_filenames]
                except KeyError as e:
                    print(finished_tiles_lookup_dict)
                    print(finished_tiles_dir)
                    raise e

            # Now if we're doing bed, find any files for which there is a "bed" version of this file.
            if bed:
                tiles_with_bed = []
                # Look either in this folder, or in a "bed" subdir. Could be in either place.
                # Get a list of all tiles that have "_bed" in the name.
                for dirname in (finished_tiles_dir, os.path.join(finished_tiles_dir, "bed")):
                    if not os.path.exists(dirname):
                        continue
                    tiles_with_bed.extend([os.path.join(dirname,fn) for fn in os.listdir(dirname) if \
                                           (re.search(r"ETOPO_2022([\w\.]*)_bed([\w\.]*)\.tif\Z", fn) is not None) \
                                           and (fn.find("_bed") >= 0) \
                                           and (fn.find("_w.tif") == -1)])

                dlist_filenames_temp = []
                # Then loop through the tilenames we're looking for, and see if you can find a bed version.
                for fname in dlist_filenames:
                    fname_dir, fname_base = os.path.split(fname)
                    tilenum = re.search(tilenum_regex_str, fname_base).group()
                    bed_tiles_w_this_tilenum = [fn for fn in tiles_with_bed if os.path.split(fn)[1].find(tilenum) >= 0]
                    # If not matching bed tiles exist, use the regular tile
                    if len(bed_tiles_w_this_tilenum) == 0:
                        dlist_filenames_temp.append(fname)
                    # If a matching bed tile does exist, use that instead.
                    else:
                        if len(bed_tiles_w_this_tilenum) > 1:
                            print(bed_tiles_w_this_tilenum)
                            raise ValueError("Too many... just looking for one match here.")
                        assert len(bed_tiles_w_this_tilenum) == 1
                        dlist_filenames_temp.append(bed_tiles_w_this_tilenum[0])

                assert len(dlist_filenames) == len(dlist_filenames_temp)
                dlist_filenames = dlist_filenames_temp

            datalist_lines = ["{0} 200 1".format(fn) for fn in dlist_filenames]
            datalist_text = "\n".join(datalist_lines)

            etopo_60s_fname = etopo_gdf['filename'].tolist()[0]
            # Add the "_bed" onto the name if we're generating the bed dataset.
            if bed is True:
                base, ext = os.path.splitext(etopo_60s_fname)
                etopo_60s_fname = base + "_bed" + ext
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
            if etopo_tile_fname is None:
                etopo_fnames = etopo_gdf['filename'].tolist()
                etopo_polygons = etopo_gdf['geometry']
            else:
                # If we only have one file, make it a one-item list for the 'for' loop.
                etopo_fnames = [etopo_tile_fname]
                etopo_polygons = etopo_gdf[etopo_gdf.filename == etopo_tile_fname]['geometry'].tolist()
            assert (len(etopo_polygons) == len(etopo_fnames))

            # 2. Get all the ETOPO source dataset objects.
            datasets_list = self.fetch_etopo_source_datasets(verbose=verbose, bed=bed, return_type=list)

            # 3. For each tile in the gdf, create a datalist name for that tile in the datalist folder.
            N = len(etopo_fnames)
            for i,(etopo_fname, etopo_poly) in enumerate(zip(etopo_fnames, etopo_polygons)):
                if (tile_id is not None) and (os.path.basename(etopo_fname).find(tile_id) == -1):
                    continue

                fname_base = os.path.splitext(os.path.split(etopo_fname)[1])[0]
                etopo_tile_datalist_fname = os.path.join(datalist_folder, fname_base + \
                                                         ("_bed" if (bed is True) else "") + \
                                                         ".datalist")

                # 5. Get all the datalist entries from that dataset object. Use the function for that.
                datalist_lines = []

                for dset_obj in datasets_list:
                    # For the bed datasets only, omit copernicus data in the southern tiles (ones that have Snn in them),
                    # so that we don't have the ice-shelve bathtub ring due to ice-edge mismatches between
                    # Copernicus and BedMachine.

                    if bed and omit_copernicus_if_bed_south and dset_obj.dataset_name == "CopernicusDEM":
                        if re.search(r"S(\d{2})", os.path.basename(etopo_fname)) is not None:
                            continue

                    this_dlist_entries = dset_obj.generate_tile_datalist_entries(etopo_poly,
                                                                                 polygon_crs = etopo_crs,
                                                                                 resolution_s = resolution,
                                                                                 verbose = verbose)

                    if len(this_dlist_entries) > 0:
                        datalist_lines.extend(this_dlist_entries)

                # 6. Put all the datalist entries together, write as lines to the datalist.
                datalist_text = '\n'.join(datalist_lines)
                with open(etopo_tile_datalist_fname, 'w') as f:
                    f.write(datalist_text)
                    f.close()
                    if verbose:
                        print("\r" + (" "*120), end="\r")
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
                    utils.progress_bar.ProgressBar(i+1, N, suffix="{0}/{1}".format(i+1, N))

        return

    def add_datestamp_to_filename(self, tilename):
        """Add a _YYYY.MM.DD stamp to the end of the ETOPO filename.

        Useful when debugging the ETOPO tiles."""
        base, ext = os.path.splitext(tilename)
        nowstr = datetime.datetime.now().strftime("%Y.%m.%d")
        assert len(nowstr) == 10
        return base + "_" + nowstr + ext

    def generate_all_etopo_tiles(self,
                                 resolution=1,
                                 numprocs=utils.parallel_funcs.physical_cpu_count(),
                                 add_datestamp_to_files = False,
                                 crm_only_if_1s = True,
                                 bed = False,
                                 subdir = None,
                                 tile_id = None,
                                 tempdir_prefix = "temp",
                                 skip_datalists = False,
                                 overwrite=False,
                                 verbose=True):
        """Generate all of the ETOPO tiles at a given resolution."""
        # Get the ETOPO_geopackage object, with the datalist filenames in it
        # (if they're not in there already, they may be)
        etopo_gdf = dataset_geopackage.ETOPO_Geopackage(resolution).add_dlist_paths_to_gdf(
                                        save_to_file_if_not_already_there = True,
                                        crm_only_if_1s = crm_only_if_1s,
                                        bed = bed,
                                        verbose = verbose)

        # Sort the lists, just 'cuz.
        etopo_tiles  = etopo_gdf['filename'].to_numpy()
        sort_mask    = numpy.argsort(etopo_tiles)

        etopo_tiles  = etopo_tiles[sort_mask]
        etopo_polys  = etopo_gdf['geometry'].to_numpy()[sort_mask]
        etopo_dlists = etopo_gdf['dlist'].to_numpy()[sort_mask]
        etopo_xres   = etopo_gdf['xres'].to_numpy()[sort_mask]
        etopo_yres   = etopo_gdf['yres'].to_numpy()[sort_mask]

        # Look through all the dlists, (re-)generate if necessary.
        if not skip_datalists:
            if verbose:
                print("Generating tile datalists:")
            self.generate_etopo_tile_datalists(gdf=etopo_gdf,
                                               resolution=resolution,
                                               subdir=subdir,
                                               tile_id=tile_id,
                                               crm_only_if_1s = crm_only_if_1s,
                                               etopo_tile_fname=None,
                                               bed = bed,
                                               verbose=verbose)

        if verbose:
            print("Generating", len(etopo_tiles), "ETOPO tiles at", str(resolution) + "s resolution:")

        # Append new active processes to this queue until they're all done.
        active_procs = []
        active_tempdirs = []
        total_finished_procs = 0
        # N = 3 # Put this back when I just want to test a small subset of the files.
        N = len(etopo_tiles)
        waiting_procs = [None] * N
        temp_dirnames = [None] * N
        # current_max_running_procs = numprocs # "max_running_procs" can change depending how many tiles are left.

        dest_tiles_list = []

        # J = 0
        # First, generate a whole list of child processes waiting to be started.
        for i,(tile, poly, dlist, xres, yres) in enumerate(zip(etopo_tiles, etopo_polys, etopo_dlists, etopo_xres, etopo_yres)):

            # This typically does nothing, but handy if we're wanting to do only a small subset of the data and set N to an artificially low number.
            # if i >= N:
            #     break

            if (tile_id is not None) and (resolution < 30) and (os.path.basename(tile).find(tile_id) == -1):
                continue

            dest_tile = tile.replace("/empty_tiles/", "/finished_tiles/")
            if subdir:
                dest_tile = os.path.join(os.path.dirname(dest_tile), subdir, os.path.basename(dest_tile))

            if bed:
                # If we're using the bed, and the bed dlist exists at this tilename, use it.
                base, ext = os.path.splitext(dlist)
                bed_dlist = base + "_bed" + ext
                if os.path.exists(bed_dlist):
                    dlist = bed_dlist
                    tile_base, tile_ext = os.path.splitext(dest_tile)
                    dest_tile = tile_base + "_bed" + tile_ext

            # If we're debugging, add the datestamp to the end of the filename.
            if add_datestamp_to_files:
                dest_tile = self.add_datestamp_to_filename(dest_tile)

            # If the destination tile already exists, just leave it and put a None in the process' place
            if (not overwrite) and os.path.exists(dest_tile):
                proc = None
                temp_dirname = None
            else:
                temp_dirname = os.path.join(self.etopo_config.project_base_directory,
                                            "scratch_data",
                                            tempdir_prefix + str(i))
                if resolution < 30:
                    # If we're at 15s or 1s resolution, use the generate_single_etopo_file function, which calls waffles -M stacks.
                    proc = multiprocessing.Process(target=self.generate_single_etopo_tile,
                                                   args=(dest_tile,
                                                         poly,
                                                         dlist,
                                                         xres,
                                                         yres,
                                                         self.etopo_config.etopo_ndv,
                                                         self.etopo_config.etopo_cudem_cache_directory,
                                                         temp_dirname),
                                                   # If we're only generating one tile and have 'verbose' set, then keep
                                                   # the output on. Else, off, to avoid the parallel-stdout mess.
                                                   kwargs = {'verbose': True if (tile_id and verbose) else False,
                                                             'algorithm': 'auto'})
                                                             # 'algorithm': 'bilinear' if resolution==1 else 'average'})
                else:
                    # If we're at coarser resolution, juse create_etopo_global_tile to just resample into a bigger grid.
                    assert resolution in (30,60)

                    proc = multiprocessing.Process(target=ETOPO_Generator.create_etopo_global_tile,
                                                   args=(self,
                                                         resolution,
                                                         dest_tile,
                                                         temp_dirname),
                                                   kwargs = {'verbose': True,
                                                             'verbose_gdal': False})


            # waiting_procs[J] = proc
            # temp_dirnames[J] = temp_dirname
            waiting_procs[i] = proc
            temp_dirnames[i] = temp_dirname
            dest_tiles_list.append(dest_tile)

            # J += 1

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

                for dproc, tmpdir in zip(procs_waiting_to_be_removed,tmpdirs_waiting_to_be_deleted):
                    active_procs.remove(dproc)
                    rm_cmd = ["rm", "-rf", tmpdir]
                    subprocess.run(rm_cmd, capture_output= True)
                    # for tfn in [os.path.join(tmpdir, tf) for tf in os.listdir(tmpdir)]:
                    #     os.remove(tfn)
                    # os.rmdir(tmpdir)
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
                    if not os.path.exists(tmpdir):
                        os.mkdir(tmpdir)
                    proc_to_start.start()
                    active_procs.append(proc_to_start)
                    active_tempdirs.append(tmpdir)

                # If we're generating more than one tile and 'verbose' is set, use a progress bar.
                if verbose and resolution < 30 and not tile_id:
                    # pass
                    utils.progress_bar.ProgressBar(total_finished_procs, N, suffix="{0:,}/{1:,}".format(total_finished_procs, N))

                time.sleep(0.01)

            # Move all the generated _w.tif "weights" files into the "weights" subdir.
            self.move_weights_to_subdir(resolution=resolution, subdir=subdir, verbose=verbose)

        except Exception as e:
            # If things shut down or crash, clean up the open temp_dirs:
            for tmpdir in active_tempdirs:
                rm_cmd = ["rm","-rf",tmpdir]
                subprocess.run(rm_cmd, capture_output=True)
                # os.rmdir(tmpdir)
            raise e

        return

    def create_etopo_global_tile(self, resolution_s, output_fname, temp_dirname, verbose=True, verbose_gdal=False):
        """Create the 30s or 60s global tile by resampling all the 15s tiles.

        The subdir, and bed/surface options should be already included in the datalist_fname and the output_fname,
        so those parameters are not needed here.
        """
        # 1. Get a list of the filenames to read (from the datalist)
        is_bed = os.path.basename(output_fname).find("_bed") >= 0
        dlist_dirname = os.path.join(self.etopo_config._abspath(self.etopo_config.etopo_datalist_directory),
                     "{0}s".format(resolution_s))
        datalist_fname = [fn for fn in os.listdir(dlist_dirname)
                          if ((re.search("\.datalist\Z", fn) is not None)
                             and (fn.find("S90W180") >= 0)
                             and ((is_bed and fn.find("_bed") >= 0) or ((not is_bed) and fn.find("bed") == -1)))]
        assert len(datalist_fname) == 1
        datalist_fname = os.path.join(dlist_dirname, datalist_fname[0])

        with open(datalist_fname, 'r') as dlist_f:
            dlist_lines = [line for line in dlist_f.readlines() if (len(line.strip()) > 0)]
            assert len(dlist_lines) == 288

        dlist_fnames = sorted([line.split()[0] for line in dlist_lines])

        # 2. From the fist 15s input tile, get the datatype, the projection, the description, the metadata, and the nodatavalue.
        sample_15s_ds = gdal.Open(dlist_fnames[0], gdal.GA_ReadOnly)
        etopo_proj = sample_15s_ds.GetProjection()
        etopo_description = os.path.basename(output_fname)
        sample_15s_band = sample_15s_ds.GetRasterBand(1)
        etopo_ds_metadata = sample_15s_ds.GetMetadata()
        etopo_band_metadata = sample_15s_band.GetMetadata()
        etopo_ndv = sample_15s_band.GetNoDataValue()
        etopo_res = resolution_s / 3600
        etopo_gt = [-180, etopo_res, 0, 90, 0, -etopo_res]
        etopo_datatype = sample_15s_band.DataType
        sample_15s_band = None
        sample_15s_ds = None

        etopo_xsize = int(3600 * 360 / resolution_s)
        etopo_ysize = int(3600 * 180 / resolution_s)
        # Do a sanity check on the sizes here.
        assert etopo_xsize == numpy.round(360 / etopo_res)
        assert etopo_ysize == numpy.round(180 / etopo_res)

        etopo_driver = gdal.GetDriverByName("GTiff")

        try:
            # 3. Create a new geotiff object of global size and the given resolution.
            etopo_ds = etopo_driver.Create(output_fname, etopo_xsize, etopo_ysize, 1, etopo_datatype, options=['COMPRESS=LZW', 'PREDICTOR=2'])

            # 4. Copy the description, metadata into the new dataset. Put the NDV into the data band, and set metadata for the band.
            etopo_ds.SetDescription(etopo_description)
            etopo_ds.SetMetadata(etopo_ds_metadata)
            etopo_ds.SetGeoTransform(etopo_gt)
            etopo_ds.SetProjection(etopo_proj)
            etopo_band = etopo_ds.GetRasterBand(1)
            etopo_band.SetMetadata(etopo_band_metadata)
            etopo_band.SetNoDataValue(etopo_ndv)

            # 5. Get the array in which to write the data.
            etopo_array = etopo_band.ReadAsArray()
            etopo_array[:,:] = etopo_ndv

            # Create a dictionary of all the (x,y) points to make sure we got them all.
            x_vals, y_vals = numpy.meshgrid(numpy.arange(-180,180,15), numpy.arange(90,-90,-15))
            xy_dict = dict([((xv, yv),False) for (xv,yv) in zip(x_vals.flatten(), y_vals.flatten())])

            # 6. For each input tile, gdal_tranlate it into a 15x15-degree lower-resolution virtual (memory) dataset.
            for i, input_fn in enumerate(dlist_fnames):
                temp_15s_tile = os.path.join(temp_dirname, os.path.splitext(os.path.basename(input_fn))[0] + "_15s.tif")
                input_ds = gdal.Open(input_fn, gdal.GA_ReadOnly)
                input_xleft, input_xres, _, input_ytop, _, input_yres = input_ds.GetGeoTransform()
                input_xsize, input_ysize = input_ds.RasterXSize, input_ds.RasterYSize
                input_xright = numpy.round(input_xleft + (input_xres * input_xsize))
                input_ybottom = numpy.round(input_ytop + (input_yres * input_ysize))
                gdalwarp_cmd = ["gdalwarp",
                                "-te", repr(input_xleft), repr(input_ybottom), repr(input_xright), repr(input_ytop),
                                "-ts", str(int(input_xsize * 15 / resolution_s)), str(int(input_ysize * 15 / resolution_s)),
                                "-r", "average",
                                "-of", "GTiff",
                                "-co", "COMPRESS=LZW",
                                "-co", "PREDICTOR=2",
                                input_fn, temp_15s_tile]

                if verbose_gdal:
                    print(" ".join(gdalwarp_cmd))
                subprocess.run(gdalwarp_cmd, capture_output=not verbose_gdal)

                in_x, in_y = int(input_xleft), int(input_ytop)
                assert xy_dict[(in_x, in_y)] is False
                xy_dict[(in_x, in_y)] = True

                # 7. Once the virtual dataset it written, get its array.
                # Check in our dict that all the xy-chunks have been entered.
                assert os.path.exists(temp_15s_tile)
                temp_ds = gdal.Open(temp_15s_tile, gdal.GA_ReadOnly)
                temp_array = temp_ds.GetRasterBand(1).ReadAsArray()
                temp_xsize, temp_ysize = temp_ds.RasterXSize, temp_ds.RasterYSize
                temp_ds = None

                # 8. Calculate what segment of the target array it should be copied into. Copy the data over.
                subset_xoff = int((in_x + 180) * (3600 / resolution_s))
                subset_yoff = int((90 - in_y) * (3600 / resolution_s))
                # Input into the big array.
                etopo_array[subset_yoff: subset_yoff+temp_ysize,
                            subset_xoff: subset_xoff+temp_xsize] = temp_array

                # 9. If verbose, update progress bar.
                if verbose:
                    utils.progress_bar.ProgressBar(i+1, len(dlist_fnames), suffix="{0}/{1}".format(i+1, len(dlist_fnames)))


            assert numpy.all(list(xy_dict.values()))
            # This is a bit redundant, but also check that the whole array is filled (no more NDVs)
            assert not numpy.any(etopo_array == etopo_ndv)

            # 10. After all input tiles, write the array to the output file.
            etopo_band.WriteArray(etopo_array)

            # 11. Save the output file.
            etopo_band = None
            etopo_ds = None

            # 12. If verbose, confirm on the screen.
            if verbose:
                print(os.path.basename(output_fname), "written.")

        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(output_fname):
                os.remove(output_fname)
            for tempfn in [os.path.join(temp_dirname, fn) for fn in os.listdir(temp_dirname)]:
                os.remove(tempfn)
            raise e

        for tempfn in [os.path.join(temp_dirname, fn) for fn in os.listdir(temp_dirname)]:
            os.remove(tempfn)
        return


    def move_all_new_tiles_into_subdir(self, subdir_name, resolution_s = 15, bed=None, include_weights = True, verbose=True):
        """Put all the new tiles specified into a sub-directory, usually a datestamped folder name but can specify."""
        source_dir = os.path.join(self.etopo_config._abspath(self.etopo_config.etopo_finished_tiles_directory), "{0}s".format(resolution_s))
        etopo_regex = r"[NS](\d{2})[EW](\d{3})([\w\.]*)\.tif\Z"
        fnames = [os.path.join(source_dir, fn) for fn in os.listdir(source_dir)
                  if re.search(etopo_regex, fn) != None and
                     ((bed and (re.search("_bed", fn) != None))
                      or ((not bed) and (re.search("_bed", fn) == None)))]

        weights_subdir = os.path.join(source_dir, "weights")
        if include_weights and os.path.exists(weights_subdir) and os.path.isdir(weights_subdir):
            dest_dir = os.path.join(source_dir, subdir_name, "weights")
            shutil.move(weights_subdir, dest_dir)
            if verbose:
                print("Moved", weights_subdir, "to", dest_dir)

        dest_dir = os.path.join(source_dir, subdir_name)
        for i,fname in enumerate(fnames):
            dest_fname = os.path.join(dest_dir, os.path.basename(fname))
            shutil.move(fname, dest_fname)
            if verbose:
                print("{0}/{1}".format(i+1, len(fnames)), fname)

    def fetch_etopo_source_datasets(self, active_only: bool = True, bed: bool = False, verbose: bool = True, return_type: type = dict):
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
        # if self.datasets_dict != None:
        #     if return_type == dict:
        #         return self.datasets_dict
        #     else:
        #         return list(self.datasets_dict.values())

        # Get this directory.
        datasets_dir = os.path.join(ETOPO_Generator.project_basedir, 'src', 'datasets')
        assert os.path.exists(datasets_dir) and os.path.isdir(datasets_dir)

        # Get all the subdirs within this directory.
        subdirs_list = sorted([fn for fn in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, fn))])

        dataset_objects_dict = dict()
        # Loop through all the sub-directories, check requirements for an active dataset class.
        for dataset_name in subdirs_list:
            # If we're looking for the bed (or surface) and we find the other set of BedMachine, skip it.
            # Only if we're looking for active objects only.
            if active_only is True:
                # If we're getting just bed datasets, omit surface datasets for BedMachine & GEBCO
                if (bed is True) and (dataset_name in ("BedMachine_Surface", "GEBCO")):
                    continue
                # If we're getting surface datasets, omit bed datasets for BedMachine & GEBCO
                elif (bed is False) and (dataset_name in ("BedMachine_Bed", "GEBCO_sub_ice")):
                    continue

            subdir = os.path.join(datasets_dir, dataset_name)
            dataset_class_name = "source_dataset_{0}".format(dataset_name)
            module_fname = os.path.join(subdir, dataset_class_name + ".py")

            if os.path.exists(module_fname):
                # Create the import path of the module, relative to the /src
                # directory (which is imported above)
                importpath = "datasets.{0}.{1}".format(dataset_name, dataset_class_name)
                module = importlib.import_module(importpath)

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

    def move_weights_to_subdir(self, resolution=15,
                                     subdir=None,
                                     weights_regex = r"_w\.tif\Z",
                                     verbose=True):
        """The waffles -M stacks module creates a "_w.tif" 'weights' file with each
        ETOPO data tile. Move these into the "weights" subdir in the same directory.
        """
        if type(resolution) in (int,float,str):
            resolution = [int(resolution)]

        for res in resolution:
            tiles_dir = os.path.join(self.etopo_config.etopo_finished_tiles_directory, str(res) + "s")
            if subdir:
                tiles_dir = os.path.join(tiles_dir, subdir)
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


    def output_empty_csv_comment_files(self, files_dirname = None, regex_filter = r"\d{2}\.tif\Z", outfile_name = None, resolution=None, verbose=True):
        """After generating a set of tiles, generate an empty CSV folder with all the tiles listed, to allow comments."""
        # If we don't specify an output file, put it in the finished_tiles directory.
        if resolution is None:
            resolution = (1,15,30,60)
        elif type(resolution) in (int, float):
            resolution = (int(resolution),)

        for res in resolution:
            if outfile_name is None:
                outfile_fn = os.path.join(self.etopo_config._abspath(self.etopo_config.etopo_finished_tiles_directory), "ETOPO_{0}s_tile_comments.csv".format(res))
            else:
                outfile_fn = outfile_name

            if files_dirname is None:
                dir_to_use = os.path.join(self.etopo_config._abspath(self.etopo_config.etopo_finished_tiles_directory), "{0}s".format(res))
            else:
                dir_to_use = os.path.join(files_dirname, "{0}s".format(res))
                if not os.path.exists(dir_to_use):
                    dir_to_use = files_dirname

            files_list = utils.traverse_directory.list_files(dir_to_use, regex_match=regex_filter, ordered=False, include_base_directory=False)
            files_list = sorted([os.path.split(fn)[1] for fn in files_list])

            header = "Filename,Comments"
            lines = [fn + "," for fn in files_list]
            text = "\n".join([header] + lines)
            with open(outfile_fn, 'w') as f:
                f.write(text)
                f.close()

            if verbose:
                print(os.path.split(outfile_fn)[1], "written with", len(files_list), "entries.")

        return

    @staticmethod
    def generate_single_etopo_tile(dest_tile_fname,
                                   dest_tile_polygon,
                                   datalist_fname,
                                   tile_xres,
                                   tile_yres,
                                   etopo_ndv,
                                   etopo_cache_dir,
                                   temp_dir_for_cwd,
                                   algorithm = "auto",
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
                        "-f", # Transform input data if/where needed.
                        # "-a", # Archive the datalist (why again?... nevermind, do not do this, just comment it out.)
                        "-k", # Keep (do not delete) cached files.
                        "-P", "EPSG:4326+3855",
                        "-S", algorithm,
                        "-D", etopo_cache_dir,
                        "-O", os.path.splitext(dest_tile_fname)[0], # Get rid of the .tif at the end. Waffles adds that automatically.
                        datalist_fname]

        if verbose:
            print(" ".join(waffles_args))

        # Run it. Make the "cwd" a temp dir to avoid conflicts.
        # proc = subprocess.run(waffles_args)
        if verbose:
            capture_flag = False
        else:
            capture_flag = subprocess.PIPE
        proc = subprocess.run(waffles_args, cwd=temp_dir_for_cwd, capture_output=capture_flag, text=True)
        if verbose:
            print(dest_tile_fname, "written.")

        return proc.returncode, proc.stdout

        # Sample (complete) waffles command; this is what we're putting together. Leave it here for reference.
        # """waffles -M stacks:supercede=True:keep_weights=True
        # -R -67.0000000/-66.0000000/18.0000000/19.0000000
        # -E 0.000277777777778
        # -N -99999
        # -w
        # -k
        # -f # If using on non-WGS84 datasets
        # -P EPSG:4326
        # -S auto
        # -D /home/mmacferrin/Research/DATA/ETOPO/scratch_data/
        # -O /home/mmacferrin/Research/DATA/ETOPO/data/finished_tiles/1s/ETOPO_2022_v1_1s_N18W067
        # /home/mmacferrin/Research/DATA/ETOPO/data/etopo_sources.datalist"""
        # ^ Except, use the source_tile_datalist created here for that entry.

    @staticmethod
    def convert_to_netcdf(resolution_s=(1, 15, 30, 60),
                          subdir=None,
                          recurse=True,
                          overwrite=False,
                          verbose=True):
        """Convert geotiffs to netcdfs."""
        if type(resolution_s) in (int, float):
            resolution_s = [int(resolution_s)]

        for res in resolution_s:
            dirname = os.path.join(etopo_config._abspath(etopo_config.etopo_finished_tiles_directory),
                                   "{0}s".format(res))
            if subdir:
                dirname = os.path.join(dirname, subdir)

            utils.gtif_to_netcdf.gtiff_to_netcdf(dirname,
                                                 recurse_subdirs=recurse,
                                                 omit_regex=r"_w\.tif\Z",
                                                 dest_subdir="netcdf",
                                                 overwrite=overwrite,
                                                 verbose=verbose)
        return


def remove_all_inf_files(verbose=True):
    """Remove all .inf files from the results directories created by waffles."""
    base_dir = etopo_config._abspath(etopo_config.etopo_finished_tiles_directory)
    inf_files = utils.traverse_directory.list_files(base_dir,
                                                    regex_match=r"\.inf\Z",
                                                    include_base_directory=True)
    for inf in inf_files:
        os.remove(inf)

    if verbose:
        print(len(inf_files), ".inf files deleted.")
    return


def get_yx_slices(dest_gt, src_gt, src_xsize, src_ysize):
    """Given the geotransform of a large dataset that fully includes the bounds of a smaller dataset, on the same
    grid, return the y-slice and x-slice of the big array in which the small array would fit."""
    src_ulx, src_xres, _, src_uly, _, src_yres = src_gt
    dst_ulx, dst_xres, _, dst_uly, _, dst_yres = dest_gt

    # If the source & dest are not one the same grids, this'll cause issues. Make sure they are.
    assert (src_xres == dst_xres) and (src_yres == dst_yres)

    xstart = int(numpy.round((src_ulx - dst_ulx) / src_xres))
    assert xstart >= 0
    xslice = slice(xstart, int(xstart + src_xsize))

    ystart = int(numpy.round((src_uly - dst_uly) / src_yres))
    assert ystart >= 0
    yslice = slice(ystart, int(ystart + src_ysize))

    return yslice, xslice

def create_15s_global_tile(bed = False, subdir=None, verbose=True):
    """Right now, the global tile is in 60s resolution. Here, paste together all the 15s tiles (of bed or surface)
    to create a 15 global tile."""
    srcdir = os.path.join(etopo_config.etopo_finished_tiles_directory, "15s")
    if subdir is not None:
        srcdir = os.path.join(srcdir, subdir)

    input_files = sorted([os.path.join(srcdir, fn) for fn in os.listdir(srcdir) if \
                   (re.search(r"ETOPO_2022_v1_15s([\w\.]+)(\.\d{2})\.tif\Z", fn) is not None) and (fn.find("_bed") == -1)])

    dlist_fname = os.path.join(etopo_config.etopo_datalist_directory ,"60s", "TEMP_15_sec_tiles.datalist")
    dlist_weights_fname = os.path.splitext(dlist_fname)[0] + "_weights.datalist"
    if bed:
        dlist_fname = os.path.splitext(dlist_fname)[0] + "_bed.datalist"
        dlist_weights_fname = os.path.splitext(dlist_fname)[0] + "_weights.datalist"

        bed_files = [os.path.join(srcdir, fn) for fn in os.listdir(srcdir) \
                     if (re.search(r"ETOPO_2022_v1_15s([\w\.]+)(\.\d{2})\.tif\Z", fn) is not None) and (fn.find("_bed") >= 0)] \
                     + \
                     [os.path.join(srcdir, "bed", fn) for fn in os.listdir(os.path.join(srcdir, "bed")) \
                     if (re.search(r"ETOPO_2022_v1_15s([\w\.]+)(\.\d{2})\.tif\Z", fn) is not None) and (fn.find("_bed") >= 0)]

        # Make a dict with the NYYWXXX identifiers as the key, and the filename as the value.
        # Then we can copy in the bed tiles, read out the values, and it should work.
        tilenames_dict = dict([(re.search(r"[NS](\d{2})[EW](\d{3})(?=([\w\.]*)\.tif\Z)", os.path.split(fn)[1]).group(), fn) for fn in input_files])
        for bfile in bed_files:
            tilenames_dict[re.search(r"[NS](\d{2})[EW](\d{3})(?=([\w\.]*)\.tif\Z)", os.path.split(bfile)[1]).group()] = bfile
        input_files = sorted(tilenames_dict.values())

    if verbose:
        print(len(input_files), "source tiles.")

    # Now, get the weights so we can include those too.
    weights_files: list[None] = [None] * len(input_files)
    # Weights could be an identical file with an "_w" on it, and/or the same thing in a "weights" subdir.
    for i, fname in enumerate(input_files):
        base, ext = os.path.splitext(fname)
        dname, base = os.path.split(base)
        weights_name_1 = os.path.join(dname, base + "_w" + ext)
        if os.path.exists(weights_name_1):
            weights_files[i] = weights_name_1
            continue

        weights_name_2 = os.path.join(dname, "weights", base + "_w" + ext)
        if os.path.exists(weights_name_2):
            weights_files[i] = weights_name_2
            continue

        weights_name_3 = os.path.join(dname, "bed_weights", base + "_w" + ext)
        if os.path.exists(weights_name_3):
            weights_files[i] = weights_name_3
            continue

        raise FileNotFoundError("Could not find weights file for input file " + fname)

    # Create the datalists.
    # dlist_text = "\n".join(["{0} 200 1".format(fn) for fn in input_files])
    # with open(dlist_fname, 'w') as f:
    #     f.write(dlist_text)
    #     f.close()
    # if verbose:
    #     print(dlist_fname, "written.")
    #
    # # Also create the datalist for the weights files.
    # dlist_weights_text = "\n".join(["{0} 200 1".format(fn) for fn in weights_files])
    # with open(dlist_weights_fname, 'w') as f:
    #     f.write(dlist_weights_text)
    #     f.close()
    # if verbose:
    #     print(dlist_weights_fname, "written.")

    dest_tile_fname = os.path.join(etopo_config.etopo_finished_tiles_directory, "15s", "global", "ETOPO_2022_v1_15s_global" + ("_bed" if bed else "") + ".tif")
    weights_tile_fname = os.path.splitext(dest_tile_fname)[0] + "_w.tif"

    empty_tilename = etopo_config.etopo_empty_15s_global_tile
    assert os.path.exists(empty_tilename)

    for dest_filename, src_file_list in [(dest_tile_fname, input_files), (weights_tile_fname, weights_files)]:

        if verbose:
            print("Creating", os.path.split(dest_filename)[1] + "...", end="")

        src_ds = gdal.Open(empty_tilename, gdal.GA_ReadOnly)
        assert src_ds is not None
        driver = src_ds.GetDriver()
        dest_ds = driver.CreateCopy(dest_filename, src_ds)
        dest_ds = None
        src_ds = None
        if verbose:
            print(" blank tile created.")

        assert os.path.exists(dest_filename)
        dest_ds = gdal.Open(dest_filename, gdal.GA_Update)
        dest_band = dest_ds.GetRasterBand(1)
        dest_array = dest_band.ReadAsArray()
        # xsize, ysize = dest_ds.RasterXSize, dest_ds.RasterYSize
        dest_gt = dest_ds.GetGeoTransform()

        for i,src_fname in enumerate(src_file_list):
            if verbose:
                print("  {0}/{1} reading".format(i+1, len(src_file_list)), os.path.split(src_fname)[1])
            # Get the source dataset array.
            src_ds = gdal.Open(src_fname, gdal.GA_ReadOnly)
            src_band = src_ds.GetRasterBand(1)
            src_array = src_band.ReadAsArray()
            xsize, ysize = src_ds.RasterXSize, src_ds.RasterYSize
            # Get the source dataset geotransform
            src_gt = src_ds.GetGeoTransform()

            # Find the slice of the output array this goes in.
            yslice, xslice = get_yx_slices(dest_gt, src_gt, xsize, ysize)
            # Insert source data into the destination array.
            dest_array[yslice, xslice] = src_array

            # Close the source dataset.
            src_band = None
            src_ds = None

        # Write out the new array.
        dest_band.WriteArray(dest_array)

        # If weights, reset the "nodata" value to zero.
        if dest_filename == weights_tile_fname:
            dest_band.SetNoDataValue(0)
        # Compute the stats of the output dataset.
        dest_band.GetStatistics(0,1)
        # Flush the cache.
        dest_ds.FlushCache()
        # Write the dest to disk.
        dest_band = None
        dest_ds = None
        if verbose:
            print(dest_filename, "" if os.path.exists(dest_filename) else "NOT", "written.")

    # empty_datalist_fname = os.path.splitext(dlist_fname)[0] + "_empty.datalist"
    # empty_datalist_weights_fname = os.path.splitext(dlist_weights_fname)[0] + "_empty.datalist"

    # for (dest_fn, dlist_fn) in ([dest_tile_fname, dlist_fname],[weights_tile_fname, dlist_weights_fname]):
    #     waffles_args = ["waffles",
    #                     "-M", "stacks:supercede=True",
    #                     "-R", "-180/180/-90/90",
    #                     "-N", "-99999",
    #                     "-E", "15s",
    #                     "-w", # Use the datalist weights.
    #                     "-S", "near",
    #                     "-P", "EPSG:4326",
    #                     "-O", os.path.splitext(dest_fn)[0], # Get rid of the .tif at the end. Waffles adds that automatically.
    #                     dlist_fn]
    #
    #     if verbose:
    #         print(" ".join(waffles_args))

    #     subprocess.run(waffles_args, capture_output=not verbose)

    if not os.path.exists(dest_tile_fname) and verbose:
        print("ERROR:", dest_tile_name, "NOT created.")
    if not os.path.exists(weights_tile_fname) and verbose:
        print("ERROR:", weights_tile_fname, "NOT created.")

    return


def translate_1s_etopo_tilenames_to_crm_template(subdir=None,
                                                 verbose=True):
    """Translate ETOPO_1s_v1_NXXWXXX.tif filenames into the crm ncei1_nYYxYY_wXXXxXX_YYYY_vZ.tif template.

    This is a temporary function. I need to refactor the code to get it to do this natively when creating the files,
    but right now it's easier to create them as-is and then change the names after-the-fact. So that's what I'm doing here."""
    etopo_tile_regex = r"ETOPO_2022_v1_1s_[NS](\d{2})[EW](\d{3})(_\d{4}\.\d{2}\.\d{2})?(_sid)?.tif\Z"
    crm_outfile_template=etopo_config.crm_1deg_filename_template

    tile_directory = os.path.join(etopo_config._abspath(etopo_config.etopo_finished_tiles_directory),
                                  "1s")
    if subdir is not None:
        tile_directory = os.path.join(tile_directory, subdir)

    tile_list = utils.traverse_directory.list_files(tile_directory, etopo_tile_regex)

    year = datetime.datetime.now().year
    version_num = 1 # TODO: Add logic to adjust this as new versions come out.

    if verbose:
        print("== Translating", len(tile_list), "files to CRM naming structure. ==")

    for i, tname in enumerate(tile_list):
        # Get the lat-lon of the SW corner from the TILEID.
        etopo_tile_basename = os.path.basename(tname)
        tile_id = re.search(r"(?<=ETOPO_2022_v1_1s_)[NS]\d{2}[EW]\d{3}", etopo_tile_basename).group()
        lat_sw = (1 if tile_id[0] == "N" else -1) * int(tile_id[1:3])
        lon_sw = (1 if tile_id[3] == "E" else -1) * int(tile_id[4:7])
        lat_nw = lat_sw + 1
        lon_nw = lon_sw

        assert -90 < lat_nw <= 90
        assert -180 <= lon_nw < 180

        tbasename_out = crm_outfile_template.format("n" if lat_nw >= 0 else "s",
                                                    abs(lat_nw),
                                                    "e" if lon_nw >= 0 else "w",
                                                    abs(lon_nw),
                                                    year,
                                                    version_num)

        # If the filename includes a date, include it in the output.
        date_regex = r"_\d{4}\.\d{2}\.\d{2}"
        re_result = re.search(date_regex, etopo_tile_basename)
        if re_result is not None:
            date_str = re_result.group()
            base, ext = os.path.splitext(tbasename_out)
            tbasename_out = base + date_str + ext

        if etopo_tile_basename.find("_sid.tif") > -1:
            base, ext = os.path.splitext(tbasename_out)
            tbasename_out = base + "_sid" + ext

        tname_out = os.path.join(os.path.dirname(tname), tbasename_out)

        if verbose:
            print(etopo_tile_basename, "->", tbasename_out)
        shutil.move(tname, tname_out)
        # print("  ", tname_out)


def remove_extraneous_files_from_outputs(resolution_s=15,
                                         subdir=None,
                                         file_regex=r"((_[uw]\.tif)|(\.aux\.xml)|(\.inf))\Z",
                                         also_remove_empty_dirs=True,
                                         verbose=True):
    """Right now the scripts output the tiles and SID tags, and also some extraneous _u and also keeps the _w weights files.
    If we wish to get rid of those, call this function to remove them.

    Also, by default, get rid of all .aux.xml and .inf files.

    It will also remove empty sub-directories if specified, but only in the base directory. Does not (yet) recurse.
    (I could add recursion later but don't need it right now so I'm not bothering for now.)"""
    # TODO: FINISH
    tile_directory = os.path.join(etopo_config._abspath(etopo_config.etopo_finished_tiles_directory),
                                  "{0:d}s".format(resolution_s))
    if subdir:
        tile_directory = os.path.join(tile_directory, subdir)

    # First, find all files matching the patterns.
    files_to_remove = utils.traverse_directory.list_files(tile_directory, regex_match=file_regex)
    if len(files_to_remove) > 0:
        if verbose:
            print("Removing", len(files_to_remove), "extraneous files.")
        for fname in files_to_remove:
            # print(fname)
            os.remove(fname)

    if also_remove_empty_dirs:
        fnames_in_dir = [os.path.join(tile_directory, fn) for fn in os.listdir(tile_directory)]
        for fn in fnames_in_dir:
            if os.path.isdir(fn) and (len(os.listdir(fn)) == 0):
                if verbose:
                    print("Removing empty dir '{0}'.".format(os.path.basename(fn)))
                # print(fn)
                os.rmdir(fn)


def weights_to_sids(resolution_s,
                    subdir=None,
                    bed=False,
                    relative_dest_dir="sid",
                    tile_id=None,
                    src_regex=r"_w\.tif\Z",
                    overwrite=False,
                    verbose=True):
    """Get all the weights files from a, and convert them to GDALByte integer arrays, and save as "_sid.tif" rather than _w.tif"

    Put in the relative source directory."""
    if type(resolution_s) in (int,float):
        resolution_s = [int(resolution_s)]

    for res in resolution_s:
        weights_dir = os.path.join(etopo_config._abspath(etopo_config.etopo_finished_tiles_directory), "{0}s".format(res))
        if subdir is not None:
            weights_dir = os.path.join(weights_dir, subdir)

        if not os.path.exists(weights_dir):
            return

        if bed and os.path.exists(os.path.join(weights_dir, "bed")):
            weights_dir = os.path.join(weights_dir, "bed")

        if os.path.exists(os.path.join(weights_dir, "weights")):
            weights_dir = os.path.join(weights_dir, "weights")

        w_files = sorted([os.path.join(weights_dir, fn) for fn in os.listdir(weights_dir) if
                          (re.search(src_regex, fn) is not None) and ((bed and (re.search("_bed_", fn) is not None)) or
                                                                      (not bed and (re.search("_bed_", fn) is None)))])
        print(weights_dir)
        assert len(w_files) > 0

        for i,wfile in enumerate(w_files):
            if tile_id is not None and os.path.basename(wfile).find(tile_id) == -1:
                continue
            # Replace the "_w" identifier with "_sid"
            sid_file = wfile.replace("_w.tif", "_sid.tif")
            # If it's in a separate "weights" dir, put it one level over into a "sid" dir.
            if os.path.split(os.path.dirname(sid_file))[1] == "weights":
                sid_file = sid_file.replace("/weights/", "/" + relative_dest_dir + "/")
            # If it's not in a "weights" dir, put it into a "sid" sub-folder.
            else:
                sid_file = os.path.join(os.path.dirname(sid_file), relative_dest_dir, os.path.split(sid_file)[1])

            if not os.path.exists(os.path.dirname(sid_file)):
                os.mkdir(os.path.dirname(sid_file))

            if os.path.exists(sid_file):
                if overwrite:
                    os.remove(sid_file)
                else:
                    if verbose:
                        print("{0}/{1} {2}".format(i+1, len(w_files), os.path.split(sid_file)[1]), "already exists.")
                    continue

            gdal_cmd = ["gdal_translate",
                        "-ot", "Byte",
                        "-of", "GTiff",
                        "-co", '"COMPRESS=ZSTD"', "-co", '"PREDICTOR=2"', "-co", '"TILED=YES"',
                        "-a_srs", "EPSG:4326",
                        wfile, sid_file]

            subprocess.run(gdal_cmd, capture_output=True)

            if verbose:
                print("{0}/{1} {2}".format(i+1, len(w_files), os.path.split(sid_file)[1]),
                      ("" if os.path.exists(sid_file) else "NOT ") + "written.")


def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Generate the tiles. All proprocessing must be done on source datasets first, including cleansing and/or vertical datum transformations.")
    parser.add_argument("-resolution", "-r", default=(15, 1, 60, 30), nargs="*", help="Grid resolution. Can add up to 4, choices of: 1 15 30 60. Will be exectued in the order given. Default: 15 1 60 30.")
    parser.add_argument("-numprocs", "-np", type=int, default=utils.parallel_funcs.physical_cpu_count(), help="Number of processes to run at once. Default: max number of physical cores available to the machine.")
    parser.add_argument("-subdir", type=str, default="", help="Put all new tiles into a subdir. If the string 'today' is given, a subdir will be selected with the YYYY.MM.DD of today's date.")
    parser.add_argument("-tempdir_prefix", type=str, default="temp" + str(os.getpid()) + "_", help="A prefix to use for temporary directories. Specify if running more than one process to keep them from interfering with each other using the same set of temp directories. Default 'temp'.")
    parser.add_argument("-tile_id", default=None, help="Do only one tile. Specify the seven-character NYYEXXX tile_id to use.")
    parser.add_argument("--all_tiles", "-all", default=False, action="store_true", help="Generate all the tiles, all resolutions, both .tif and .nc, surface and bed.")
    parser.add_argument("--datestamp", "-d", default=False,action="store_true", help="Add a _YYYY.MM.DD datestamp to the end of the tiles. Useful for debugging.")
    parser.add_argument("--bed", "-b", default=False, action="store_true", help="Compute tiles only for bed topographies. Compute just the tiles that overlap the ice sheet beds.")
    parser.add_argument("--datalists_only", "-dl", default=False, action="store_true", help="Only generate the datalists, not the tiles.")
    parser.add_argument("--global_15s", default=False, action="store_true", help="Generate a 15-sec resolution global tile.")
    parser.add_argument("--to_netcdf", default=False, action="store_true", help="After processing tiles, convert geotiff to netcdf files.")
    parser.add_argument("--to_netcdf_only", default=False, action="store_true", help="Just convert geotiffs to NetCDFs. Don't do any other processing.")
    parser.add_argument("--ini_to_csv", default=False, action="store_true", help="Crank out the latest datasets CSV, from the current config files.")
    parser.add_argument("--csv_to_ini", default=False, action="store_true", help="Read any changes made in the CSV, back into the respective .INI files. All old .ini's will be saved to a _old.ini file. Any previous _old.ini's will be overwritten though, so be careful and make sure that changes are what we want.")
    parser.add_argument("--remove_inf", default=False, action="store_true", help="Remove all .inf files from finished_tiles directory.")
    parser.add_argument("--move_to_subdir", default=False, action="store_true", help="Just move the created tiles into a sub-directory. '-subdir' must be specified.")
    parser.add_argument("--weights_to_sids", "-w2s", default=False, action="store_true", help="Convert floating-point 'weights' files to byte 'sid' files.")
    parser.add_argument("--skip_datalists", "-sd", default=False, action="store_true", help="Skip recreating the datalists (just use whatever was there already).")
    parser.add_argument("--overwrite", "-o", default=False, action="store_true", help="Overwrite all existing files.")
    parser.add_argument("--quiet", "-q", default=False, action="store_true", help="Run in quiet mode.")
    return parser.parse_args()

if __name__ == "__main__":

    # translate_1s_etopo_tilenames_to_crm_template(subdir="2023.07.09")
    # remove_extraneous_files_from_outputs(resolution_s=1, subdir="2023.07.09")
    # foobar

    args = define_and_parse_args()

    if args.subdir == "":
        args.subdir = None
    elif args.subdir.lower() == "today":
        args.subdir = datetime.datetime.today().strftime("%Y.%m.%d")

    args.resolution = [int(res) for res in args.resolution]

    if args.global_15s:
        create_15s_global_tile(bed = args.bed, subdir = args.subdir, verbose=not args.quiet)

    elif args.ini_to_csv:
        EG = ETOPO_Generator()
        EG.export_ranks_and_ids_csv(verbose=not args.quiet)

    elif args.csv_to_ini:
        EG = ETOPO_Generator()
        EG.write_ranks_and_ids_from_csv(verbose=not args.quiet)

    elif args.remove_inf:
        remove_all_inf_files(verbose = not args.quiet)

    elif args.weights_to_sids:
        weights_to_sids(args.resolution,
                        subdir=args.subdir,
                        bed=args.bed,
                        tile_id=args.tile_id,
                        overwrite=args.overwrite,
                        verbose=not args.quiet)

    elif args.move_to_subdir:
        assert ((args.subdir != "") and (args.subdir is not None))
        EG = ETOPO_Generator()
        for res in args.resolution:
            EG.move_all_new_tiles_into_subdir(args.subdir,
                                              resolution_s=res,
                                              bed=args.bed,
                                              include_weights=True,
                                              verbose=not args.quiet)

    elif args.to_netcdf_only:
        EG = ETOPO_Generator()
        for res in args.resolution:
            EG.convert_to_netcdf(res,
                                 subdir=args.subdir,
                                 recurse=True,
                                 overwrite=args.overwrite,
                                 verbose=not args.quiet)

    elif args.all_tiles:
        EG = ETOPO_Generator()
        for res in args.resolution:
            if res == 1:
                bed_opts = (False,)
            else:
                bed_opts = (True, False)

            for bed_o in bed_opts:
                EG.generate_all_etopo_tiles(resolution=res,
                                            numprocs = args.numprocs,
                                            add_datestamp_to_files= args.datestamp,
                                            crm_only_if_1s=True,
                                            bed=bed_o,
                                            tile_id=args.tile_id,
                                            overwrite=args.overwrite,
                                            subdir=args.subdir,
                                            tempdir_prefix=args.tempdir_prefix,
                                            verbose=not args.quiet)

                weights_to_sids(res,
                                subdir=args.subdir,
                                bed=bed_o,
                                overwrite=args.overwrite,
                                verbose=not args.quiet)

            EG.convert_to_netcdf(res,
                                 subdir=args.subdir,
                                 recurse=True,
                                 overwrite=args.overwrite,
                                 verbose=not args.quiet)

        # If we're only producing the 1s versions, don't re-generate the output directory. Otherwise, if we're
        # generating tiles at 15, 30, and/or 60s resolution, re-create the output directory with those tiles.
        if not ((len(args.resolution) == 1) and (args.resolution[0] == 1)):
            etopo.map_finished_tiles_to_release_directory.generate_output_directory_and_files(src_subdir=args.subdir,
                                                                                              delete_old=True,
                                                                                              skip_pdfs=True,
                                                                                              print_only=False,
                                                                                              summarize=not args.quiet)

    else:
        EG = ETOPO_Generator()

        for res in args.resolution:
            res = int(res)
            if args.datalists_only:

                EG.generate_etopo_tile_datalists(resolution=res,
                                                 crm_only_if_1s=True,
                                                 subdir=args.subdir,
                                                 tile_id=args.tile_id,
                                                 bed=args.bed,
                                                 verbose=not args.quiet)
            else:
                EG.generate_all_etopo_tiles(resolution=res,
                                            add_datestamp_to_files = args.datestamp,
                                            crm_only_if_1s = True,
                                            overwrite= args.overwrite,
                                            bed=args.bed,
                                            subdir=args.subdir,
                                            tile_id=args.tile_id,
                                            numprocs=args.numprocs,
                                            tempdir_prefix=args.tempdir_prefix,
                                            skip_datalists=args.skip_datalists,
                                            verbose=not args.quiet)

                weights_to_sids(res,
                                subdir=args.subdir,
                                bed=args.bed,
                                overwrite=args.overwrite,
                                verbose=not args.quiet)

                if res == 1:
                    translate_1s_etopo_tilenames_to_crm_template(subdir=args.subdir, verbose=not args.quiet)

                if args.to_netcdf:
                    EG.convert_to_netcdf(res,
                                         subdir=args.subdir,
                                         recurse=True,
                                         overwrite=args.overwrite,
                                         verbose=not args.quiet)

                remove_extraneous_files_from_outputs(resolution_s=res, subdir=args.subdir, verbose=not args.quiet)
        # remove_all_inf_files(verbose = not args.quiet)

