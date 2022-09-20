# -*- coding: utf-8 -*-

"""Source code for the NOAA_regional ETOPO source dataset class."""

import os
import subprocess
import re
import sys
from osgeo import gdal
import pyproj
import multiprocessing
import argparse
import pandas

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset
import utils.configfile
etopo_config = utils.configfile.config()

class source_dataset_NOAA_regional(etopo_source_dataset.ETOPO_source_dataset):
    """Look in "src/datasets/etopo_source_dataset.py" to get base class definition."""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "NOAA_regional_config.ini" )):
        """Initialize the NOAA_regional source dataset object."""

        super(source_dataset_NOAA_regional, self).__init__("NOAA_regional", configfile)

    def print_unique_vdatum_ids(self):
        """Print a list of unique vertical datum codes from the file names.

        The source .nc data comes in a variety of vertical datums, which are outlined in the filename.
        """
        fnames = [fn for fn in os.listdir(self.config.source_datafiles_directory) if re.search(self.config.datafiles_regex, fn) is not None]

        vdatum_codes = [re.search("(?<=\_)[a-zA-Z0-9]+_([m\d]+)(?=\.nc\Z)", fn).group().split("_")[0] for fn in fnames]
        unique_vdatum_codes = sorted(list(set([vd.lower() for vd in vdatum_codes])))
        print(", ".join(unique_vdatum_codes))
        print(len(unique_vdatum_codes), "unique vdatum codes.")

        files_without_weird_codes = [fn for fn in fnames if numpy.any([(fn.lower().find("_" + vd + "_") > -1) for vd in ("isl","mhhw","mhw","mllw","msl","navd88","prvd02")])]
        print("\n".join(files_without_weird_codes))
        print(len(files_without_weird_codes), "files remaining.")

    def move_estuarine_tiles(self):
        """Some tiles originally downloaded are actually part of an older 'Estuarine' dataset. Move them to that layer."""
        fnames = [fn for fn in os.listdir(self.config.source_datafiles_directory) if re.search(self.config.datafiles_regex, fn) is not None]
        fpaths = [os.path.join(self.config.source_datafiles_directory, fn) for fn in fnames]

        vdatum_codes = [re.search("(?<=\_)[a-zA-Z0-9]+_([m\d]+)(?=\.nc\Z)", fn).group().split("_")[0].lower() for fn in fnames]

        for fp, vd in zip(fpaths, vdatum_codes):
            if vd not in ("isl","mhhw","mhw","mllw","msl","navd88","prvd02"):
                estuary_dirname = os.path.join(os.path.dirname(os.path.dirname(fp)), "estuarine")
                assert os.path.exists(estuary_dirname) and os.path.isdir(estuary_dirname)
                shutil.move(fp, os.path.join(estuary_dirname, os.path.split(fp)[1]))
                print("moved", os.path.split(fp)[1])

    def convert_to_gtiff(self, overwrite=False):
        """Convert all the .nc files to geotiffs."""
        fnames = sorted([os.path.join(self.config.source_datafiles_directory, fn) for fn in os.listdir(self.config.source_datafiles_directory) if re.search(self.config.datafiles_regex, fn) is not None])
        # Go ahead and include everything in the "estuarine" folder too. Take care of both datasets at once.
        source_dir_estuarine = os.path.join(os.path.dirname(self.config.source_datafiles_directory), "estuarine")
        fnames = fnames + sorted([os.path.join(source_dir_estuarine, fn) for fn in os.listdir(source_dir_estuarine) if re.search(self.config.datafiles_regex, fn) is not None])

        for i, fname in enumerate(fnames):
            fout = os.path.join(os.path.dirname(fname), "gtif", os.path.splitext(os.path.split(fname)[1])[0] + ".tif")
            if os.path.exists(fout):
                if overwrite:
                    os.remove(fout)
                else:
                    print("{0}/{1} {2} already exists.".format(i+1, len(fnames), os.path.split(fout)[1]))
                    continue
            gdal_cmd = ["gdal_translate",
                        "-co", "COMPRESS=LZW",
                        "-co", "PREDICTOR=2",
                        fname,
                        fout]
            # print(" ".join(gdal_cmd))
            print("{0}/{1} {2}".format(i + 1, len(fnames), os.path.split(fout)[1]), end="")
            sys.stdout.flush()
            try:
                subprocess.run(gdal_cmd, capture_output=True)
            except (Exception, KeyboardInterrupt) as e:
                if os.path.exists(fout):
                    os.remove(fout)
                    print(" deleted.")
                    sys.stdout.flush()
                raise e
            print(" written.")
            sys.stdout.flush()

    @staticmethod
    def convert_to_wgs84_and_egm2008_single_tile(fname, i, N, test_stop = None, overwrite=False):
        """Convert a single tile to wgs84 horizontal, and egm2008 vertical datums.

        test_stop is for debugging purposes. Set to "PROJECTED", "CONVERTED", or "TIDAL" to report the output of those
        respective commands (rather than running them silently) and then immediately stop the program.
        """
        if test_stop == "":
            test_stop = None
        elif test_stop is not None:
            # Put the flag in upper-case, and strip away any whitespace.
            test_stop = test_stop.strip().upper()

        print("{0}/{1} {2}".format(i + 1, N, os.path.split(fname)[1]))

        # First, just list out all the epsg values to see here.
        ds = gdal.Open(fname, gdal.GA_ReadOnly)
        wkt = ds.GetProjection()
        # At least one of these files includes no projection information. Assign 4326 to it.
        if wkt == '':
            ds = None
            ds = gdal.Open(fname, gdal.GA_Update)
            gt = ds.GetGeoTransform()
            xmin, ymax = gt[0], gt[3]
            # Make sure the coordinates make sense for a lat/lon projection before assigning this.
            assert (-180 <= xmin <= 180) and (-90 <= ymax <= 90)
            wkt = pyproj.crs.CRS.from_epsg(4326).to_wkt()
            ds.SetProjection(wkt)
            print("\t", os.path.split(fname)[1], "set to EPSG:4326")

        proj = pyproj.Proj(wkt)
        epsg = proj.crs.to_epsg()

        # Check to make sure the image's geotransform is in the bounds of EPSG: 4326. Some are not.
        if epsg == 4326:

            gt = ds.GetGeoTransform()
            xmin, ymax = gt[0], gt[3]
            assert (-90 <= ymax <= 90)
            if xmin > 180 or xmin < -180:
                # At least some of these files seem to have xmins that are wrapped around the globe,
                # i.e. +190 rather than -170. Detect this and correct for it.
                ds = None
                ds = gdal.Open(fname, gdal.GA_Update)
                gt = ds.GetGeoTransform()
                xmin, ymax = gt[0], gt[3]
                if xmin < -180:
                    xmin = xmin + 360
                    assert xmin >= -180
                elif xmin > 180:
                    xmin = xmin - 360
                    assert xmin < 180
                new_gt = (xmin, gt[1], gt[2], gt[3], gt[4], gt[5])
                ds.SetGeoTransform(new_gt)
                print("\t", os.path.split(fname)[1], "GT corrected from", gt, "to", new_gt)

        ds = None

        try:
            vdatum_str = re.search("(?<=\w_)[A-Za-z0-9]+(?=_[\dm]+(_epsg4326)?\.tif\Z)",
                                   os.path.split(fname)[1]).group()
        except AttributeError:
            print(os.path.split(fname)[1])
            return fname, False
        if vdatum_str not in ("isl", "mhhw", "mhw", "mllw", "msl", "navd88", "prvd02"):
            vdatum_str = "mllw"

        # If it's not already in EPSG 4326, convert it to EPSG 4326
        if epsg == 4326:
            # Create a symbolic link into the projected folder.
            # fname_projected = os.path.join(os.path.dirname(fname), "projected", os.path.basename(fname))
            # os.symlink(fname, fname_projected)
            # print("\t", "Symlink", os.path.basename(fname), "created in 'projected' folder.")
            fname_projected = fname
        else:
            fname_projected = os.path.join(os.path.dirname(fname), "projected" if os.path.split(os.path.dirname(fname))[
                                                                                      1] != "projected" else ".",
                                           os.path.splitext(os.path.basename(fname))[0] + "_epsg4326.tif")
            if os.path.exists(fname_projected):
                if overwrite:
                    os.remove(fname_projected)

            if not os.path.exists(fname_projected):
                gdal_cmd = ["gdalwarp",
                            "-t_srs", "EPSG:4326",
                            "-dstnodata", "-99999",
                            "-co", "COMPRESS=LZW",
                            "-co", "PREDICTOR=3",
                            fname,
                            fname_projected]

                # print(" ".join(gdal_cmd))
                print("\t", "Creating", os.path.split(fname_projected)[1])
                try:
                    if test_stop == "PROJECTED":
                        print(" ".join(gdal_cmd))
                        subprocess.run(gdal_cmd)
                        sys.exit(0)

                    p = subprocess.run(gdal_cmd, text=True, capture_output=True)
                except (KeyboardInterrupt, Exception) as e:
                    if os.path.exists(fname_projected):
                        os.remove(fname_projected)
                    raise e

            if os.path.exists(fname_projected):
                print("\t", os.path.split(fname_projected)[1], "created.")
            else:
                print("ERROR in projection.")
                print(" ".join(gdal_cmd))
                print("STDOUT:", p.stdout)
                print("STDERR:", p.stderr)
                raise FileNotFoundError(fname_projected)

        # continue

        # If the vertical datum is already in either "msl" or "isl", then don't bother converting it.
        # Just put "_msl" or "_isl" at *end* of the file (rather than _egm2008) and make a symbolic link to it.
        if vdatum_str in ("msl", "isl"):
            fname_converted = os.path.splitext(fname_projected)[0] + "_" + vdatum_str + ".tif"
            fname_converted = fname_converted.replace("/projected/", "/converted/")
            if not os.path.exists(fname_converted):
                os.symlink(fname_projected, fname_converted)
                print("\t", "Symlink", os.path.split(fname_converted)[1], "created.")
            else:
                print("\t", "Symlink", os.path.split(fname_converted)[1], "already exists.")
            return fname_converted, True

        # Then, convert it to EGM2008 vdatum, if possible.
        fname_converted = os.path.splitext(fname_projected)[0] + "_egm2008.tif"
        # Put the converted file into a "converted" directory, from the "projected" directory.
        fname_converted = os.path.join(os.path.dirname(os.path.dirname(fname_converted)), "converted",
                                       os.path.basename(fname_converted))
        if os.path.exists(fname_converted):
            if overwrite:
                os.remove(fname_converted)
        elif os.path.exists(fname_converted.replace("_egm2008.tif", "_msl.tif")):
            fname_converted = fname_converted.replace("_egm2008.tif", "_msl.tif")
            if overwrite:
                os.remove(fname_converted)
        elif os.path.exists(fname_converted.replace("_emg2008.tif", "_isl.tif")):
            fname_converted = fname_converted.replace("_egm2008.tif", "_isl.tif")
            if overwrite:
                os.remove(fname_converted)

        converted_already_existed = True
        if not os.path.exists(fname_converted):
            converted_already_existed = False
            did_error_occur = False

            # temp_folder = os.path.join(etopo_config.etopo_cudem_cache_directory, "temp" + str(i + 100))
            # If we're just testing, change to a different default test folder name to not conflict with another running process.
            temp_folder = os.path.join(etopo_config.etopo_cudem_cache_directory, "temp" + str((i + 1) if test_stop is None else (i+1001)))
            if not os.path.exists(temp_folder):
                os.mkdir(temp_folder)

            if fname.find("hawaii_36_mllw") > -1 or fname.find("hawaii_6_mllw") > -1:
                # It's hanging on the hawaii_36_mllw tile. Not sure why. We know it needs to use the tide-gauge correction, so just skip the next step.
                did_error_occur = True
            elif fname_converted.find("_egm2008.tif") > -1:
                convert_cmd = ["python", "/home/mmacferrin/.local/bin/vertical_datum_convert.py",
                               "-i", vdatum_str,
                               "-o", "3855",
                               "-D", etopo_config.etopo_cudem_cache_directory,
                               "-k",
                               fname_projected,
                               fname_converted]

                print("\t", "Creating", os.path.split(fname_converted)[1])

                try:
                    if test_stop == "CONVERTED":
                        print(" ".join(convert_cmd))
                        subprocess.run(convert_cmd, cwd=temp_folder)
                        sys.exit(0)

                    # print(" ".join(convert_cmd))
                    # p = subprocess.run(convert_cmd, cwd=temp_folder)
                    # import sys
                    # sys.exit(0)
                    # print("TRY AGAIN SILENT.")
                    # We're going to capture the output on this one to check and see if there were any errors after-the-fact.
                    p = subprocess.run(convert_cmd, capture_output=True, text=True, cwd=temp_folder)

                except (KeyboardInterrupt, Exception) as e:
                    if os.path.exists(fname_converted):
                        os.remove(fname_converted)
                    if type(e) == KeyboardInterrupt:
                        raise e

                stdout_text = repr(p.stdout) + repr(p.stderr)
                # Look for a particular vertical_datum_convert.py error message in the stdout + stderr
                # Check both the bash-formatted version and non-formatted version of output string.
                did_error_occur = (stdout_text.find(
                    r"vertical_datum_convert.py: \x1b[31m\x1b[1merror\x1b[m, could not locate [VDATUM] or tss in the region -R".replace(
                        "[VDATUM]", vdatum_str)) > -1) \
                                  or (stdout_text.find(
                    r"vertical_datum_convert.py: error, could not locate [VDATUM] or tss in the region -R".replace(
                        "[VDATUM]", vdatum_str)) > -1)

                # print("STDOUT:")
                # print(p.stdout.splitlines())
                # print("STDERR:")
                # print(repr(p.stderr))
                # Gathering stdout plus stderr to look for errors
                # Using repr here because text includes bash string formatting codes.

            if did_error_occur or fname_converted.find("_msl.tif") > -1:
                # *sigh* this wasn't produced correctly, delete it and try again using a tidal correction.
                print("\t", "Missing grids for '{0}' conversion. Trying tidal correction instead.".format(vdatum_str))
                # Get rid of the file that was already created.
                if fname_converted.find("_egm2008.tif") > -1 and os.path.exists(fname_converted):
                    os.remove(fname_converted)

                # We're not trying to convert to egm2008 now. Just go for mean-sea-level.
                fname_converted = fname_converted.replace("_egm2008.tif", "_msl.tif")

                # stdout_lines = p.stdout.splitlines()
                # # size_x, size_y = [int(x) for x in stdout_lines[0].split()]
                # minx, maxx, miny, maxy = [float(x) for x in stdout_lines[1][len("-R"):].split("/")]
                # stepx, stepy = [float(x) for x in stdout_lines[2].split()]
                ds_temp = gdal.Open(fname_projected, gdal.GA_ReadOnly)
                minx, stepx, _, maxy, _, stepy = ds_temp.GetGeoTransform()
                assert stepy < 0
                x_size, y_size = ds_temp.RasterXSize, ds_temp.RasterYSize
                maxx = minx + (stepx * x_size)
                miny = maxy + (stepy * y_size)
                ds_temp = None
                # The y-step is typically negative. We want all positive values from here on.
                stepy = abs(stepy)

                # The difference between the x- and y-steps should be miniscule
                assert (stepx - stepy) < 1e-8
                # Create a temporary tidal grid based on tide stations within the area.
                temp_tidal_file = os.path.splitext(fname)[0] + "_TEMP_TIDAL.tif"
                waffles_cmd = ["waffles",
                               "-P", "EPSG:4326",
                               "-R", "{0}/{1}/{2}/{3}".format(minx, maxx, miny, maxy),
                               "-M", "nearest:radius1=2:radius2=2",
                               "-O", os.path.splitext(temp_tidal_file)[0],
                               # strip the extension off for waffles, it adds .tif
                               "-E", "{0}/{1}".format(stepx, stepy),
                               "-D", etopo_config.etopo_cudem_cache_directory,
                               "-k",
                               "tides:s_datum={0}:t_datum=msl".format(vdatum_str)]

                try:
                    if test_stop == "TIDAL":
                        print(" ".join(waffles_cmd))
                        p = subprocess.run(waffles_cmd, cwd=temp_folder)
                        sys.exit(0)

                    p = subprocess.run(waffles_cmd, text=True, capture_output=True, cwd=temp_folder)
                except KeyboardInterrupt as e:
                    if os.path.exists(temp_tidal_file):
                        os.remove(temp_tidal_file)
                    raise e

                if os.path.exists(temp_tidal_file):
                    print("\t", os.path.split(temp_tidal_file)[1], "created.")
                else:
                    print("\t", "ERROR creating", os.path.split(temp_tidal_file)[1])
                    print("CMD:", " ".join(waffles_cmd))
                    print(p.stdout)
                    print(p.stderr)
                    # list_of_failed_files.append(fname)
                    return fname, False

                # This creates a .inf file that we no longer need. Get rid of it.
                # subprocess.run(["rm", "*.inf"], text=True, capture_output=True, cwd=os.path.dirname(temp_tidal_file))


                gdal_calc_cmd = ["gdal_calc.py",
                                 "--calc", "A+B",
                                 "-A", fname_projected,
                                 "-B", temp_tidal_file,
                                 "--format", "GTiff",
                                 "--co", "COMPRESS=LZW",
                                 "--co", "TILED=YES",
                                 "--co", "PREDICTOR=2",
                                 "--outfile", fname_converted]

                try:
                    # print(" ".join(gdal_calc_cmd))
                    # subprocess.run(gdal_calc_cmd)
                    subprocess.run(gdal_calc_cmd, text=True, capture_output=True)
                except KeyboardInterrupt as e:
                    if os.path.exists(fname_converted):
                        os.remove(fname_converted)
                    if os.path.exists(temp_tidal_file):
                        os.remove(temp_tidal_file)
                    raise e

                # assert os.path.exists(fname_converted)

                if os.path.exists(temp_tidal_file):
                    os.remove(temp_tidal_file)

            # Get rid of the temp folder & everything in it. This is the most efficient way of doing it, with
            # a single rm -rf command.
            if os.path.exists(temp_folder):
                rm_cmd = ["rm", "-rf", temp_folder]
                subprocess.run(rm_cmd, capture_output=True)

        # If we created the projected filename in this process, delete it (it's not needed any more).
        if fname_projected != fname and os.path.exists(fname_projected):
            os.remove(fname_projected)

        if converted_already_existed and os.path.exists(fname_converted):
            print("\t", os.path.split(fname_converted)[1], "already exists.")
            return fname_converted, True
        elif os.path.exists(fname_converted):
            print("\t", os.path.split(fname_converted)[1], "written.")
            return fname_converted, True
        else:
            # list_of_failed_files.append(fname)
            print("\t", os.path.split(fname_converted)[1], "NOT written.")
            return fname, False

        # return

    def convert_to_wgs84_and_egm2008(self, test_stop = None, overwrite = False, strings_to_omit=None):
        """Make sure all the tiles are in WGS84 lat-lon coords, as well as EGM2008 vertical datum."""
        fnames = sorted(list(set([os.path.join(self.config.source_datafiles_directory, fn) for fn in os.listdir(self.config.source_datafiles_directory) if (re.search(self.config.datafiles_regex, fn) is not None)])))
        # Go ahead and include everything in the "estuarine" folder too. Take care of both datasets at once.
        source_dir_estuarine = self.config.source_datafiles_directory.replace("/thredds/", "/estuarine/")
        fnames = fnames + sorted(list(set([os.path.join(source_dir_estuarine, fn) for fn in os.listdir(source_dir_estuarine) if (re.search(self.config.datafiles_regex, fn) is not None)])))

        failed_files= []

        for i, fname in enumerate(fnames):

            fout, success = self.convert_to_wgs84_and_egm2008_single_tile(fname, i, len(fnames), test_stop=test_stop, overwrite=overwrite)
            if not success:
                failed_files.append(fout)

        if len(failed_files) > 0:
            print("FAILED FILES:")
            for fn in failed_files:
                print(" ", fn)

    def generate_tile_datalist_entries(self, polygon, polygon_crs=None, resolution_s = None, verbose=True, weight=None):
        """Given a polygon (ipmortant, in WGS84/EPSG:4326 coords), return a list
        of all tile entries that would appear in a CUDEM datalist. If no source
        tiles overlap the polygon, return an empty list [].

        This is an overloaded function of etopo_source_dataset::generate_tile_datalist_entries.
        This one assigns a slightly different weight depending upon whether it's a BlueTopo_US5, _US4, or _US3 tile.
        Since the higher numbers (i.e. US5) are higher resolution, we want those to be slightly differnt, so rather than
        7, it'll be 7.3, 7.4, and 7.5, respectively.
        I will need to post-process to make sure the 7.x is changed back to 7 in the end.

        Each datalist entry is a 3-value string, as such:
        [path/filename] [format] [weight]
        In this case, format will always be 200 for raster. Weight will the weights
        returned by self.get_dataset_ranking_score().

        If polygon_crs can be a pyproj.crs.CRS object, or an ESPG number.
        If poygon_crs is None, we will use the CRS
        from the source dataset geopackage.

        If weight is None, use the weight returned by self.get_dataset_ranking_score.
        Else use the weight provided in "weight" for all entries.
        """
        if polygon_crs is None:
            polygon_crs = self.get_crs(as_epsg=False)

        if weight is None:
            weight = self.get_dataset_ranking_score()

        # Do a command-line "waffles -h" call to see datalist options. The datalist
        # type for raster files is "200"
        DTYPE_CODE = 200

        list_of_overlapping_files = self.retrieve_list_of_datafiles_within_polygon(polygon,
                                                                                   polygon_crs,
                                                                                   resolution_s = resolution_s,
                                                                                   verbose=verbose)

        return ["{0} {1} {2}".format(fname, DTYPE_CODE, self.get_tile_weight_from_fname(fname, weight)) \
                for fname in list_of_overlapping_files]

    @staticmethod
    def get_tile_weight_from_fname(fname, orig_weight):
        """looking at the filename, get the year of releawse from it (e.g. "_2006_"), and add it as a decimal to the weight.

        A filename of "adak_1_mhw_2009_msl.tif" with an original weight of 3 will return 3.2009.

        This will ensure that more recent regional tiles are weighted above older ones."""
        # Look for the number just after "BlueTopo_US" and just before "LLNLL" wher L is a capital letter and N is a number.
        try:
            # Look for the 4 digits alone, with underscores on either side.
            # That's the year. We want to put that in the ranking to rank them by year with latest files on top.
            tile_year = int(re.search(r"(?<=_)\d{4}(?=_)", os.path.basename(fname)).group())
        except AttributeError as e:
            print("ERROR FINDING YEAR IN FILENAME:", fname, "(using 1990).")
            tile_year = 1990
        assert 1990 <= tile_year <= 2022
        new_weight = orig_weight + (tile_year / 10000.0)
        return new_weight

    def output_mixmax_csv(self):
        """Go through all the tiles in there, and pull out the min/max value from them. Stick in a CSV"""
        dirname = self.config.source_datafiles_directory.replace("/projected","/converted")
        csvname = os.path.abspath(os.path.join(dirname, "..","..","..", "NOAA_regional_minmax.csv"))

        fnames = sorted(list(set([os.path.join(dirname, fn) for fn in os.listdir(dirname) if (re.search(self.config.datafiles_regex, fn) is not None)])))
        # Go ahead and include everything in the "estuarine" folder too. Take care of both datasets at once.
        source_dir_estuarine = self.config.source_datafiles_directory.replace("/thredds/", "/estuarine/").replace("/projected","/converted")
        fnames = fnames + sorted(list(set([os.path.join(source_dir_estuarine, fn) for fn in os.listdir(source_dir_estuarine) if (re.search(self.config.datafiles_regex, fn) is not None)])))

        mins  = [None] * len(fnames)
        maxes = [None] * len(fnames)
        means = [None] * len(fnames)
        stds  = [None] * len(fnames)

        for i, fname in enumerate(fnames):
            print("{0}/{1} {2}".format(i+1, len(fnames), os.path.basename(fname)))
            dset = gdal.Open(fname, gdal.GA_ReadOnly)
            mins[i], maxes[i], means[i], stds[i] = dset.GetRasterBand(1).GetStatistics(0,1)

        df = pandas.DataFrame(data={'filename': [os.path.basename(fname) for fname in fnames],
                                    'min': mins,
                                    'max': maxes,
                                    'mean': means,
                                    'std': stds})
        df.to_csv(csvname, index=False)
        print(csvname, "written with", len(df), "entries.")

def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Code to convert both horizontal projections and vertical tidal datums into friendly formats, as well as fix other bugs in the NOAA_regional and NOAA_estuarine  datasets.")
    parser.add_argument("-stop", "-s", default=None, help="For debugging purposes, write out stdout for operations on one of the 3 key steps and then exit the program after that step on the first processed file. Steps are 'PROJECTED', 'CONVERTED', or 'TIDAL'.")
    parser.add_argument("--overwrite", "-o", default=False, action="store_true", help="Overwrite existing tiles.")
    parser.add_argument("--minmax", "-m", default=False, action="store_true", help="Compute the min/max/mean/std of each tile and save to a .csv.")
    parser.add_argument("--gpkg", default=False, action="store_true", help="Rebuild the geopackage.")
    return parser.parse_args()

if __name__ == "__main__":
    args = define_and_parse_args()

    noaa = source_dataset_NOAA_regional()

    if args.minmax:
        noaa.output_mixmax_csv()
    elif args.gpkg:
        gpkg_fname = noaa.config._abspath(noaa.config.geopackage_filename)
        if os.path.exists(gpkg_fname):
            os.remove(gpkg_fname)

        noaa.get_geodataframe()
    else:
        noaa.convert_to_wgs84_and_egm2008(test_stop=args.stop, overwrite=args.overwrite)

    # noaa.get_geodataframe()
    # noaa.print_unique_vdatum_ids()
    # noaa.move_estuarine_tiles()
    # noaa.convert_to_gtiff()
