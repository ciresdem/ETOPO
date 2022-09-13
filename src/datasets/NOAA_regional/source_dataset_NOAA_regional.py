# -*- coding: utf-8 -*-

"""Source code for the NOAA_regional ETOPO source dataset class."""

import os
import subprocess
import re
import sys
from osgeo import gdal
import pyproj

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
                        "-co", "PREDICTOR=3",
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

    def convert_to_wgs84_and_egm2008(self, overwrite = False):
        """Make sure all the tiles are in WGS84 lat-lon coords, as well as EGM2008 vertical datum."""
        fnames = sorted([os.path.join(self.config.source_datafiles_directory, fn) for fn in os.listdir(self.config.source_datafiles_directory) if re.search(self.config.datafiles_regex, fn) is not None and fn.find("_egm2008.tif") == -1 and fn.find("_epsg4326") == -1])
        # Go ahead and include everything in the "estuarine" folder too. Take care of both datasets at once.
        source_dir_estuarine = self.config.source_datafiles_directory.replace("/thredds/", "/estuarine/")
        fnames = fnames + sorted([os.path.join(source_dir_estuarine, fn) for fn in os.listdir(source_dir_estuarine) if re.search(self.config.datafiles_regex, fn) is not None and fn.find("_egm2008.tif") == -1 and fn.find("_epsg4326") == -1])

        list_of_failed_files = []

        for i, fname in enumerate(fnames):
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
                print("\t" + os.path.split(fname)[1], "set to ESPG:4326")

            proj = pyproj.Proj(wkt)
            epsg = proj.crs.to_epsg()

            # Check to make sure the image's geotransform is in the bounds of ESPG: 4326. Some are not.
            if epsg == 4326:
                # SHIT, I corrected the GT's of these two akutan_ files by 180, not 360. Fix that shit.
                # if os.path.split(fname)[1].find("akutan_83_") > -1 or \
                #     os.path.split(fname)[1].find("akutan_8_") > -1:
                #     ds = None
                #     ds = gdal.Open(fname, gdal.GA_Update)
                #     gt = ds.GetGeoTransform()
                #     new_gt = (gt[0] - 180, gt[1], gt[2], gt[3], gt[4], gt[5])
                #     ds.SetGeoTransform(new_gt)
                #     print("\t", os.path.split(fname)[1], "GT corrected from", gt, "to", new_gt)
                #     continue

                gt = ds.GetGeoTransform()
                xmin, ymax = gt[0], gt[3]
                assert (-90 <= ymax <=90)
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
                vdatum_str = re.search("(?<=\w_)[A-Za-z0-9]+(?=_[\dm]+\.tif\Z)", os.path.split(fname)[1]).group()
            except AttributeError:
                print(os.path.split(fname)[1])
                break
            if vdatum_str not in ("isl","mhhw","mhw","mllw","msl","navd88","prvd02"):
                vdatum_str = "mllw"

            # If it's not already in EPSG 4326, convert it to EPSG 4326
            if epsg == 4326:
                fname_projected = fname
            else:
                fname_projected = os.path.splitext(fname)[0] + "_espg4326.tif"
                if os.path.exists(fname_projected):
                    if overwrite:
                        os.remove(fname_projected)

                if not os.path.exists(fname_projected):
                    gdal_cmd = ["gdalwarp",
                                "-t_srs", "ESPG:4326",
                                "dstnodata", "-99999",
                                "-co", "COMPRESS=LZW",
                                "-co", "PREDICTOR=3",
                                fname,
                                fname_projected]

                    print(" ".join(gdal_cmd))
                    try:
                        subprocess.run(gdal_cmd)
                    except (KeyboardInterrupt, Exception) as e:
                        if os.path.exists(fname_projected):
                            os.remove(fname_projected)
                        raise e

            # Then, convert it to EGM2008 vdatum, if possible.
            fname_converted = os.path.splitext(fname_projected)[0] + "_egm2008.tif"
            if os.path.exists(fname_converted):
                if overwrite:
                    os.remove(fname_converted)

            converted_already_existed = True
            if (not os.path.exists(fname_converted)) and os.path.exists(fname_projected):
                converted_already_existed = False
                convert_cmd = ["python", "/home/mmacferrin/.local/bin/vertical_datum_convert.py",
                               "-i", vdatum_str,
                               "-o", "3855",
                               "-D", etopo_config.etopo_cudem_cache_directory,
                               "-k",
                               fname_projected,
                               fname_converted]

                print(" ".join(convert_cmd))
                try:
                    p = subprocess.run(convert_cmd)
                except (KeyboardInterrupt, Exception) as e:
                    if os.path.exists(fname_converted):
                        os.remove(fname_converted)
                    if type(e) == KeyboardInterrupt:
                        raise e

            print("{0}/{1} {2}".format(i+1, len(fnames), os.path.split(fname_converted)[1]), end="")
            if converted_already_existed:
                assert os.path.exists(fname_converted)
                print(" already exists.\n")
            elif p.returncode == 0:
                assert os.path.exists(fname_converted)
                print(" written.\n")
            else:
                list_of_failed_files.append(fname)
                print(" NOT written.\n")

if __name__ == "__main__":
    noaa = source_dataset_NOAA_regional()
    # noaa.print_unique_vdatum_ids()
    # noaa.move_estuarine_tiles()
    # noaa.convert_to_gtiff()
    noaa.convert_to_wgs84_and_egm2008()