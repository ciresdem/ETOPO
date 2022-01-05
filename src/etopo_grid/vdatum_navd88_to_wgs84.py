# -*- coding: utf-8 -*-

import subprocess
import os
# import osgeo.gdal as gdal
import config
import re
import argparse
# Just make sure it's here.
# import cudem.waffles

# dem_dirpath = "/home/mmacferrin/Research/DATA/DEMs/NCEI/ma_nh_me_deliverables20211007105912/ma_nh_me_deliverables/1_9"
# dem_outpath = "/home/mmacferrin/Research/DATA/DEMs/NCEI/ma_nh_me_deliverables20211007105912/ma_nh_me_deliverables/1_9_wgs84"
# dem_list = sorted( [f for f in os.listdir(dem_dirpath) if os.path.splitext(f)[1].lower() == ".tif"] )

# The old way of doing it via the cudem waffles vdatum module (slow as dirt)
# waffles_vdatum_template = "waffles -M vdatum:ivert=NAVD88:overt=ITRF2008:region={0:d} -O {1:s} -E {2:.14f} -R {3}/{4}/{5}/{6}"
# gdal_calc_template = "gdal_calc.py -A {0:s} -B {1:s} --calc=A+B --outfile={2:s}"

# TWO gdalwarp commands to rule them all.
# Examples:
# gdalwarp -s_srs "+proj=lonlat +datum=NAD83 +ellps=GRS80 +no_defs +geoidgrids=./g2018u0.gtx" -t_srs "+proj=lonlat +datum=WGS84 +no_defs" ncei19_n41x25_w070x00_2021v1.tif tmp_transit.tif
# gdalwarp -s_srs "+proj=lonlat +datum=WGS84 +no_defs +geoidgrids=./transit2g1674.gtx" -t_srs "+proj=lonlat +datum=WGS84 +no_defs" tmp_transit.tif ncei19_n41x25_w070x00_2021v1_wgsg1674.tif
# Templates
gdal_template_1 = 'gdalwarp -s_srs "+proj=lonlat +datum=NAD83 +ellps=GRS80 +no_defs +geoidgrids={0}" -t_srs "+proj=lonlat +datum=WGS84 +no_defs" {1} {2}'
gdal_template_2 = 'gdalwarp -s_srs "+proj=lonlat +datum=WGS84 +no_defs +geoidgrids={0}" -t_srs "+proj=lonlat +datum=WGS84 +no_defs" {1} {2}'

my_config = config.config()

def cmd_smart_split(cmd_str, strip_quotes=True):
    """Split up a string using posix standards where quotes are respected.

    strip_quotes:
        If True (default), remove any "" or '' surrounding an argument.
        If False, leave the quotes there.

    Good for splitting up command-line arguments where a quoted string should be a single argument."""
    items = re.findall(r'(?:[^\s,"]|"(?:\\.|[^"])*")+', cmd_str)
    if strip_quotes:
        items = [item.strip("'\"") for item in items]

    return items

def convert_navd88_to_wgs84(dem_name, dem_out_name=None, verbose=True):
    if dem_out_name is None:
        base, ext = os.path.splitext(dem_name)
        dem_out_name = base + "_wgs84" + ext

    dem_out_dir, dem_out_fname = os.path.split(dem_out_name)
    tempfile_name = os.path.join(dem_out_dir, "tmp_" + dem_out_fname)

    gdal_cmd1 = gdal_template_1.format(my_config.g2018u0_gtx, dem_name, tempfile_name)
    gdal_cmd2 = gdal_template_2.format(my_config.g1674_gtx, tempfile_name, dem_out_name)

    if verbose:
        print(gdal_cmd1)
    subprocess.run(cmd_smart_split(gdal_cmd1), capture_output=not verbose)

    # print("gdalwarp --version")
    # subprocess.run(["gdalwarp", "--version"])

    if verbose:
        print(gdal_cmd2)
    subprocess.run(cmd_smart_split(gdal_cmd2), capture_output=not verbose)

    os.remove(tempfile_name)

    return dem_out_name


def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Utility for converting from NAVD88 elevations to WGS84 ellipsoid elevations.")


# for dem_name in [dem_list[1]]:
#     dem_path = os.path.join(dem_dirpath, dem_name)
#     # File to put the conversion grid into.
#     # conv_grid_path = os.path.join(dem_outpath, "navd2wgs_" + dem_name)
#     tempfile_name = os.path.join(dem_outpath, "tmp_navd2wgs_" + dem_name)
#     base, ext = os.path.splitext(dem_name)
#     # Output file converted to WGS84
#     dem_path_wgs84 = os.path.join(dem_outpath, base + "_wgs84" + ext)

#     # Get metadata from original DEM file (extent, resolution)
#     # ds = gdal.Open(dem_path)

#     print("\n======", dem_path, "======")

#     # print(ds.GetProjection())

#     # print(ds.RasterXSize, ds.RasterYSize)

#     # band = ds.GetRasterBand(1)
#     # dtype = band.DataType
#     # if band.GetMinimum() is None or band.GetMaximum() is None:
#     #     band.ComputeStatistics(0)
#     #     print("Statistics Computed.")
#     # print(band)

#     # print ("[ NO DATA VALUE ] = ", band.GetNoDataValue()) # none
#     # print ("[ MIN ] = ", band.GetMinimum())
#     # print ("[ MAX ] = ", band.GetMaximum())
#     # print(ds.RasterCount)

#     # ulx, xres, xskew, uly, yskew, yres = ds.GetGeoTransform()

#     # lrx = ulx + (ds.RasterXSize * xres)
#     # lry = uly + (ds.RasterYSize * yres)

#     # xmin, xmax = min(ulx, lrx), max(ulx, lrx)
#     # ymin, ymax = min(uly, lry), max(uly, lry)
#     # assert abs(xres) == abs(yres)
#     # resolution = abs(xres)

#     # waffles_command = waffles_vdatum_template.format(4, os.path.splitext(conv_grid_path)[0], resolution, xmin, xmax+(xres/100), ymin, ymax+(yres/100))

#     # gdal_command = gdal_calc_template.format(dem_path, conv_grid_path, dem_path_wgs84)

#     # print(waffles_command)
#     # subprocess.run(waffles_command.split())

#     # print(gdal_command)
#     # subprocess.run(gdal_command.split())

#     gdal_cmd1 = gdal_template_1.format(my_config.g2018u0_gtx, dem_path, tempfile_name)
#     gdal_cmd2 = gdal_template_2.format(my_config.g1674_gtx, tempfile_name, dem_path_wgs84)

#     print(gdal_cmd1)
#     # print(gdal_cmd1.split())
#     # print(cmd_smart_split(gdal_cmd1))
#     # foobar
#     subprocess.run(cmd_smart_split(gdal_cmd1))

#     # print("gdalwarp --version")
#     # subprocess.run(["gdalwarp", "--version"])

#     print(gdal_cmd2)
#     # print(gdal_cmd2.split())
#     subprocess.run(cmd_smart_split(gdal_cmd2))
#     os.remove(tempfile_name)

#     # print("{0}".format(resolution))
#     # print(xmin, xmax, ymin, ymax)

#     # print(ds.GetMetadata())

#     # Create the navd2wgs conversion grid file.

#     # Perform the NAVD88 to WGS calculation using gdal_calc
