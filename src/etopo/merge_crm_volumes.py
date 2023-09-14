
import argparse
import geopandas
import os
# import numpy
from osgeo import gdal
import subprocess
import shapely.ops
import re

import import_parent_dir; import_parent_dir.import_src_dir_via_pythonpath()
import utils.traverse_directory
import utils.configfile
etopo_config = utils.configfile.config()

def sort_crm_tiles_into_volume_dict(sids=False) -> dict:
    """Retrieve all the CRM tile filenames, and sort them into a dictionary, with the crm volume nunmber [1-5] as
    key and a list of (filename, polygon) tuples as values."""
    etopo_finished_tiles_directory = os.path.join(etopo_config._abspath(etopo_config.etopo_finished_tiles_directory),
                                                  "1s")
    # Get the latest YYYY.MM.DD year in that directory.
    crms_latest_date_dir = os.path.join(etopo_finished_tiles_directory,
                                        sorted([dname for dname in os.listdir(etopo_finished_tiles_directory) \
                                                if (os.path.isdir(os.path.join(etopo_finished_tiles_directory, dname)) and
                                                   (re.search("\A\d{4}\.\d{2}\.\d{2}\Z", dname) is not None))])[-1])
    crm_tiles_list = utils.traverse_directory.list_files(crms_latest_date_dir,
                                                         regex_match= \
            (r'ncei1_n\d{2}x00_w\d{3}x00_2023v1(_\d{4}\.\d{2}\.\d{2})?_sid\.tif' \
            if sids else r'ncei1_n\d{2}x00_w\d{3}x00_2023v1(_\d{4}\.\d{2}\.\d{2})?\.tif'))

    crm_vol_shps_dir = etopo_config._abspath(etopo_config.crm_volume_shapefiles_dir)
    crm_vol_shps = sorted(
        [os.path.join(crm_vol_shps_dir, fn) for fn in os.listdir(crm_vol_shps_dir) if os.path.splitext(fn)[1] == ".shp"])

    # The tiles are named A-V (latitude 46 - 24), 07-40 (longitude -99 to -66).
    # Use these to translate the A_40 names to lat,lon pairs to find the correct CRM tiles.
    lat_dict = dict(zip([*"ABCDEFGHIJKLMNOPQRSTUV"], range(46, 23, -1)))
    lon_dict = dict(zip(range(7, 41), range(-99, -65)))

    output_dict = dict()
    for i in range(5):
        gdf = geopandas.read_file(crm_vol_shps[i])
        print(i + 1, crm_vol_shps[i], len(gdf), "tiles.")
        # print(gdf, "\n")
        polygons = gdf.geometry.tolist()
        names = gdf.NAME.tolist()
        list_of_crm_tiles = [None]*len(gdf)
        for j, name, poly in zip(range(len(gdf)), names, polygons):
            name_lat = lat_dict[name[0]]
            name_lon = lon_dict[int(name[2:])]
            # Make sure the upper-right corner we derived from the name coincides with the upper-right of the polygon.
            assert name_lat == poly.exterior.coords.xy[1][0]
            assert name_lon == poly.exterior.coords.xy[0][0]

            # print(names[0], lat_dict[names[0][0]], lon_dict[int(names[0][2:])], polygons[0], polygons[0].exterior.coords.xy[0][0])
            # Get the name of the ncei1_... CRM tiles that corresponds to this geometry.
            crm_name = os.path.join(crms_latest_date_dir, etopo_config.crm_1deg_filename_template.format("n",
                                                                                                         abs(name_lat),
                                                                                                         "w",
                                                                                                         abs(name_lon),
                                                                                                         2023, 1))

            # If the crm files seem to have date strings after them, include that in the name.
            if crm_name not in crm_tiles_list:
                if re.search(r"_\d{4}\.\d{2}\.\d{2}", crm_tiles_list[0]) is not None:
                    base, ext = os.path.splitext(crm_name)
                    datestr = re.search(r"_\d{4}\.\d{2}\.\d{2}", crm_tiles_list[0]).group()
                    crm_name = base + datestr + ext
                else:
                    datestr = " "

                # Also, put the "_sid" in the right place., and look in the sid folder.
                if sids:
                    crm_name = crm_name.replace("_sid", "") \
                        .replace(".tif", "_sid.tif") \
                        .replace(datestr[1:] + "/ncei", datestr[1:] + "/sid/ncei")


            # Make sure it exists in the CRM files on disk.
            # print(crm_name)
            # print(crm_tiles_list)
            # print(re.search(r"_\d{4}\.\d{2}\.\d{2}", crm_tiles_list[0]) is not None)
            assert crm_name in crm_tiles_list and os.path.exists(crm_name)
            # Add it to the list.
            list_of_crm_tiles[j] = (crm_name, poly)

        output_dict[i + 1] = list_of_crm_tiles

    return output_dict

def merge_crm_volumes(sids=False):
    crm_tiles_dict = sort_crm_tiles_into_volume_dict(sids=sids)
    for crm_i in sorted(crm_tiles_dict.keys()):
        crm_tile_pairs = crm_tiles_dict[crm_i]
        crm_tile_names = [tp[0] for tp in crm_tile_pairs]
        crm_tile_polys = [tp[1] for tp in crm_tile_pairs]
        crm_vol_filename = os.path.join(os.path.dirname(crm_tile_names[0]), etopo_config.crm_volume_name_template.format(crm_i, 2023))

        # If doing sids, make sure we output a _sid labeled file.
        if sids:
            base, ext = os.path.splitext(crm_vol_filename)
            crm_vol_filename = base + "_sid" + ext

        print("==", crm_i, crm_vol_filename, "==")

        # lats_all = []
        # lons_all = []
        # # Get the lat and lon bounds of the merged CRM
        # for poly in crm_tile_polys:
        #     lons_all.extend(poly.exterior.coords.xy[0])
        #     lats_all.extend(poly.exterior.coords.xy[1])
        # max_lat = numpy.max(lats_all)
        # min_lat = numpy.min(lats_all)
        # max_lon = numpy.max(lons_all)
        # min_lon = numpy.min(lons_all)
        # print("   ", max_lat, min_lon, min_lat, max_lon)
        # print("   ", crm_tile_names[0])

        # Get metadata from the first CRM tile in the list.
        ds = gdal.Open(crm_tile_names[0], gdal.GA_ReadOnly)
        ndv = ds.GetRasterBand(1).GetNoDataValue()
        metadata = ds.GetMetadata()

        if sids:
            # If we're doing sids, update the metadata to not list is as elevation.
            metadata['TIFFTAG_IMAGEDESCRIPTION'] = 'SourceData ID Tag (SID)'
            del metadata['AREA_OR_POINT']

        # print(metadata)
        # metadata_tags = []
        # for mtag_key in metadata.keys():
        #     metadata_tags.extend(["-mo", "{0}={1}".format(mtag_key,
        #                                                   metadata[mtag_key] if \
        #                                                       (metadata[mtag_key].find(" ") == -1) else \
        #                                                       ("'" + metadata[mtag_key] + "'"))])


        if os.path.exists(crm_vol_filename):
            print("Removing old", os.path.basename(crm_vol_filename))
            os.remove(crm_vol_filename)

        # Merge the tiles.
        # Only use "PREDICTOR=3" if we're working with floating point tiles (not the SIDs)
        # It's not a valid argument with integer arrays.
        gdalmerge_cmd = ["gdal_merge.py",
                         "-o", crm_vol_filename,
                         "-of", "GTiff",
                         "-co", "COMPRESS=DEFLATE",
                         "-co", "INTERLEAVE=BAND"] + \
                        ([] if sids else ["-co", "PREDICTOR=3"]) + \
                        ["-co", "TILED=YES",
                         "-n", str(ndv),
                         "-a_nodata", str(ndv),
                         "-init", str(ndv)
                         ] + crm_tile_names

        print(" ".join(gdalmerge_cmd))
        subprocess.run(gdalmerge_cmd)

        print("Setting metadata and computing statistics...", end=" ")
        # Add the metadata, at the end.
        vol_ds = gdal.Open(crm_vol_filename, gdal.GA_Update)
        vol_ds.SetMetadata(metadata)
        # Also, set the band statistics.
        vol_ds.GetRasterBand(1).GetStatistics(0, 1)
        del vol_ds
        print("Done.\n")

    return

def merge_crm_outline_polygons(include_hi_pr=True):
    """Read the 5 tile shapefiles and merge the polygons.

    Save as a Geopackage in WGS84 coordinates (ignore input parameters).

    If include_hi_pr, include the outlines of the Hawaii and Puerto Rico tiles as well."""
    crm_vol_shps_dir = etopo_config._abspath(etopo_config.crm_volume_shapefiles_dir)
    crm_vol_shps = sorted(
        [os.path.join(crm_vol_shps_dir, fn) for fn in os.listdir(crm_vol_shps_dir) if os.path.splitext(fn)[1] == ".shp"])

    # Get the merged geometry polygons
    output_polys = []
    for i in range(5):
        gdf = geopandas.read_file(crm_vol_shps[i])
        output_polys.append( shapely.ops.unary_union(gdf.geometry.tolist()) )

    # Get the CRM Volume IDs (1 thru 5)
    output_ids = list(range(1,6))

    if include_hi_pr:
        hi_gpkg = geopandas.read_file("/home/mmacferrin/Research/DATA/DEMs/CUDEM/data/Hawaii/CUDEM_Hawaii.gpkg")
        output_polys.append(shapely.ops.unary_union(hi_gpkg.geometry.tolist()))
        output_ids.append(9)

        pr_gpkg = geopandas.read_file("/home/mmacferrin/Research/DATA/DEMs/CUDEM/data/Puerto_Rico/CUDEM_PuertoRico.gpkg")
        output_polys.append(shapely.ops.unary_union(pr_gpkg.geometry.tolist()))
        output_ids.append(10)

    new_gdf = geopandas.GeoDataFrame(data={"crm_volume": output_ids, "geometry": output_polys},
                                     geometry="geometry",
                                     crs=4326)

    # Write out to disk
    out_gpkg_name = os.path.join(crm_vol_shps_dir, "crm_outlines.gpkg")
    if os.path.exists(out_gpkg_name):
        os.remove(out_gpkg_name)
    new_gdf.to_file(out_gpkg_name)
    print(out_gpkg_name, "written with {0} outlines".format(len(new_gdf)))
    return

def define_and_parse_args():
    parser = argparse.ArgumentParser("Merge the CRM 1* tiles in 5 volumes.")
    parser.add_argument("--outlines", action="store_true", default=False,
                        help="Merge the polygons from the shapefiles into an outline Geopackage.")
    parser.add_argument("--sids", action="store_true", default=False,
                        help="Merge the SIDs rather than the DEMs.")
    return parser.parse_args()

if __name__ == "__main__":
    args = define_and_parse_args()
    if args.outlines:
        merge_crm_outline_polygons()
    else:
        merge_crm_volumes(sids=args.sids)