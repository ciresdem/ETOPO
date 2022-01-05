# -*- coding: utf-8 -*-
"""Code for generating original compressed TIF grids for the ETOPO project.
This code generates empty grids (with all -99999.0 empty-values) for the ETOPO project to fill in with data.

Grids are generated at the 15", 15° resolution for the global dataset,
as well as the 1", 1° resolution for land and land-adjacent grids.

Also tools for generating outline shapefiles for each one, for mapping purposes.
"""

from osgeo import gdal, osr, ogr
import os
import numpy

import config

my_config = config.config()
ETOPO_datatype = gdal.GDT_Float32
ETOPO_source_datatype = gdal.GDT_Byte

dtypes_dict_numpy = {'char'   : numpy.dtype,
                     'int8'   : None,
                     'uint8'  : None,
                     'int16'  : None,
                     'uint16' : None,
                     'int32'  : None,
                     'uint32' : None,
                     'int64'  : None,
                     'float16': None,
                     'float32': None,
                     'float64': None}

dtypes_dict_gdal  = {'char'   : None,
                     'int8'   : None,
                     'uint8'  : None,
                     'int16'  : None,
                     'uint16' : None,
                     'int32'  : None,
                     'uint32' : None,
                     'int64'  : None,
                     'float16': None,
                     'float32': None,
                     'float64': None}

def create_empty_tiles(config_obj = my_config,
                       resolution=15,
                       verbose=True,
                       compression_options = ["COMPRESS=DEFLATE", "PREDICTOR=2"],
                       shapefile_only = False):
    """Create a set of new empty tiles at either 15 or 1s resolutions.

    For 1s, create for land-only, using a list based on Copernicus tiles."""
    # Only 1s or 15s resolutions available at this time.
    assert resolution in (1,15)

    if resolution == 1:
        tiledir = config_obj.etopo_1_sec_tile_directory_empty
        filedimsize = int(config_obj.etopo_1_sec_tile_size)
        shapefile_name = config_obj.etopo_1s_shapefile
    elif resolution == 15:
        tiledir = config_obj.etopo_15_sec_tile_directory_empty
        filedimsize = int(config_obj.etopo_15_sec_tile_size)
        shapefile_name = config_obj.etopo_15s_shapefile
    else:
        raise ValueError("Unhandles ETOPO resolution: {0}".format(resolution))

    tuple_list = create_list_of_tile_tuples(resolution=resolution)
    fname_template = config_obj.etopo_filename_template
    fname_metadata_template = config_obj.etopo_metadata_template
    emptyval = config_obj.etopo_emptyval
    metadata_ndv_list = config_obj.etopo_metadata_ndvs
    metadata_dtypes_list = config_obj.etopo_metadata_dtypes
    metadata_band_names_list = config_obj.etopo_metadata_band_names

    empty_array = numpy.zeros((filedimsize, filedimsize), dtype = numpy.float32) + emptyval
    empty_metadata_array = numpy.zeros((filedimsize, filedimsize), dtype = numpy.int8) + source_emptyval

    driver = gdal.GetDriverByName("GTiff")

    resolution_deg = resolution/(60*60)

    projection = osr.SpatialReference()
    projection.SetWellKnownGeogCS("WGS84")

    boundaries_list = [None] * len(tuple_list)

    for i,tiletuple in enumerate(tuple_list):
        ymin = tiletuple[0]
        ymax = ymin + resolution # NOTE: This only works because the resolution in seconds (15") is equal to the tile size in degrees (15°)
                                 # If that assumption changes, change this part of the code.
        xmin = tiletuple[1]
        xmax = xmin + resolution # NOTE: This only works because the resolution in seconds (15") is equal to the tile size in degrees (15°)

        boundaries_list[i] = (xmin, xmax, ymin, ymax)

        if shapefile_only:
            continue

        xmin_letter = "W" if xmin < 0 else "E"
        xmax_letter = "W" if xmax < 0 else "E"
        ymin_letter = "S" if ymin < 0 else "N"
        ymax_letter = "S" if ymax < 0 else "N"

        geotransform = (xmin, resolution_deg, 0, ymin, 0, resolution_deg)

        fname = os.path.join(tiledir, fname_template.format(ymin_letter, abs(ymin),
                                                            ymax_letter, abs(ymax),
                                                            xmin_letter, abs(xmin),
                                                            xmax_letter, abs(xmax),
                                                            resolution))

        ds = driver.Create(fname, filedimsize, filedimsize, 1, ETOPO_datatype, options=compression_options)
        ds.SetGeoTransform(geotransform)
        ds.SetProjection(projection.ExportToWkt())
        band = ds.GetRasterBand(1)
        band.WriteArray(empty_array)
        band.SetNoDataValue(emptyval)
        # band.GetStatistics(0,1)
        ds.FlushCache() # Save to disk
        band = None
        ds = None

        if verbose:
            print(fname, "written.")

        fname_metadata = os.path.join(tiledir, fname_metadata_template.format(
                                                            ymin_letter, abs(ymin),
                                                            ymax_letter, abs(ymax),
                                                            xmin_letter, abs(xmin),
                                                            xmax_letter, abs(xmax),
                                                            resolution))


        ds2 = driver.Create(fname_metadata, filedimsize, filedimsize, 1, options=compression_options)
        ds2.SetGeoTransform(geotransform)
        ds2.SetProjection(projection.ExportToWkt())
        band2 = ds2.GetRasterBand(1)
        band2.WriteArray(empty_source_array)
        band2.SetNoDataValue(source_emptyval)
        # band2.GetStatistics(0,1)
        ds2.FlushCache() # Save to disk
        band2 = None
        ds2 = None

        if verbose:
            print(fname_source, "written.")

    # Now, generate a shapefile with teh boundaries list.
    create_tile_shapefile(shapefile_name,
                          boundaries_list,
                          projection,
                          config_obj = config_obj,
                          verbose = verbose)

    return


def create_tile_shapefile(shapefile_name,
                          tile_boundaries_list,
                          projection,
                          config_obj = my_config,
                          overwrite = True,
                          verbose=True):
    """Generate tile outline shapefile given the boundaries of each box.
    tile_boundaries is a 4-tuple with (xmin, xmax, ymin, ymax)."""

    driver = ogr.GetDriverByName("ESRI Shapefile")

    if overwrite and os.path.exists(shapefile_name):
        driver.DeleteDataSource(shapefile_name)

    ds = driver.CreateDataSource(shapefile_name)
    layer = ds.CreateLayer("Tiles", projection, ogr.wkbPolygon)

    id_field = ogr.FieldDefn("id", ogr.OFTInteger)
    layer.CreateField(id_field)
    pct_done_field = ogr.FieldDefn("pct_done", ogr.OFTReal)
    layer.CreateField(pct_done_field)
    pct_land_field = ogr.FieldDefn("pct_land", ogr.OFTReal)
    layer.CreateField(pct_land_field)
    # TODO: Add other fields here as needed.

    feature_defn = layer.GetLayerDefn()

    for i, tile_boundary_tuple in enumerate(tile_boundaries_list):
        feature = ogr.Feature(feature_defn)
        xmin, xmax, ymin, ymax = [float(x) for x in tile_boundary_tuple]

        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(xmin, ymin)
        ring.AddPoint(xmin, ymax)
        ring.AddPoint(xmax, ymax)
        ring.AddPoint(xmax, ymin)
        ring.AddPoint(xmin, ymin)

        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        feature.SetGeometry(poly)

        feature.SetField("id", i+1)
        feature.SetField("pct_done", 0.0)
        feature.SetField("pct_land", -1.0)

        layer.CreateFeature(feature)

        feature = None

    ds = None

    if verbose:
        print(len(tile_boundaries_list), "polygons written to", shapefile_name)

    return

def copy_empty_tiles_to_active_dir(resolution=15, config_obj = my_config, overwrite=True):
    # TODO: finish
    pass

def create_list_of_tile_tuples(resolution = 15,
                               config_obj = my_config,
                               verbose=True):
    """Based on the Copernicus DEM tiles, create a list of 1° x 1° tiles to include.

    Each item will be a 2-tuple containing the (southernmost_latitude, westernmost_logitude).
    Add 1 to each number to get the (norternmost_latitude, easternmost_longitude) bbox coordinates."""
    assert resolution in (1,15)

    if resolution == 1:
        copernicus_dir = my_config.copernicus_dem_data_dir
        print(copernicus_dir)
        fnames = [f for f in os.listdir(copernicus_dir) if os.path.splitext(f)[-1] == ".tif"]
        fnames.sort()

        dem_tuples = [None] * len(fnames)
        if verbose:
            print("{0:,} 1° DEM tiles over land.".format(len(dem_tuples)))
        for i,fname in enumerate(fnames):
            # Make sure the file name is formatted exactly how we're expecting.
            assert len(fname) == len("Copernicus_DSM_COG_10_N70_00_E020_00_DEM.tif")
            assert fname.find("Copernicus_DSM_COG_10_") == 0
            assert fname[22] in ("N","S")
            lat_sign = 1 if fname[22] == "N" else -1
            assert fname[29] in ("E","W")
            lon_sign = 1 if fname[29] == "E" else -1


            dem_tuples[i] = (int(fname[23:25]) * lat_sign,
                             int(fname[30:33]) * lon_sign)

    elif resolution == 15:
        dem_lons = numpy.arange(-180, 180, 15, dtype=int)
        dem_lats = numpy.arange(-90, 90, 15, dtype=int)
        lons_list, lats_list = numpy.meshgrid(dem_lons, dem_lats)
        lons_list = lons_list.flatten()
        lats_list = lats_list.flatten()
        dem_tuples = list(zip(lats_list, lons_list))
    else:
        raise ValueError("Unhandles ETOPO resolution: {0}".format(resolution))

    return dem_tuples

if __name__ == "__main__":
    create_empty_tiles(resolution=1, shapefile_only = False)
    # create_list_of_tile_tuples(resolution=1)
