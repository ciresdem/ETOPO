# -*- coding: utf-8 -*-
"""Code for generating original compressed TIF grids for the ETOPO project.
This code generates empty grids (with all -99999.0 empty-values) for the ETOPO project to fill in with data.

Grids are generated at the 15", 15° resolution for the global dataset,
as well as the 1", 1° resolution for land and land-adjacent grids.

Also tools for generating outline shapefiles for each one, for mapping purposes.
"""

from osgeo import gdal, osr
import os
import numpy
import re

###############################################################
# Quick code for importing the parent /src/ directory, to access other modules.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
###############################################################

import utils.configfile
import datasets.dataset_geopackage

my_config = utils.configfile.config()
# ETOPO_datatype = gdal.GDT_Float32
# ETOPO_source_datatype = gdal.GDT_Byte

dtypes_dict_numpy = { 'int8'   : numpy.dtype("int8"),
                      'uint8'  : numpy.dtype("uint8"),
                      'int16'  : numpy.dtype("int16"),
                      'uint16' : numpy.dtype("uint16"),
                      'int32'  : numpy.dtype("int32"),
                      'uint32' : numpy.dtype("uint32"),
                      'float32': numpy.dtype("float32"),
                      'float64': numpy.dtype("float64")}

dtypes_dict_gdal  = {'int8'   : gdal.GDT_Byte,
                     'uint8'  : gdal.GDT_Byte,
                     'int16'  : gdal.GDT_Int16,
                     'uint16' : gdal.GDT_UInt16,
                     'int32'  : gdal.GDT_Int32,
                     'uint32' : gdal.GDT_UInt32,
                     'float32': gdal.GDT_Float32,
                     'float64': gdal.GDT_Float64}

def create_empty_tiles(directory,
                       fname_template_tif = r"ETOPO_2022_v1_{0:d}s_{1:s}{2:02d}{3:s}{4:03d}.tif",
                       fname_template_netcdf = None, # = r"ETOPO_2022_v1_{0:d}_{1:s}{2:02d}{3:s}{4:03d}.nc",
                       dtype = "float32",
                       tile_width_deg = 15,
                       resolution_s = 15,
                       ndv = -99999.0,
                       verbose=True,
                       compression_options = ["COMPRESS=DEFLATE", "PREDICTOR=2"],
                       also_write_geopackage = True):
    """Create a set of new empty tiles at either 15 or 1s resolutions.

    For 1s, create for land-tiles-only, using a list based on Copernicus tiles."""

    # Check to see that our input datatype is accepted.
    # First check to make sure our two lookup dictionaries coincide.
    dtypes_accepted = sorted(dtypes_dict_gdal.keys())
    assert dtypes_accepted == sorted(dtypes_dict_gdal.keys())
    # Then check to make sure our given datatypes is included therein.
    dtype_lower = dtype.lower().strip()
    if dtype_lower not in dtypes_accepted:
        raise TypeError("Unknown value for dtype: '{0}'. Accepted dtype values are:".format(dtype), " ".join(dtypes_accepted))
    else:
        dtype = dtype_lower

    # From the with in degrees and the resolution in seconds, find the number of pixels in each dimension
    # NOTE: This is assuming lat/lon geographic coordinate systems here. It should still work if we're doing something else.
    file_dim_size = tile_width_deg * 60 * 60 / resolution_s
    # This should be a whole number division here. If not, warn the user.
    if int(file_dim_size) != file_dim_size:
        raise UserWarning("The tile size {0}-deg is not evenly divislble by the resolution {1}-sec. Results not guaranteed to work well.".format(tile_width_deg, resolution_s))
    # Convert to an int to not break the numpy.zeros command 2 lines down.
    file_dim_size = int(file_dim_size)

    tuple_list = create_list_of_tile_tuples(resolution=resolution_s)
    # fname_metadata_template = config_obj.etopo_metadata_template

    if resolution_s == 60:
        # For the 60s (1-arc-minute) array, we're covering 360*lon by 180*lat
        empty_array = numpy.zeros((int(file_dim_size/2), file_dim_size), dtype = dtypes_dict_numpy[dtype]) + ndv
    else:
        empty_array = numpy.zeros((file_dim_size, file_dim_size), dtype = dtypes_dict_numpy[dtype]) + ndv

    if os.path.splitext(fname_template_tif)[1].lower() == ".tif":
        driver = gdal.GetDriverByName("GTiff")
    else:
        raise NotImplementedError("Unimplemented file type:", os.path.splitext(fname_template_tif)[1])

    resolution_deg = resolution_s/(60.*60.)

    projection = osr.SpatialReference()
    projection.SetWellKnownGeogCS("WGS84")

    for i,tiletuple in enumerate(tuple_list):
        ymin = tiletuple[0]
        xmin = tiletuple[1]
        ymax = ymin + (tile_width_deg if (resolution_s != 60) else tile_width_deg/2)
        # foobar

        # if shapefile_only:
        #     continue

        xmin_letter = "W" if xmin < 0 else "E"
        ymin_letter = "S" if ymin < 0 else "N"

        geotransform = (xmin, resolution_deg, 0, ymax, 0, -resolution_deg)

        fname = os.path.join(directory, fname_template_tif.format(resolution_s,
                                                                  ymin_letter, abs(ymin),
                                                                  xmin_letter, abs(xmin)
                                                                  )
                             )

        file_dim_x = file_dim_size
        file_dim_y = int(file_dim_size/2) if (resolution_s == 60) else file_dim_size

        ds = driver.Create(fname, file_dim_x, file_dim_y, 1, dtypes_dict_gdal[dtype], options=compression_options)
        ds.SetGeoTransform(geotransform)
        ds.SetProjection(projection.ExportToWkt())
        band = ds.GetRasterBand(1)
        band.WriteArray(empty_array)
        band.SetNoDataValue(ndv)
        # band.GetStatistics(0,1)
        ds.FlushCache() # Save to disk
        band = None
        ds = None

        if verbose:
            print(str(i+1) + "/" + str(len(tuple_list)), fname, "written.")

    if also_write_geopackage:
        datasets.dataset_geopackage.ETOPO_Geopackage(1).create_dataset_geopackage()

    return

def get_azerbaijan_1deg_bboxes():
    """Return the (xmin,ymin,xmax,ymax) bounding boxes for 1s tiles over Azerbaijan.

    Copernicus and FABDEM leaves them out, so we need to include them manually here!"""
    bboxes = []
    y = 41
    for x in range(43,50):
        bboxes.append((x,y,x+1,y+1))

    y = 40
    for x in range(43,51):
        bboxes.append((x,y,x+1,y+1))

    y = 39
    for x in range(44,50):
        bboxes.append((x,y,x+1,y+1))

    y = 38
    for x in (45,46,48,49):
        bboxes.append((x,y,x+1,y+1))

    return bboxes

# def create_tile_shapefile(shapefile_name,
#                           tile_boundaries_list,
#                           projection,
#                           config_obj = my_config,
#                           overwrite = True,
#                           verbose=True):
#     """Generate tile outline shapefile given the boundaries of each box.
#     tile_boundaries is a 4-tuple with (xmin, xmax, ymin, ymax)."""

#     driver = ogr.GetDriverByName("ESRI Shapefile")

#     if overwrite and os.path.exists(shapefile_name):
#         driver.DeleteDataSource(shapefile_name)

#     ds = driver.CreateDataSource(shapefile_name)
#     layer = ds.CreateLayer("Tiles", projection, ogr.wkbPolygon)

#     id_field = ogr.FieldDefn("id", ogr.OFTInteger)
#     layer.CreateField(id_field)
#     pct_done_field = ogr.FieldDefn("pct_done", ogr.OFTReal)
#     layer.CreateField(pct_done_field)
#     pct_land_field = ogr.FieldDefn("pct_land", ogr.OFTReal)
#     layer.CreateField(pct_land_field)
#     # TODO: Add other fields here as needed.

#     feature_defn = layer.GetLayerDefn()

#     for i, tile_boundary_tuple in enumerate(tile_boundaries_list):
#         feature = ogr.Feature(feature_defn)
#         xmin, xmax, ymin, ymax = [float(x) for x in tile_boundary_tuple]

#         ring = ogr.Geometry(ogr.wkbLinearRing)
#         ring.AddPoint(xmin, ymin)
#         ring.AddPoint(xmin, ymax)
#         ring.AddPoint(xmax, ymax)
#         ring.AddPoint(xmax, ymin)
#         ring.AddPoint(xmin, ymin)

#         poly = ogr.Geometry(ogr.wkbPolygon)
#         poly.AddGeometry(ring)
#         feature.SetGeometry(poly)

#         feature.SetField("id", i+1)
#         feature.SetField("pct_done", 0.0)
#         feature.SetField("pct_land", -1.0)

#         layer.CreateFeature(feature)

#         feature = None

#     ds = None

#     if verbose:
#         print(len(tile_boundaries_list), "polygons written to", shapefile_name)

#     return

def create_list_of_tile_tuples(resolution = 15,
                               config_obj = my_config,
                               verbose=True):
    """Based on the Copernicus DEM tiles, create a list of 1° x 1° tiles to include.

    Each item will be a 2-tuple containing the (southernmost_latitude, westernmost_logitude).
    Add 1 to each number to get the (norternmost_latitude, easternmost_longitude) bbox coordinates."""
    assert resolution in (1,15,60)

    if resolution == 1:
        # Import the Copernicus source dataset, and get the datafiles directory from it.
        import datasets.CopernicusDEM.source_dataset_CopernicusDEM as copernicus
        copds = copernicus.source_dataset_CopernicusDEM()
        copernicus_dir = os.path.abspath(os.path.join(os.path.split(copds.config._configfile)[0], copds.config.source_datafiles_directory))
        print(copernicus_dir)

        fnames = [f for f in os.listdir(copernicus_dir) if re.search("Copernicus_DSM_COG_10_(\w{3})_00_(\w{4})_00_DEM.tif\Z", f) != None] # os.path.splitext(f)[-1] == ".tif"]
        # fnames.sort()

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

        # Since Copernicus annoyingly omits the 25 tiles over Azerbaijan, include them here.
        azerbaijan_bboxes = get_azerbaijan_1deg_bboxes()
        azerbaijan_tuples = [(bbox[1],bbox[0]) for bbox in azerbaijan_bboxes]
        dem_tuples.extend(azerbaijan_tuples)
        # Sort them out, for good measure.
        dem_tuples.sort()

        if verbose:
            print("{0:,} 1° DEM tiles over land (including Azerbaijan).".format(len(dem_tuples)))


    elif resolution == 15:
        dem_lons = numpy.arange(-180, 180, 15, dtype=int)
        dem_lats = numpy.arange(-90, 90, 15, dtype=int)
        lons_list, lats_list = numpy.meshgrid(dem_lons, dem_lats)
        lons_list = lons_list.flatten()
        lats_list = lats_list.flatten()
        dem_tuples = list(zip(lats_list, lons_list))

    elif resolution == 60:
        # For 60s resolution, we're literlaly just amking one big-assed tile of the whole world. Bottom corner is -180 lon, -90 lat
        dem_tuples = [(-90,-180),]

    else:
        raise ValueError("Unhandles ETOPO resolution: {0}".format(resolution))

    return dem_tuples

if __name__ == "__main__":
    create_empty_tiles(os.path.join(my_config.etopo_empty_tiles_directory, "1s"),
                       tile_width_deg = 1,
                       resolution_s = 1,
                       ndv = my_config.etopo_ndv,
                       also_write_geopackage = True)
