# -*- coding: utf-8 -*-

"""granule_shapefiles.py -- Utility functions for dealing with and processing\
    line shapefiles of icesat-2 granules.

Created by : Mike MacFerrin, NOAA NCEI
Created on : 2022.02.04
"""

import os
from osgeo import osr, ogr
import numpy

import atl_granules

def output_shapefile_of_granules(shapefile_name,
                                 granule_ids_or_dir,
                                 bbox=None,
                                 bbox_converter = None,
                                 verbose=True):
    # It's fine if the output is still in EPSG 4326 (lats/lons)
    """Create a shapefile of path points from a list of granules, or from a photon_dataframe.

    The projection will be WGS84, ESPG 4326.

    Bbox should be (xmin,ymin,xmax,ymax). Can be in any projection.
    If bbox is not WGS84, then a bbox_converter object should be provided, of type osr.CoordinateTransoformation,
    in order to transform the lat/lons into projected x/y coordaintes for subsetting."""
    ## Get all the X,Y positions of the ATL08 data. Export a shapefile.
    # dtype = numpy.dtype([('latitude', numpy.float64),
    #                      ('longitude', numpy.float64)])

    beams = ['gt1l','gt1r','gt2l','gt2r','gt3l','gt3r']
    # bbox=(-79.03,25.78,-76.91,26.95)

    track_dict = {} # d[granule_id][beam] --> lat/lon array

    if (type(granule_ids_or_dir) in (tuple, list)):
        if (len(granule_ids_or_dir) >= 1) and (type(granule_ids_or_dir[0]) is str):
            atl08_granule_ids = granule_ids_or_dir
        else:
            print("Unhandled type for argument 'granule_ids_or_dir'... must either be a list or tuple of strings for filenames, or a string to a diretory.")
    elif os.path.isdir(granule_ids_or_dir):
        fnames = os.listdir(granule_ids_or_dir)
        atl08_granule_ids = [os.path.join(granule_ids_or_dir, fn) for fn in fnames if ((fn.upper().find("ATL08") >= 0) and (os.path.splitext(fn)[1].lower() == ".h5"))]
    else:
        raise ValueError(f"Unknown value for 'granule_ids_or_dir': {granule_ids_or_dir}")

    for gid in atl08_granule_ids:
        atl08 = atl_granules.ATL08_granule(gid)

        for beam in beams:

            lons, lats = atl08.get_coordinates(beam=beam, include_heights=False, warn_if_not_present=True)
            if (len(lats) == 0) or (len(lons) == 0):
                continue

            # Filter out lons/lats:
            # Handle alternate bounding boxes in other coordinate reference systems.
            # Use translated points if a bbox_converter is provided.
            if bbox != None:
                xmin, ymin, xmax, ymax = bbox

                if bbox_converter:
                    # Convert all the lat/lon coordinates into the new projection.
                    points = bbox_converter.TransformPoints( list(zip(lons, lats)) )
                    p_x = numpy.array([p[0] for p in points])
                    p_y = numpy.array([p[1] for p in points])
                    # Then submset the bounding box by the projected coordinates.
                    bbox_mask = (p_x >= xmin) & (p_x < xmax) & (p_y >= ymin) & (p_y < ymax)

                else:
                    # Else just filter out by lats & lons
                    bbox_mask = (lons >= xmin) & (lons <= xmax) & (lats >= ymin) & (lats <= ymax)

                lons = lons[bbox_mask]
                lats = lats[bbox_mask]
            assert len(lons) == len(lats)

            if verbose:
                print(gid, beam, len(lons), "ATL08 points.")

            if not(gid in track_dict):
                track_dict[gid] = {}

            if len(lons) >= 2:
                track_dict[gid][beam] = (lons, lats)

    # Create a shapefile
    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(shapefile_name)
    sr = osr.SpatialReference()   # create spatial reference object
    sr.ImportFromEPSG(4326)       # set it to EPSG:4326
    layer = ds.CreateLayer('ATL08', sr, ogr.wkbLineString)
    layer.CreateField(ogr.FieldDefn('beam_id', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('granule_id', ogr.OFTInteger64))

    defn = layer.GetLayerDefn()

    for gid in track_dict.keys():
        gid1, gid2 = atl08.granule_id_to_intx2(granule_id=gid)

        for beam in track_dict[gid].keys():
            beam_num = atl08.beam_name_to_int(beam)
            lons, lats = track_dict[gid][beam]
            assert len(lons) == len(lats)
            if verbose:
                print(gid, beam, len(lons), "points.")

            feature = ogr.Feature(defn)
            wkt = "LINESTRING(" + ",".join(["{0} {1}".format(lon, lat) for lon, lat in zip(lons, lats)]) + ")"
            geom = ogr.CreateGeometryFromWkt(wkt)
            feature.SetGeometry(geom)
            feature.SetField("beam_id", beam_num)
            feature.SetField("granule_id", gid1)
            layer.CreateFeature(feature)

            feature = None
            geom = None

    # Save the dataset
    ds = None
    layer = None
    if verbose:
        print(shapefile_name, "written.")
    return shapefile_name
