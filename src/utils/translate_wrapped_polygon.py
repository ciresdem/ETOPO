# -*- coding: utf-8 -*-

"""translate_polygon.py -- code for translating polygons and bounding-boxes
   from projected coordaintes into lat/lon geographic coordaintes. This is
   ostensibly (and tyipcally) a simple reprojection, but can get tricky when
   polygons & bounding boxes overlap the longitude cutoff of +/- 180 degrees.
   In that particular case, we need to "split" the polygon into two disjoint
   polygons. Also, some transformations cause significant warping near the edges
   of the projection (such as polar stereo, etc), so adding more granularity to
   the polygons will help with this.

Author: mmacferrin
Created: 2022.04.12
"""

from shapely import geometry, wkt
from osgeo import gdal
import pyproj
import numpy

# These are all in lon/lat (ESPG:4326) coordinates
NPOLE_point = geometry.Point(0,90)
SPOLE_point = geometry.Point(0,-90)
WORLD_POLYGON = geometry.Polygon([(-180, -90), (-180, 90), (180, 90), (180, -90), (-180, -90)])

# WGS84 lat/lon projection
PROJ_WGS84 = pyproj.CRS.from_epsg(4326)

# TEST POLYGONS
# PTEST1 - wraps around the pole. Works for both north and south.
PTEST1 = geometry.Polygon([(-100,-100),(-100,100),(100,100),(100,-100),(-100,-100)])
PTEST_proj_N = pyproj.CRS.from_epsg(3413)
PTEST_proj_S = pyproj.CRS.from_epsg(3976)

# PTEST2_N is an Arctic DEM tile that wraps around the E-W longitude boundary.
PTEST2_N = geometry.Polygon([(-3100000, 3000000), (-3100000, 3100000), (-3000000, 3100000), (-3000000, 3000000), (-3100000, 3000000)])
# PTEST3_S is a REMA tile that wraps around the E-W longitude boundary.
PTEST3_S = geometry.Polygon([(-100000, -1200000), (-100000, -1100000), (0, -1100000), (0, -1200000), (-100000, -1200000)])

def is_polygon_split_at_180(polygon, polygon_crs, tolerance_degrees=30):
    """Return True if a polygon overlaps the boundary of +/- 180 degrees longitude.

    This is technically ambiguous, and relies in the assumption that the polygon
    corner points should be fairly close (within some tolerance) of both the East and West
    boundary lines of the earth. Polygons that truly span the majority of the way around
    the Earth would violate this assumption, as they could be close to the left-and-right
    boundary while not actually crossing it. Here we assume that polygons that intersect
    the E-W boundary line would not extend more than 30 degrees to either side of that line. Use a
    different value for "tolerance_degrees" (must be between 0 and 180) to adjust this assumption.

    Return False otherwise.

    This only analyzes the "outer ring" of a polygon. It does not look at inner rings.
    """
    crs_to_wgs84, wgs84_to_crs = create_transformers(polygon_crs)

    polygon_points = list(polygon.exterior.coords)

    wgs84_polygon_points = [crs_to_wgs84.transform(*p) for p in polygon_points]

    # Test: Does it contain x-values both within the 'tolerance_degrees' on both the left and right sides?
    is_close_to_left = numpy.any([p[0] <= (-180 + tolerance_degrees) for p in wgs84_polygon_points])
    is_close_to_right = numpy.any([p[0] >= (180 - tolerance_degrees) for p in wgs84_polygon_points])

    return is_close_to_left and is_close_to_right

def does_polygon_overlap_a_pole(polygon, polygon_crs):
    """Return "N" or "S" if a polygon fully contains the North or South pole (respectively).
    Return "NI" or "SI" if the pole intersects the boundary of the polygon but does not lie fully within it.

    Am assuming for now that no polygon wraps over both the north *and* south poles.
    Hypotheticaqlly I supposed that could be true in some projections but I'm not worrying about it here.

    Return False otherwise.
    """
    crs_to_wgs84, wgs84_to_crs = create_transformers(polygon_crs)

    npole_projected = wgs84_to_crs.transform(*NPOLE_point.coords[0])
    if npole_projected.within(polygon):
        return "N"

    spole_projected = wgs84_to_crs.transform(*SPOLE_point.coords[0])
    if spole_projected.within(polygon):
        return "S"

    if npole_projected.intersects(polygon):
        return "NI"

    if spole_projected.intersects(polygon):
        return "SI"

    return False

def get_coords_and_crs_from_raster_image(raster_img):
    """Read a geoTiff raster image, and return
    1) The bounding box around the image, and
    2) The coordinate reference system (crs) in well-known-text (wkt) format.
    """
    # TODO: Finish

# def get_lonlat_bbox_from_raster_image(raster_img, side_segments_when_projected=10):
#     """Read a geotiff, presumably from a projected coordinate system, and return
#     a lat-lon bounding box.

#     side_segments_when_projected is the number of segment to split each side of the
#     bounding-box into before reprojecting. Caveat: If no projection occurs
#     (if the polygon was in geographic coordaintes already), then this argument is
#     ignored and corner coordinates only are returned.
#     Dividing into segments helps near the edge-cases where polygons can be heavily
#     warped and using only corner-coordaintes will cause a poor translation between
#     projections. Sub-dividing the polygon edges into finer granularity helps with that.
#     """
#     # TODO: Finish

def create_transformers(source_crs, dest_crs = pyproj.CRS.from_epsg(4326)):
    """Create 2 transformers and return them:
        1. From the source_crs to the dest_crs,
        and
        2. From the dest_crs to the source_crs.

    When using <tranformer>.transform(), the "always_xy" flag is set TRUE, so make sure
    to always use (lon, lat) rather than (lat, lon) coordinates.
    """
    return (pyproj.Transformer.from_crs(source_crs, dest_crs, always_xy=True), \
            pyproj.Transformer.from_crs(dest_crs, source_crs, always_xy=True))

def create_lat_lon_polygons(polygon, polygon_crs, side_segments=10, type="POLYGONS"):
    """
    If a polygon intersects the 180* line, split it into 2 or more polygons and return them.
    If a polygon overlaps or its boundaries intersects one of the poles, change the polygon to include the pole.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon, or the WKT for a polygon
        A polygon in an arbirtrary crs. It may be a shapely.geometry.Polygon object,
        or the well-known-text for a polygon object, which can be converted into
        a shapely.geometry.Polygon object.

    polygon_crs: pyproj.crs.CRS object, the WKT of a CRS, or an integer EPSG code.
        The geographic coordinate system in which the polygon given exists.
        May be a pyproj.crs.CRS object, the well-known-text of a CRS, or
        an integer ESPG code. The latter two will be converted to a pyproj.crs.CRS
        object for processing.

    side_segments: integer
        The number of segments in which to split each polygon side into before
        reprojections. Since reprojection can cause warping--sometimes significantly
        at projection boundaries--increasing the number of segments on each polygon
        boundary helps better preserve the original shape of the polygon in a new
        projection, at the cost of increased complexity of the polygon. A value of
        1 will not increase complexity at all. Must be an integer >= 1.

    type: str
        The type of geometry to create if splitting up the polygon. Can be
        "POLYGONS", which will return multiple bound polygons, or
        "MULTIPOLYGON", which will return one single multi-polygon.
        Both encompass the exact same area. The Multi-polygon is useful if you're
        creating a Shapefile or other geometry object, but the polygons option is
        useful if you need simple polygons (such as querying NSIDC's web API for data).

    Returns
    -------
    A list of shapely.geometry.Polygons in WGS84 (ESPG:4326) coordinates.
    """
    if type(polygon) == geometry.Polygon:
        input_poly_obj = polygon
    elif type(polygon) in (list,tuple):
        input_poly_obj = geometry.Polygon(polygon)
    elif type(polygon) == str:
        input_poly_obj = wkt.loads(polygon)
    else:
        raise TypeError("Uknown polygon type: {0} for input argument 'polygon'.".format(str(type(polygon))))

    poly_crs_obj = pyproj.CRS.from_user_input(polygon_crs)

    # If the input polygon is already in lat/lon, just return it.
    if poly_crs_obj.to_espg() == 4326:
        return input_poly_obj
