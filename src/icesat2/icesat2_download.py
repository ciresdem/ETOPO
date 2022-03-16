# -*- coding: utf-8 -*-

"""icesat2_download.py -- Code for downloading ICESat-2 data (both ATL03 & ATL08 granules) for DEM validation."""

def download_icesat2_granules(bounding_area_or_shapefile,
                              date_range,
                              icesat2_directory,
                              icesat2_granules_gpkg=None,
                              datasets=["ATL03","ATL08"],
                              overwrite_existing=False,
                              verbose=True):
    """Given an outline area (bounding-box text, coordinates, or a shapefile. Must be in WGS84 coordinates),
    and a (start,end) range of dates in YYYY-MM-DD format, download ICESat-2 granules.

    This function relies on the ICESat-2 API, documented at these sites:
    - https://nsidc.org/support/how/how-do-i-programmatically-request-data-services
    - https://nsidc.org/support/faq/what-data-subsetting-and-reformatting-services-are-available-icesat-2-data
    - https://nsidc.org/support/tool/table-key-value-pair-kvp-operands-subsetting-reformatting-and-reprojection-services

    Parameters:
        bounding_area_or_shapefile:
            Can be any of the following:
            - The name of a .zip shapefile.
            - A 4-value bounding box (as a list, numpy array, or string)
            - A multi-value polygon (as a list, numpy array, or string)
            The documentation for these formats is here: https://nsidc.org/support/tool/table-key-value-pair-kvp-operands-subsetting-reformatting-and-reprojection-services
        date_range:
            A 2-value list of strings in YYYY-MM-DD format. The latter date must be equal to or later than the former date.
        icesat2_directory:
            An existing directory where ICESat-2 granules will be written.
        datasets:
            A string of the ATL dataset to download, or a list of strings. Default to ["ATL03", "ATL08"]
        overwrite_existing:
            If the granules already exist, download and overwrite them. If False, download only granules
            that do not already exist in the icesat2_directory. Default False.
        verbose:
            If True, put output to the screen. If False, run quietly. Default true.

    Return values:
        A list of ICESat-2 granule filenames that were downloaded.
    """
    # TODO: Finish
    # Step 1: Check inputs for validity.

    # Step 2: Query NSIDC API for lists of the granules within that bounding area.

    # Step 3: (If not overwrite_existing): Filter out granules that we already have.

    # Step 4: Download the new granules we need.

    # Step 5: If icesat2_granules_gpkg is given, add the granules to the geopackage given.

    # Step 6: Return a list of the granules downloaded. NOTE: This will NOT necessarily be all the granules
    #         within the dataset, especially if some already existed on disk.

    return None
