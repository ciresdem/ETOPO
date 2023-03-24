# -*- coding: utf-8 -*-

"""validate_dem_collection.py
Code for validating and summarizing an entire list or directory of DEMs.
"""

import os
import pandas
import numpy
from osgeo import gdal, osr
import argparse
import re

####################################
# Include the base /src/ directory of thie project, to add all the other modules.
import import_parent_dir; import_parent_dir.import_src_dir_via_pythonpath()
####################################
import icesat2.validate_dem as validate_dem
# import icesat2.icepyx_download as icepyx_download
import icesat2.nsidc_download as nsidc_download
import icesat2.icesat2_photon_database as icesat2_photon_database
import icesat2.classify_icesat2_photons as classify_icesat2_photons
import icesat2.granule_shapefile as granule_shapefile
import icesat2.plot_validation_results as plot_validation_results
import etopo.coastline_mask as coastline_mask

def read_or_create_photon_h5(dem_list,
                             photon_h5,
                             output_dir=None,
                             icesat2_dir=None,
                             dates=["2021-01-01","2022-01-01"],
                             create_shapefile=False,
                             shapefile_name=None,
                             skip_icesat2_download=False,
                             overwrite=False,
                             verbose=True):
    """If the photon_h5 file exists, read it. Else, create one from the large bounding box of the DEMs.

    If create_shapefile is set, create a shapefile of the granule paths. (Sometimes useful for debugging or visualization).
    Note: If the dataframe already exists (and is just being read), 'create_shapefile' is ignored.
    """
    # If the photon hdf5 file already exists, read it and return the dataframe.
    if os.path.exists(photon_h5) and not overwrite:
        if verbose:
            print("Reading", photon_h5 + "...", end="")
        photon_df = pandas.read_hdf(photon_h5, mode='r')
        if verbose:
            print("Done.")
        return photon_df

    # Otherwise, generate a large bounding box, and download icesat-2 data into the photon bounding box.
    if icesat2_dir is None:
        icesat2_dir = os.path.dirname(photon_h5)

    # 1. Get the "master" bounding box of all the DEMs in this set.
    # NOTE OF WARNING: This will not work if DEM's straddle the longitudinal dateline (+/- 180* longitude).
    # We are presuming they don't. This will break if that is not true.
    # Start with nonsense min/max values.
    xmin_total = numpy.inf
    xmax_total = -numpy.inf
    ymin_total = numpy.inf
    ymax_total = -numpy.inf

    # Set empty variables for the DEM projection and EPSG values.
    dset_projection = None
    dset_epsg = None
    # Loop through the DEMs to get an emcompassing bounding box.
    for dem_name in dem_list:
        if not os.path.exists(dem_name):
            print("File", dem_name, "does not appear to exist at the location specified. Skipping.")
            continue

        dset = gdal.Open(dem_name, gdal.GA_ReadOnly)
        gtf = dset.GetGeoTransform()
        xsize, ysize = dset.RasterXSize, dset.RasterYSize
        this_dset_projection = dset.GetProjection()
        xleft, xres, xskew, ytop, yskew, yres = gtf
        xright = xleft + (xsize*xres)
        ybottom = ytop + (ysize*yres)
        this_dset_epsg = coastline_mask.get_dataset_epsg(dset)

        # Check to make sure this DEM is in the same projection & EPSG as the others. Quit if not.
        if dset_projection != None and ((this_dset_projection != dset_projection) or (this_dset_epsg != dset_epsg)):
            raise Exception("Not all dems in 'dem_list' are in the same projection. Cannot create a global bounding box.")
        else:
            dset_projection = this_dset_projection
            dset_epsg = this_dset_epsg

        # Set the bounding box to the outer edge of the previous bounding box plus this DEM.
        xmin_total = min(xleft, xright, xmin_total)
        xmax_total = max(xleft, xright, xmax_total)
        ymin_total = min(ytop, ybottom, ymin_total)
        ymax_total = max(ytop, ybottom, ymax_total)


    # If we'd set the boundaries, create the bounding box.
    if (xmin_total != numpy.inf) and (xmax_total != -numpy.inf) and \
       (ymin_total != numpy.inf) and (ymax_total != -numpy.inf):

        bbox = (xmin_total, ymin_total, xmax_total, ymax_total)
    # Otherwise just don't use one.
    else:
        raise Exception("Failed to create bounding box from listed DEMs.")

    # if bbox is None:
    #     # If there's no bounding box, there should be no dataset projection gathered either.
    #     # This is just a logical sanity check to make sure that's true. If we ever
    #     # fail this assertion, come back here and figure out what odd logic is going on.
    #     assert dset_projection == None and dset_epsg == None

    #     if verbose:
    #         print("Warning: No valid DEMs were provided to create a bounding box in {0}.read_or_create_photon_h5(). No ICESat-2 data downloaded.")
    #     # Theoretically, there still could be ICESat-2 data already in the data directory,
    #     # and the function could still succeed even without any newly downloaded data,
    #     # so I'm not raising an exception here. If there is indeed no data in the
    #     # directory, the call to classify_icesat2_photons.save_photon_data_from_directory() below
    #     # should return None, which will be passed back to calling function.

    # else:
    # If the projection is not in WGS84 (EPSG:4326) as icesat is, we need to convert.
    # It can also be in NAD83 (EPSG:4269), which has equivalent lat/lon coordinates.
    if dset_epsg not in (4326, 4269):
        # TODO: Perhaps create a shapefile of the bounding box with its projection?
        # We would need to test whether or not a projected shapefile works in the icepyx API,
        # or if it needs to be in WGS84 (ESPG:4326)
        # FOR NOW, just convert the bbox into ESPG 4326, and send it as polygon
        # points in ESPG 4236 projection.This is fine for now, but THIS WILL BREAK
        # when dealing with polar stereo projections that include the N or S poles,
        # or which overlaps the -180/180 longitude line (which many do).
        # A box that includes one of the poles in polar stereo will not do so in
        # geogrphic coordinates.
        # Update this code later to deal with it more elegantly.
        icesat2_srs = osr.SpatialReference()
        icesat2_srs.SetWellKnownGeogCS("EPSG:4326")
        dem_srs = osr.SpatialReference(wkt=dset_projection)
        # Convert bbox points from DEM projection into
        proj_to_wgs84 = osr.CoordinateTransformation(dem_srs, icesat2_srs)

        # Create a list of bbox points in counter-clockwise order.
        xmin, ymin, xmax, ymax = bbox
        points = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)]
        output_points = proj_to_wgs84.TransformPoints(points)
        bbox_wgs84 = [(p[0], p[1]) for p in output_points]
    else:
        bbox_wgs84 = bbox


    if skip_icesat2_download:
        urls = nsidc_download.download_granules(short_name=["ATL03","ATL08"],
                                                region=bbox_wgs84,
                                                local_dir=icesat2_dir,
                                                dates=dates,
                                                version=validate_dem.etopo_config.nsidc_atl_version,
                                                fname_python_regex=r"\.h5\Z",
                                                force=False,
                                                download_only_matching_granules = True,
                                                query_only=True, # IMPORTANT: we're just querying to see what ATL03 granules are within this bbox, not actually downloading them.
                                                quiet=False)

        atl_fnames = [url.split("/")[-1] for url in urls]
        atl03_granules_list = [os.path.join(icesat2_dir, fname) for fname in atl_fnames if fname.find("ATL03") > -1]
        atl08_granules_list = [os.path.join(icesat2_dir, fname) for fname in atl_fnames if fname.find("ATL08") > -1]

        # Only get names of files that actually exist locally on disk.
        atl03_granules_list = [fn for fn in atl03_granules_list if os.path.exists(fn)]
        atl08_granules_list = [fn for fn in atl08_granules_list if os.path.exists(fn)]

    else:
        # granules_list = nsidc_download.download_granules(short_name="ATL08",
        #                                                  region=bbox_wgs84,
        #                                                  local_dir=icesat2_dir,
        #                                                  version=validate_dem.etopo_config.nsidc_atl_version,
        #                                                  dates = dates,
        #                                                  fname_python_regex="\.h5\Z", # Only download H5 files, not XML files.
        #                                                  force=False,
        #                                                  download_only_matching_granules = True,
        #                                                  query_only=False,
        #                                                  quiet=not verbose)
        # print("Find 'FOOBAR' in validate_dem_collection.py. Remove the grnaules_list download command above it. Then re-run.")
        # FOOBAR

        # If we're not just querying, actually download the granules (query_only=False)
        granules_list = nsidc_download.download_granules(short_name=["ATL03","ATL08"],
                                                         region=bbox_wgs84,
                                                         local_dir=icesat2_dir,
                                                         version=validate_dem.etopo_config.nsidc_atl_version,
                                                         dates = dates,
                                                         fname_python_regex="\.h5\Z", # Only download H5 files, not XML files.
                                                         force=False,
                                                         download_only_matching_granules = True,
                                                         query_only=False,
                                                         quiet=not verbose)

        # print("Granules:", len(granules_list), "\n", granules_list)

        atl03_granules_list = [fn for fn in granules_list if (os.path.split(fn)[1]).find("ATL03") > -1]
        atl08_granules_list = [fn for fn in granules_list if (os.path.split(fn)[1]).find("ATL08") > -1]

    # Get a list of the common granules between the ATL03 and ATL08 datasets (usually the same list, but occasionally
    # a granule in the corner of the search box may be included in one dataset's NSIDC query but omitted from another.)
    common_granule_ids = []
    atl08_fnames = [os.path.split(a8)[1] for a8 in atl08_granules_list]
    atl03_fnames = [os.path.split(a3)[1] for a3 in atl03_granules_list]
    # print("ATL03:", len(atl03_fnames), "\n", atl03_fnames)
    # print("ATL08:", len(atl08_fnames), "\n", atl08_fnames)
    for atl03_path, atl03_gid in zip(atl03_granules_list, atl03_fnames):
        if atl03_gid.replace("ATL03", "ATL08") in atl08_fnames:
            common_granule_ids.append(atl03_path)

    # Make sure we actually have some files to use here.
    if len(common_granule_ids) == 0:
        raise Exception("No common ATL03/08 ICESat-2 granules exist within the bounding box of the DEMs, within the {0} directory.".format(icesat2_dir))

    # If we're using a bounding box and it's not in WGS84 projection, then we must provide a
    # converter for the data to fit within the bounding box.

    if bbox != None and dset_epsg != 4326:
        icesat2_srs = osr.SpatialReference()
        icesat2_srs.SetWellKnownGeogCS("EPSG:4326")
        dem_srs = osr.SpatialReference(wkt=dset_projection)
        bbox_converter = osr.CoordinateTransformation(icesat2_srs, dem_srs)
    else:
        bbox_converter = None

    # Read the point cloud photon data into an HDF5 dataset.
    # NOTE: This doesn't work if we're using the cache directory with *all* the icesat-2 photons.
    # Must use the list of photons provided above.
    photon_df = classify_icesat2_photons.save_photon_data_from_directory_or_list_of_granules(common_granule_ids,
                                                                                             photon_h5 = photon_h5,
                                                                                             bounding_box=bbox,
                                                                                             bbox_converter = bbox_converter,
                                                                                             verbose=verbose)

    if create_shapefile:
        if shapefile_name is None:
            if output_dir is None:
                shapefile_name = os.path.splitext(photon_h5)[0] + ".shp"
                sdir, sname = os.path.split(shapefile_name)
                shapefile_name = os.path.join(sdir, sname.replace("photons", "granules"))
            else:
                sname = os.path.split(os.path.splitext(photon_h5)[0])[1].replace("photons","granules") + ".shp"
                shapefile_name = os.path.join(output_dir, sname)

        if bbox != None and dset_epsg != 4326:
            icesat2_srs = osr.SpatialReference()
            icesat2_srs.SetWellKnownGeogCS("EPSG:4326")
            dem_srs = osr.SpatialReference(wkt=dset_projection)
            bbox_converter = osr.CoordinateTransformation(icesat2_srs, dem_srs)
        else:
            bbox_converter = None

        # Create the shapefile if either it doesn't exist, or we've flagged to overwrite it.
        if overwrite or not os.path.exists(shapefile_name):
            granule_shapefile.output_shapefile_of_granules(shapefile_name,
                                                           icesat2_dir,
                                                           bbox=bbox,
                                                           bbox_converter=bbox_converter,
                                                           verbose=verbose)


    return photon_df

def write_summary_csv_file(total_photon_df: pandas.DataFrame,
                           csv_name: str) -> None:
    """Write a summary csv of all the results in a collection, after they've been run."""
    # TODO: Finish

def validate_list_of_dems(dem_list_or_dir,
                          photon_h5,
                          use_icesat2_photon_database=True,
                          results_h5=None,
                          fname_filter=r"\.tif\Z",
                          fname_omit=None,
                          output_dir=None,
                          icesat2_dir=None,
                          input_vdatum="wgs84",
                          output_vdatum="wgs84",
                          overwrite=False,
                          place_name = None,
                          create_individual_results = False,
                          date_range=["2021-01-01", "2021-12-31"],
                          skip_icesat2_download=False,
                          delete_datafiles=False,
                          include_photon_validation=True,
                          write_result_tifs=False,
                          shapefile_name = None,
                          omit_bad_granules = True,
                          write_summary_csv = True,
                          # omission_bboxes = None,
                          verbose=True):
    """Take a list of DEMs, presumably in a single area, and output validation files for those DEMs.

    DEMs should encompass a contiguous area so as to use the same set of ICESat-2 granules for
    validation."""

    if output_dir is None:
        if photon_h5 is None:
            if os.path.isdir(dem_list_or_dir):
                stats_and_plots_dir = dem_list_or_dir
            elif type(dem_list_or_dir) == str:
                stats_and_plots_dir = os.path.dirname(dem_list_or_dir)
            else:
                dem_list_fitting_filter = [fn for fn in dem_list_or_dir if (((fname_filter is None) or (re.search(fname_filter, os.path.split(fn)[1]) != None)) \
                                                                            and \
                                                                            ((fname_omit is None) or (re.search(fname_omit, os.path.split(fn)[1]) == None)))]
                stats_and_plots_dir = os.path.dirname(dem_list_fitting_filter[0])
        else:
            stats_and_plots_dir = os.path.split(os.path.abspath(photon_h5))[0]
    else:
        if os.path.isdir(output_dir):
            stats_and_plots_dir = output_dir
        else:
            # If the output dir appears to be a relative path, then join it with the input dir.
            if type(dem_list_or_dir) == str:
                stats_and_plots_dir = os.path.join(os.path.dirname(dem_list_or_dir), output_dir)
            else:
                dem_list_fitting_filter = [fn for fn in dem_list_or_dir if (((fname_filter is None) or (re.search(fname_filter, os.path.split(fn)[1]) != None)) \
                                                                            and \
                                                                            ((fname_omit is None) or (re.search(fname_omit, os.path.split(fn)[1]) == None)))]
                stats_and_plots_dir = os.path.join(os.path.dirname(dem_list_fitting_filter[0]), output_dir)


    if place_name is None:
        stats_and_plots_base = "summary_results"
    else:
        stats_and_plots_base = place_name + "_results"

    statsfile_name = os.path.join(stats_and_plots_dir, stats_and_plots_base + ".txt")
    plot_file_name = os.path.join(stats_and_plots_dir, stats_and_plots_base + ".png")

    # If the .h5 results file already exists but not the other files, just
    # create them and exit.
    if (not overwrite) and (results_h5 is not None) and os.path.exists(results_h5):
        results_df = None

        if not os.path.exists(statsfile_name):
            results_df = pandas.read_hdf(results_h5)
            if verbose:
                print(results_df, "read.")
            validate_dem.write_summary_stats_file(results_df,
                                                  statsfile_name,
                                                  verbose=verbose)

        if not os.path.exists(plot_file_name):
            if results_df is None:
                results_df = pandas.read_hdf(results_h5)
                if verbose:
                    print(results_df, "read.")
            plot_validation_results.plot_histogram_and_error_stats_4_panels(results_df,
                                                                            plot_file_name,
                                                                            place_name=place_name,
                                                                            verbose=verbose)
        if (results_df is None) and verbose:
            print("Files '" + results_h5 + "',",
                  "'" + statsfile_name + "', and '",
                  plot_file_name + "' are all already written. There's nothing left to do here.\n",
                  "To recompute them, run with --overwrite enabled, or delete output files as needed and re-run to create them again.\n",
                  "Exiting.")
        return

    path = dem_list_or_dir
    # If we have a one-item list here, get the item in that list.
    if type(path) in (list, tuple) and len(path) == 1:
        path = path[0]


    if (type(path) in (list, tuple)) and (len(path) > 1):
        dem_list = path
    elif os.path.isdir(path):
        dem_list = sorted([os.path.join(path, fname) for fname in os.listdir(path)])
    else:
        assert os.path.exists(path)
        dem_list = [path]

    # Filter for needed strings in filenames, such as "_wgs84.tif"
    if fname_filter != None:
        # Include only filenames that MATCH the match string.
        dem_list = [fn for fn in dem_list if (re.search(fname_filter, fn) != None)]

    # Filter out unwanted filename strings.
    if fname_omit != None:
        # Only include filenames that DO NOT MATCH the omission string.
        dem_list = [fn for fn in dem_list if (re.search(fname_omit, fn) == None)]

    if use_icesat2_photon_database:
        # Generate a single photon database object and pass it repeatedly to all the objects.
        # This saves us a lot of re-reading the geodataframe repeatedly.
        photon_db_obj = icesat2_photon_database.ICESat2_Database()
    else:
        # If a common photon dataframe already exists, open and use it.
        # Otherwise, create it.
        photon_df = read_or_create_photon_h5(dem_list,
                                             photon_h5,
                                             output_dir=output_dir,
                                             icesat2_dir=icesat2_dir,
                                             skip_icesat2_download = skip_icesat2_download,
                                             overwrite=overwrite,
                                             create_shapefile = False if (shapefile_name is None) else True,
                                             shapefile_name = shapefile_name,
                                             verbose=verbose)
        photon_db_obj = None

    # if not os.path.exists(photon_h5):
    #     if verbose:
    #         print("Photon database {0} is not available. Ending process.".format(photon_h5))
    #         return

    list_of_results_dfs = []
    for i, dem_path in enumerate(dem_list):
        if verbose:
            print("\n=======", os.path.split(dem_path)[1], "(" + str(i+1), "of", str(len(dem_list)) + ")", "=======")

        if output_dir is None:
            this_output_dir = os.path.split(dem_path)[0]
        elif os.path.isdir(output_dir):
            this_output_dir = output_dir
        else:
            # If it's a relative dir, append it to where the dems are.
            this_output_dir = os.path.join(os.path.dirname(dem_list[0]), output_dir)
            if not os.path.exists(this_output_dir):
                os.mkdir(this_output_dir)

        results_h5_file = os.path.join(this_output_dir, os.path.splitext(os.path.split(dem_path)[1])[0] + "_results.h5")

        # Do the validation.
        # Note: We automatically skip the icesat-2 download here because we already downloaded it above for the whole directory.
        validate_dem.validate_dem_parallel(dem_path,
                                           photon_dataframe_name = None if use_icesat2_photon_database else photon_df,
                                           use_icesat2_photon_database = use_icesat2_photon_database,
                                           icesat2_photon_database_obj= photon_db_obj,
                                           dem_vertical_datum=input_vdatum,
                                           output_vertical_datum = output_vdatum,
                                           granule_ids=None,
                                           results_dataframe_file = results_h5_file,
                                           icesat2_date_range = date_range,
                                           interim_data_dir = this_output_dir,
                                           overwrite=overwrite,
                                           delete_datafiles = delete_datafiles,
                                           write_result_tifs = write_result_tifs,
                                           write_summary_stats = create_individual_results,
                                           include_photon_level_validation = include_photon_validation,
                                           skip_icesat2_download = True,
                                           plot_results = create_individual_results,
                                           mark_empty_results=True,
                                           omit_bad_granules = omit_bad_granules,
                                           quiet=not verbose)

        if os.path.exists(results_h5_file):
            list_of_results_dfs.append(results_h5_file)

    # An extra newline is appreciated here just for readability's sake.
    print()

    if len(list_of_results_dfs) == 0:
        if verbose:
            print("No results dataframes generated. Aborting.")
        return

    # Generate the overall summary stats file.
    total_results_df = plot_validation_results.get_data_from_h5_or_list(list_of_results_dfs,
                                                                        verbose=verbose)

    # if omission_bboxes != None:
    #     # We have chosen to omit results from one or more bounding boxes. The should be a list of length-4 lists, or just a single length-4 list of [xmin,ymin,xmax,ymax].
    #     # We want to OMIT any pixels that are included in any of those boxes. We can do that here.

    #     # If it's just a 4-tuple of numbers, make it a list of lists to the next loop works.
    #     if len(omission_bboxes) == 4 and type(omission_bboxes[0]) in (int, float):
    #         omission_bboxes = [omission_bboxes]

    #     for bbox in omission_bboxes:
    #         xmin, ymin, xmax, ymax = bbox

    if write_summary_csv:
        summary_csv_name = os.path.join(this_output_dir, stats_and_plots_base+".csv")
        write_summary_csv_file(total_results_df, summary_csv_name)

    # Output the statistics summary file.
    validate_dem.write_summary_stats_file(total_results_df,
                                          statsfile_name,
                                          verbose=verbose)

    # Output the validation results plot.
    plot_validation_results.plot_histogram_and_error_stats_4_panels(total_results_df,
                                                                    plot_file_name,
                                                                    place_name=place_name,
                                                                    verbose=verbose)

    if results_h5 is not None:
        total_results_df.to_hdf(results_h5, "results", complib="zlib", complevel=3)
        if verbose:
            print(results_h5, "written.")

    return

def define_and_parse_args():
    parser = argparse.ArgumentParser(
        description="Tool for validating a list or directory of DEMs against ICESat-2 photon data.")

    parser.add_argument("directory_or_files", type=str, nargs='+',
        help="A directory path, or a list of individual DEM tiles. Defaults to the same as the input directory, or the directory in which the first DEM resides.")

    parser.add_argument("-fname_filter", "-ff", type=str, default=r"\.tif\Z",
        help="A regex string to search for in all DEM file names, to use as a filter. Defaults to r'\.tif\Z', indicating .tif at the end of the file name. Helps elimiate files that shouldn't be considered.")

    parser.add_argument("-fname_omit", "-fo", type=str, default=None,
        help="A regex string to search for and OMIT if it contains a match in the file name. Useful for avoiding derived datasets (such as converted DEMs) in the folder.")

    parser.add_argument("-output_dir", "-od", type=str, default=None,
        help="Directory to output results (and intermittent data, if no data_dir is specified).")

    parser.add_argument("-icesat2_dir", "-is2d", type=str, default=None,
        help="Directory to put ICESat-2 granules. Default: Use the output_dir.")

    parser.add_argument("-photon_h5", "-h5", type=str, default=None,
        help="Name of the ICESat-2 photon dataframe file to create or use for this analysis. Will use existing files unless 'overwrite' is specified.")

    parser.add_argument("-results_h5", type=str, default=None,
        help="Name of an output .h5 file to store the compiled grid-cell-level results for the entire dataset. Default: Just stores the summary without the actual results h5 file.")

    parser.add_argument("-input_vdatum", "-ivd", default="wgs84",
        help="The vertical datum of the input DEMs. [TODO: List possibilities here.] Default: 'wgs84'")

    parser.add_argument("-output_vdatum", "-ovd", default="wgs84",
        help="The vertical datume of the output analysis. Must be a vdatum compatible with Icesat-2 granules. Default: Use the same vdatum as the input files.")

    parser.add_argument("-date_range", nargs=2, default=["2021-01-01", "2022-01-01"],
        help="Input a pair of date ranges to searcvh for ICESat-2 data, in yyyy-mm-dd format. Defaults to calendar year 2021, or '2021-01-01 2021-12-31'.")

    parser.add_argument("-place_name", "-name", type=str, default=None,
        help="Readable name of the location being validated. Will be used in output summary plots and validation report.")

    parser.add_argument("-shapefile", "-shp", type=str, default=None,
        help="Name of a shapefile to save locations of the granule paths. Default: No shapefile created.")

    parser.add_argument('--use_icesat2_photon_database', '-is2db', action='store_true', default=False,
        help="Use the optimized ICESat-2 photon database rather than downloading granules separately. This can save time & memory if the database has already been built on this machine.")

    parser.add_argument("--overwrite", "-o", action="store_true", default=False,
        help="Overwrite all files, including intermittent data files. Default: False (skips re-computing already-computed reseults.")

    parser.add_argument("--skip_icesat2_download", action="store_true", default=False,
        help="Skip dowloads of ICESat-2 data. Just use the files already there.")

    parser.add_argument("--create_folders", action="store_true", default=False,
        help="Create folders specified in -output_dir and -data_dir, as well as the full path to -photon_h5, if they do not already exist. Default: Raise errors if paths don't already exist.")

    parser.add_argument("--individual_results","--ind", action="store_true", default=False,
        help="By default, a summary plot and text file are generated for the dataset. If this is selected, they will be generated for each individual DEM as well. Files will be placed in the -output_dir directory.")

    parser.add_argument("--include_photon_validation","--ph", action="store_true", default=False,
        help="Produce a photon database (stored in '*_photon_level_results.h5') with errors on a photon-level (not cell-level) scale. Useful for identifying bad ICESat-2 granules.")

    parser.add_argument("--delete_datafiles", "--del", action="store_true", default=False,
        help="By default, all data files generted in this process are kept. If this option is chosen, delete them.")

    parser.add_argument("--write_result_tifs", action='store_true', default=False,
        help="""Write output geotiff with the errors in cells that have ICESat-2 photons, NDVs elsewhere.""")

    parser.add_argument("--write_summary_csv", action='store_true', default=False,
        help="Write a CSV with summary results of each individual DEM.")

    parser.add_argument("--quiet", "-q", action="store_true", default=False,
        help="Suppress output.")

    return parser.parse_args()

def main():
    args = define_and_parse_args()

    if args.output_dir is None:
        # Default to data dir or directory of first data file.
        path = args.directory_or_files[0]
        if type(path) is str and os.path.isdir(path):
            args.output_dir = path
        else:
            args.output_dir = os.path.split(path)[0]
            assert type(args.output_dir) == str

    if args.icesat2_dir is None:
        # Default: set the data directory to the same as the output directory
        args.icesat2_dir = args.output_dir

    if args.photon_h5 is None and not args.use_icesat2_photon_database:
        # Default: sat the photon H5 file to a file in the data directory with th same name as the data directory.
        dirname = os.path.split(os.path.abspath(args.icesat2_dir))[1]
        args.photon_h5 = os.path.join(args.icesat2_dir, dirname + ".h5")
        if not args.quiet:
            print ("ICESat-2 photon data will be stored in", args.photon_h5 + ".")

    if (type(args.fname_filter) in (list, tuple)) and (len(args.fname_filter) == 1):
        args.fname_filter= args.fname_filter[0]

    # Check for the existence of appropriate output & data directories.

    output_dir = os.path.abspath(args.output_dir)
    if not os.path.exists(output_dir):
        if args.create_folders:
            os.makedirs(output_dir)
        else:
            raise FileNotFoundError("Output directory '{0}' does not exist. Create directory or use the --create_folders flag upon execution.".format(args.output_dir))

    if not os.path.exists(args.icesat2_dir):
        if args.create_folders:
            os.makedirs(args.icesat2_dir)
        else:
            raise FileNotFoundError("ICESat-2 data directory '{0}' does not exist. Create directory or use the --create_folders flag upon execution.".format(args.icesat2_dir))

    if args.photon_h5 is not None:
        h5_dir, h5_file = os.path.split(args.photon_h5)
        if h5_dir != "" and not os.path.exists(h5_dir):
            if args.create_folders:
                os.makedirs(h5_dir)
            else:
                raise FileNotFoundError("Directory '{0}' does not exist to write photon point-cloud file '{1}'. Create directory or use the --create_folders flag upon execution.".format(h5_dir, h5_file))

    # NOTE: This code assumes that if we create the directory here, it will
    # not be erased befor the code gets to putting files there later. Seems
    # like a generally safe assumption, and behavior is okay if another process
    # or the user manually deletes directories during execusion, it will cause
    # the program to crash when it tries to write files there. That's a user error.

    validate_list_of_dems(args.directory_or_files, args.photon_h5,
                          use_icesat2_photon_database=args.use_icesat2_photon_database,
                          results_h5=args.results_h5,
                          fname_filter=args.fname_filter,
                          fname_omit=args.fname_omit,
                          output_dir=args.output_dir,
                          icesat2_dir=args.output_dir,
                          input_vdatum=args.input_vdatum,
                          output_vdatum=args.output_vdatum,
                          overwrite=args.overwrite,
                          place_name=args.place_name,
                          create_individual_results=args.individual_results,
                          date_range=args.date_range,
                          skip_icesat2_download=args.skip_icesat2_download,
                          delete_datafiles=args.delete_datafiles,
                          include_photon_validation=args.include_photon_validation,
                          write_result_tifs=args.write_result_tifs,
                          shapefile_name=args.shapefile,
                          write_summary_csv=args.write_summary_csv,
                          omit_bad_granules=True,
                          verbose=not args.quiet)

if __name__ == "__main__":
    main()
