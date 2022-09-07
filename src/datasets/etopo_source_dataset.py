# -*- coding: utf-8 -*-

"""source_dataset.py -- Defines the definition of the parent class for each subset
dataset.

Each individual dataset source code is inside a folder in /src/datasets with the name of the
dataset, and the code for that package is defined in /src/datasets/DatasetName/source_dataset_DatasetName.py.

The parent class defined here is SourceDataset. The sub-classes defined in each sub-module
should be named SourceDataset_DatasetName and directly inherit SourceDataset.

This allows the code to dynamically ingest and inherit new datasets as they come online rather
than having them all hard-coded."""

# SourceDataset is the base class for each of the source dataset classes that inherit from this.
# This defines the methods that all the child sub-classes should use.

import os
# import geopandas
import importlib
from osgeo import gdal
import multiprocessing
import subprocess
import time
import argparse
import shutil

###############################################################################
# Import the project /src directory into PYTHONPATH, in order to import all the
# other modules appropriately.
import import_parent_dir; import_parent_dir.import_src_dir_via_pythonpath()
###############################################################################

import utils.configfile
import utils.progress_bar as progress_bar
import utils.parallel_funcs
import datasets.dataset_geopackage as dataset_geopackage
import etopo.convert_vdatum

def get_source_dataset_object(dataset_name, verbose=True):
    """Given the name of a dataset, import the dataset object from the subdirectory that contains that derived object.
    Using ETOPO source naming convention, the source code will reside in
    datasets.[name].source_dataset_[name].py:source_dataset_[name]"""
    dname = dataset_name.strip()
    module_name = "datasets.{0}.source_dataset_{0}".format(dname)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        if verbose:
            print("Module {0} not found.".format(module_name))
        return None

    try:
        class_name = "source_dataset_{0}".format(dname)
        class_obj = getattr(module, class_name)()
    except AttributeError:
        if verbose:
            print("No class definition for '{0}' found in {1}.".format(class_name, module_name))
        return None

    return class_obj

class ETOPO_source_dataset:
    # The base directory of the project is two levels up. Retrieve the absolute path to it on this machine.
    # This file resides in [project_basedir]/src/datasets
    project_basedir = os.path.abspath(os.path.join(os.path.split(__file__)[0], "..", ".."))

    def __init__(self, dataset_name,
                       configfile):
        # Get information from the configuration file. See fields in the [dataset]_config.ini
        self.configfilename = configfile
        self.config = utils.configfile.config(configfile=configfile)

        # Local variables.
        self.dataset_name = dataset_name
        self.geopackage_filename = self.config._abspath(self.config.geopackage_filename)
        # print(self.geopackage_filename)
        # foobar
        self.default_ranking_score = self.config.default_ranking_score

        # The geodataframe of all the tile outlines in the dataset.
        self.geopkg = None

        # The Coordinte Reference System of this dataset. NOTE: All files within the dataset
        # should have the same coordainte reference system.

    def is_active(self):
        """A switch to see if thais dataset is yet being used."""
        # Switch to 'True' when the .ini is filled out and this is ready to go.
        return self.config.is_active

    def get_geopkg_object(self, verbose=True):
        """Get the dataset_geopackage.DatasetGeopackage object associated with this dataset.
        If it doesn't exist, create it.
        """
        if self.geopkg is None:
            self.geopkg = dataset_geopackage.DatasetGeopackage(self.config)
        return self.geopkg

    def get_geodataframe(self, resolution_s = None, verbose=True):
        """Retrieve the geodataframe of the tile outlines. The geometries are polygons.
        If the dataframe does not exist where it says, it will be created.
        """
        geopkg = self.get_geopkg_object(verbose=verbose)
        return geopkg.get_gdf(resolution_s = resolution_s, verbose=verbose)

    def get_crs(self, as_epsg=True):
        """Get the CRS or EPSG of the coordinate reference system associated with this dataset."""
        gdf = self.get_geodataframe(verbose=False)
        if as_epsg:
            return gdf.crs.to_epsg()
        else:
            return gdf.crs

    def create_waffles_datalist(self, resolution_s = None, verbose=True):
        """Create a datalist file for the dataset, useful for cudem "waffles" processing.
        It will use the same name as the geopackage
        object, just substituting '.gpkg' for '.datalist'.

        NOTE: This function is still usable, but is deprecated. We no longer create a full dataset datalist, but
        rather datalists for each tile to increase processing speeds.
        """
        datalist_fname = self.get_datalist_fname(resolution_s = resolution_s)

        gdf = self.get_geodataframe(resolution_s = resolution_s, verbose=verbose)
        filenames = gdf['filename'].tolist()
        DLIST_DTYPE = 200 # Datalist rasters are data-type #200
        ranking_score = self.get_dataset_ranking_score()

        dlist_lines = ["{0} {1} {2}".format(fname, DLIST_DTYPE, ranking_score) for fname in filenames]
        dlist_text = "\n".join(dlist_lines)

        with open(datalist_fname, 'w') as f:
            f.write(dlist_text)
            f.close()
            if verbose:
                print(datalist_fname, "written.")

    def get_datalist_fname(self, resolution_s=None):
        """Derive the source datalist filename from the geopackage filename.
        Just substitute .gpkg or .datalist
        """
        # If the geopackage filename contains a {0} to insert a resolution (1 or 15), use it.
        if (resolution_s != None) and (self.config.geopackage_filename.find("{0}") >= 0):
            gpkg_fname = self.config.geopackage_filename.format(resolution_s)
        else:
            gpkg_fname = self.config.geopackage_filename

        return os.path.splitext(self.config._abspath(gpkg_fname))[0] + ".datalist"

    def retrieve_all_datafiles_list(self, resolution_s= None, verbose=True):
        """Return a list of every one of the DEM tiff data files in this dataset."""
        gdf = self.get_geodataframe(resolution_s = resolution_s, verbose = verbose)
        return gdf['filename'].tolist()

    def retrieve_list_of_datafiles_within_polygon(self, polygon,
                                                        polygon_crs,
                                                        resolution_s = None,
                                                        return_fnames_only=True,
                                                        verbose=True):
        """Given a shapely polygon object, return a list of source data files that
        intersect that polygon (even if only partially).

        If return_fnames_only is True, just return a list of the filenames.
        Else, return the subset of the dataframe table."""
        geopkg = self.get_geopkg_object(verbose=verbose)
        subset = geopkg.subset_by_polygon(polygon, polygon_crs, resolution_s = resolution_s)
        if return_fnames_only:
            return subset["filename"].tolist()
        else:
            return subset

    def set_ndv(self, verbose=True, fail_if_different=True):
        """Some datasets have a nodata value but it isn't listed in the GeoTIFF.
        If [DATASET]_config.ini file has a .dem_ndv attribute in it, go ahead and open all the source
        datasets and forcefully insert the nodata-value so that it will behave properly in waffles.
        """
        if not hasattr(self.config, "dem_ndv"):
            if verbose:
                print("No manual NDV is set in", self.config._configfile + ".", "Exiting.")
            return
        filenames = self.get_geodataframe()['filename'].tolist()
        NDV = self.config.dem_ndv

        if verbose:
            print("Setting NDVs for {0:,} {1} DEM tiles to {2}.".format(len(filenames), self.dataset_name, NDV))
        for i, fname in enumerate(filenames):
            self.set_ndv_individual_tile(fname, NDV, fail_if_different=fail_if_different)
            if verbose:
                progress_bar.ProgressBar(i+1, len(filenames), suffix = "{0:,}/{1:,}".format(i+1, len(filenames)))

    def set_ndv_individual_tile(self, fname, ndv_value, fail_if_different=True):
        """For an individual source tile, set the ndv if it doesn't have one.

        If it already has one, ignore it and just close the file. If that previous value is different than
        the ndv_value and warn_if_different is set, then print a warning when doing it.
        """
        dset = gdal.Open(fname, gdal.GA_Update)
        band = dset.GetRasterBand(1)
        existing_ndv = band.GetNoDataValue()
        # If they are already the same, just move along.
        if existing_ndv == ndv_value:
            return

        if fail_if_different and (existing_ndv != None) and (existing_ndv != ndv_value):
            print(f"Warning in {fname}: existing NDV ({existing_ndv}) != new NDV ({ndv_value}).\n" + \
                  "New data will NOT be written.")
            band = None
            dset = None
            return

        band.SetNoDataValue(ndv_value)
        # Write out the dataset to this NDV sticks before we re-compute stats.
        dset.FlushCache()
        # Re-generate the statistics on the file.
        # "GetStatistics()" only overwrites the stats if they don't already exist.
        # If they do already exist, we need to compute them again (force it) and
        # write them in there using "SetStatistics()"
        band.SetStatistics(*band.ComputeStatistics(0))

        dset.FlushCache()
        band = None
        dset = None
        return

    def generate_tile_datalist_entries(self, polygon,
                                             polygon_crs=None,
                                             resolution_s = None,
                                             verbose=True):
        """Given a polygon (ipmortant, in WGS84/EPSG:4326 coords), return a list
        of all tile entries that would appear in a CUDEM datalist. If no source
        tiles overlap the polygon, return an empty list [].

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

        weight = self.get_dataset_ranking_score()

        # Do a command-line "waffles -h" call to see datalist options. The datalist
        # type for raster files is "200"
        DTYPE_CODE = 200

        list_of_overlapping_files = self.retrieve_list_of_datafiles_within_polygon(polygon,
                                                                                   polygon_crs,
                                                                                   resolution_s = resolution_s,
                                                                                   verbose=verbose)

        return ["{0} {1} {2}".format(fname, DTYPE_CODE, weight) for fname in list_of_overlapping_files]

    def convert_vdatum(self, output_folder = None,
                             output_vdatum="egm2008",
                             update_gpkg = True,
                             resolution_s = None,
                             overwrite = False,
                             numprocs = utils.parallel_funcs.physical_cpu_count(),
                             verbose = True):
        """Convert all the tiles in the dataset to a consistent vertical datum.

        If they are already set to the listed vdatum (or equivalent), leave them.
        If they are converted, write all files to the "output_folder" directory.
            If None, look for a "datafiles_directory_converted" folder.
            If that is not available, create a "converted" subdir in the directory of each source file, and put it there.

        If update_gpkg, update the filenames of each file in the geopackage. Save that back to disk.
        """
        dset_vdatum_epsg = self.get_dataset_vdatum(name=False)

        # We can input the name, but use the number.
        # This is only set up for egm2008 right now, would probably break on other
        # output vdatums. I can generalize the code if we need to spit it out in something else.
        if type(output_vdatum) == int:
            output_vdatum_epsg = output_vdatum
            if output_vdatum == 3855:
                output_vdatum_name = "egm2008"
            else:
                raise ValueError("Unhandled output_vdatum: " + str(output_vdatum))
        elif output_vdatum.strip().lower() == "egm2008":
            output_vdatum_epsg = 3855
            output_vdatum_name = output_vdatum.strip().lower()
        else:
            raise ValueError("Unhandled output_vdatum: " + str(output_vdatum))

        # If the vdatums match, or the the output vdatum is in MSL (5714), leave it.
        if (dset_vdatum_epsg == output_vdatum_epsg) or ((dset_vdatum_epsg == 5714) and (output_vdatum_epsg == 3855)):
            if verbose:
                print("Dataset {0} already in vdatum {1}.".format(self.dataset_name, output_vdatum))
            return

        gdf = self.get_geodataframe(resolution_s = resolution_s, verbose=verbose)

        gdf_changed = False

        running_procs = []
        running_fnames = []
        running_tempdirs = []
        tempdirs_all = []
        scratchdir = utils.configfile.config().etopo_cudem_cache_directory

        # Get all the files from the geo-dataframe
        N = len(gdf.index)
        n_finished = 0
        last_progress_str = ""

        try:

            for i,row in gdf.iterrows():
                
            # Make sure it isn't already converted. (If so, it'd be in the output directory already and end with '_egm2008'.)
            # If it is already converted, skip it.
                fname = row['filename']

                if output_folder is None:
                    # If we say in the config file where to put it, put it there.
                    if hasattr(self.config, "datafiles_directory_converted"):
                        output_folder_thisfile = self.config.datafiles_directory_converted
                    else:
                        # If it's not already in a directory called "converted", put it in a sub-directory called "converted".
                        if os.path.split(os.path.dirname(fname))[1] == "converted":
                            output_folder_thisfile = os.path.dirname(fname)
                        else:
                            output_folder_thisfile = os.path.join(os.path.dirname(fname), "converted")
                else:
                    # If we give the output directory as a parameter, use that.
                    output_folder_thisfile = output_folder

                if not os.path.exists(output_folder_thisfile):
                    os.mkdir(output_folder_thisfile)

                base, ext = os.path.splitext(os.path.split(fname)[1])
                # Put it in the output folder, with an "_egm2008" tag on the filename.
                # If it already has an "_egm2008" tag at the end of the filename, leave it unchanged.
                output_fname = os.path.join(output_folder_thisfile,
                                            base + ("" \
                                                    if base[-(len(output_vdatum_name)+1):] == ("_" + output_vdatum_name) \
                                                    else ("_" + output_vdatum_name)) \
                                                 + ext)

                # If this GDF already reflects the output file, just continue.
                if os.path.abspath(fname) == os.path.abspath(output_fname):
                    n_finished += 1
                    if verbose:
                        progress_bar.ProgressBar(n_finished, N, suffix = "{0}/{1}".format(n_finished, N))
                    continue

                if update_gpkg:
                    gdf.at[i,'filename'] = output_fname
                    gdf_changed = True

                # If the file already exits and we haven't specified to overwrite, leave it there.
                if (not overwrite) and os.path.exists(output_fname):
                    n_finished += 1
                    if verbose:
                        progress_bar.ProgressBar(n_finished, N, suffix = "{0}/{1}".format(n_finished, N))
                    continue

                # print(fname)
                # print(output_fname)

                # Loop through checking sub-processes until one opens up.
                assert len(running_procs) == len(running_tempdirs)
                while len(running_procs) >= numprocs:
                    procs_to_remove = []
                    fnames_to_remove = []
                    tdirs_to_remove = []
                    # First, loop through to see if any existing processes are finished.
                    for proc, fname_out, tempdir in zip(running_procs, running_fnames, running_tempdirs):
                        if not proc.is_alive():
                            proc.join()
                            proc.close()
                            # Remove the temp directory (should be empty now)
                            for fname in [os.path.join(tempdir, fn) for fn in os.listdir(tempdir)]:
                                try:
                                    os.remove(fname)
                                except:
                                    pass
                            os.rmdir(tempdir)

                            # Add this process & tempdir to the list to remove.
                            procs_to_remove.append(proc)
                            fnames_to_remove.append(fname_out)
                            tdirs_to_remove.append(tempdir)
                            n_finished += 1

                            # Update the progress bar
                            if verbose:
                                if len(last_progress_str) > 0:
                                    print(" " * len(last_progress_str), end="\r")
                                print(fname_out)
                                last_progress_str = progress_bar.ProgressBar(n_finished, N, suffix = "{0}/{1}".format(n_finished, N))

                    # Remove any finished procs, tempdirs from the queues.
                    for pr, fn, td in zip(procs_to_remove, fnames_to_remove, tdirs_to_remove):
                        running_procs.remove(pr)
                        running_fnames.remove(fn)
                        running_tempdirs.remove(td)

                    # Sleep for just a moment to keep this process from eating too much CPU.
                    time.sleep(0.01)

                assert len(running_procs) < numprocs

                # Create the tempdir.
                tempdir = os.path.join(scratchdir, "temp" + str(i))
                tempdirs_all.append(tempdir)
                if not os.path.exists(tempdir):
                    os.mkdir(tempdir)

                # Create a new process to convert the next DEM.
                proc = multiprocessing.Process(target = etopo.convert_vdatum.convert_vdatum,
                                               args = (fname, output_fname),
                                               kwargs = {"input_vertical_datum": self.get_dataset_vdatum(name=True),
                                                         "output_vertical_datum": output_vdatum_name,
                                                         "cwd": tempdir,
                                                         "verbose": False},
                                               )
                # Add the process and the tempdirs to the queues.
                running_procs.append(proc)
                running_fnames.append(output_fname)
                running_tempdirs.append(tempdir)
                # Start the process.
                proc.start()

                # # Write the file.
                # etopo.convert_vdatum.convert_vdatum(fname,
                #                                     output_fname,
                #                                     input_vertical_datum = self.get_dataset_vdatum(name=True),
                #                                     output_vertical_datum = output_vdatum_name,
                #                                     cwd = tempdir,
                #                                     verbose=False)

            # Clean up the remaining running processes.
            while len(running_procs) > 0:
                procs_to_remove = []
                fnames_to_remove = []
                tdirs_to_remove = []
                # First, loop through to see if any existing processes are finished.
                for proc, fn_out, tempdir in zip(running_procs, running_fnames, running_tempdirs):
                    if not proc.is_alive():
                        proc.join()
                        proc.close()
                        # Remove the temp directory (should be empty now)
                        try:
                            os.rmdir(tempdir)
                        except OSError:
                            # If the directory is not empty, this fails. Wait for a second a try again.
                            # Do this 3 times. If it's *still* not empty at that point, then manually remove all the files.
                            tries = 0
                            while tries < 3:
                                if len(os.listdir(tempdir)) == 0:
                                    os.rmdir(tempdir)
                                    break
                                tries += 1
                                time.sleep(0.5)
                            for fname in [os.path.join(tempdir, fn) for fn in os.listdir(tempdir)]:
                                try:
                                    os.remove(fname)
                                except FileNotFoundError:
                                    pass
                            os.rmdir(tempdir)

                        # Add this process & tempdir to the list to remove.
                        procs_to_remove.append(proc)
                        fnames_to_remove.append(fn_out)
                        tdirs_to_remove.append(tempdir)
                        n_finished += 1

                        # Update the progress bar
                        if verbose:
                            if len(last_progress_str) > 0:
                                print((" " * len(last_progress_str)) + "\r")
                            print(fn_out)

                            last_progress_str = progress_bar.ProgressBar(n_finished, N, suffix = "{0}/{1}".format(n_finished, N))

                # Remove any finished procs, tempdirs from the queues.
                for pr, fn, td in zip(procs_to_remove, fnames_to_remove, tdirs_to_remove):
                    running_procs.remove(pr)
                    running_fnames.remove(fn)
                    running_tempdirs.remove(td)

                # Sleep for just a moment to keep this process from eating too much CPU.
                time.sleep(0.01)

        except (KeyboardInterrupt, Exception) as e:
            # If this crashes, kill all the procs and remove all the tempdirs.
            for proc in running_procs:
                proc.terminate()
                tries = 0
                while tries < 3:
                    try:
                        if proc.is_alive():
                            proc.terminate()
                        else:
                            tries = 3
                            break
                    except ValueError:
                        pass
                    time.sleep(0.1)
                    tries += 1
                # If terminate hasn't worked, then send the SIGKILL interrupt. Shut it down, then give 0.1s.
                if tries >= 3 and proc.is_alive():
                    proc.kill()
                    time.sleep(0.1)

                assert not proc.is_alive()
                proc.close()
                
            for tdir in running_tempdirs:
                # First delete any contents of the directory, then the directory.
                for fn in [os.path.join(tdir, fn) for fn in os.listdir(tdir)]:
                    os.remove(fn)
                os.rmdir(tdir)
            # After closing other processes, cleaning up files and removing tempdirs, go ahead and error-out gracefully.
            raise e

        if update_gpkg and gdf_changed:
            gpkg_object = self.get_geopkg_object()
            gpkg_fname = gpkg_object.get_gdf_filename(resolution_s = resolution_s)
            print(gpkg_fname)
            assert os.path.splitext(gpkg_fname)[1].lower() == ".gpkg"
            # Set the geodataframe object to our updated version.
            gpkg_object.gdf = gdf
            if verbose:
                print("Writing", gpkg_fname + "...", end="")
            gdf.to_file(gpkg_fname, layer=gpkg_object.default_layer_name, driver="GPKG")
            if verbose:
                print("Done.")

        # These *should* all be gone, but in case, go and clean up the temp dirs.
        for tdir in tempdirs_all:
            if not os.path.exists(tdir):
                continue
            existing_files = [os.path.join(tdir, fn) for fn in os.listdir(tdir)]
            for fn in existing_files:
                os.remove(fn)
            os.rmdir(tdir)

        return

    def get_dataset_id_number(self):
        """Return the (presumably unique) ID number of the dataset."""
        return self.dataset_id_number

    def get_dataset_ranking_score(self):
        """Return the ranking score of the dataset."""
        return self.default_ranking_score

    def get_dataset_vdatum(self, name=True):
        """Return the vertical datum EPSG code or name for the dataset native vertical datum.
        NOTE: This is not necessarily the vertical datum of the file that has been
        validated or used in ETOPO, which may have been converted to EGM2008.
        """
        if name == True:
            return self.config.dataset_vdatum_name
        else:
            return self.config.dataset_vdatum_epsg

    def get_dataset_validation_results(self):
        """After a validation has been run on this dataset, via the validate_dem_collection.py
        or validate_etopo_dataset.py scripts, collect a dataframe of all the validation results.

        This is probably in a summary results.h5 file. If we're dealing with CUDEM_CONUS,
        it would be the composite of all the summary results.h5 files for each region."""
        # TODO: Finish

    def reproject_tiles_from_nad83(self, suffix="_epsg4326", overwrite = False, range_start=None, range_stop=None, verbose=True):
        """Project all the tiles into WGS84/latlon coordinates.

        The fucked-up NAD83 / UTM zone XXN is fucking with waffles. Converting them is the easiest answer now.
        After we do this, we can change the config.ini to look for the new "_epsg4326" tiles. Also, we can disable and
        delete the 5 BlueTopo_14N to _19N datasets, because they'll all be projected into the same coordinate system,
        which will be a lot easier.
        """
        tilenames = self.retrieve_all_datafiles_list(verbose=verbose)
        for i,fname in enumerate(tilenames):
            if ((range_start is not None) and (i < range_start)) or ((range_stop is not None) and (i >= range_stop)):
                continue
            dest_fname = os.path.splitext(fname)[0] + suffix + ".tif"
            if os.path.exists(dest_fname):
                if overwrite or gdal.Open(dest_fname, gdal.GA_ReadOnly) is None:
                    os.remove(dest_fname)
                else:
                    print("{0}/{1} {2} already written.".format(i+1, len(tilenames), os.path.split(dest_fname)[1]))
                    continue

            # This ONLY FUCKING WORKS if it's another lat/lon dataset, such as NAD83 or similar.
            # If it's a projected coordinate system the output will be garbage
            _, xstep, _, _, _, ystep = gdal.Open(fname, gdal.GA_ReadOnly).GetGeoTransform()
            assert (xstep > 0) and (ystep < 0)
            gdal_cmd = ["gdalwarp",
                        "-t_srs", "EPSG:4326",
                        "-dstnodata", "0.0",
                        "-tr", str(xstep), str(abs(ystep)),
                        "-r", "bilinear",
                        "-of", "GTiff",
                        "-co", "COMPRESS=DEFLATE",
                        "-co", "PREDICTOR=2",
                        "-co", "ZLEVEL=5",
                        fname, dest_fname]
            process = subprocess.run(gdal_cmd, capture_output = True, text=True)
            if verbose:
                print("{0}/{1} {2} ".format(i+1, len(tilenames), os.path.split(dest_fname)[1]), end="")
            if process.returncode == 0:
                if verbose:
                    print("written.")
            elif verbose:
                    print("FAILED")
                    print(" ".join(gdal_cmd), "\n")
                    print(process.stdout)

        return


def TEMP_move_CUDEM_egm2008_tiles(dset_names = ("CUDEM_AmericanSamoa",
                      "CUDEM_CONUS", "CUDEM_Guam", "CUDEM_Northern_Mariana",
                      "CUDEM_Puerto_Rico", "CUDEM_USVI"),
                                  end = "_egm2008.tif",
                                  output_subdir = "converted",
                                  use_symlinks = True,
                                  verbose = True):
    """We're spending a lot of time re-creating EGM2008 versions of the CUDEM tiles. We've already done that, they're just in the "icesat2_results"
    directories rather than the "converted" directories. Find them all and fix that.

    If use_symlinks, don't actually move the files, just create a symlink to the original.

    This is just a one-off function to help avoid some redundant processing, not really a generalized function for development.

    After doing this, run the dset.convert_vdatum() code to find all these files and update the geopackage file.
    (No need to re-create that logic here.)"""
    for dset_name in dset_names:
        if verbose:
            print("===========", dset_name, "===========")
        dset_obj = get_source_dataset_object(dset_name, verbose = verbose)

        gdf = dset_obj.get_geodataframe(verbose=verbose)

        N = len(gdf)
        last_outstr = ""
        for n,(i,row) in enumerate(gdf.iterrows()):
            fname = row.filename
            # If the filename already listed is in a "converted" folder and ends in "_egm2008.tif", just skip it.
            if (os.path.split(os.path.dirname(fname))[1] == output_subdir) and (fname[-len(end):] == end):
                continue

            dirname, fname = os.path.split(fname)
            base, ext = os.path.splitext(fname)
            output_path = os.path.join(dirname, output_subdir, base + os.path.splitext(end)[0] + ext)
            if os.path.exists(output_path):
                continue

            if not os.path.exists(os.path.dirname(output_path)):
                os.mkdir(os.path.dirname(output_path))
            if use_symlinks:
                os.symlink(row.filename, output_path)
            else:
                shutil.copy(row.filename, output_path)

            if verbose:
                if len(last_outstr) > 0:
                    print(" " * len(last_outstr), end="\r")
                print(output_path)
                last_outstr = progress_bar.ProgressBar(n+1, N, suffix="{0}/{1}".format(n+1,N))




def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Utility base class for a source dataset object. Only used as a standalone script for specific tasks. Look in code at 'if __name__ == \"main\": block to see what's being executed.")
    parser.add_argument("-numprocs", "-np", type=int, default=utils.parallel_funcs.physical_cpu_count(),  help="Number of processors to use.")
    return parser.parse_args()

if __name__ == "__main__":
    
    # Rebuild the CUDEM datapackages for each one after new files have been downloaded.
    # for groupname in ("CONUS", "CONUS_Sandy", "Hawaii", "Guam", "AmericanSamoa", "Puerto_Rico", "USVI", "Northern_Mariana"):
    for groupname in (["BlueTopo_{0:02d}N".format(z) for z in (14,15,16,18,19)]):
        dset = get_source_dataset_object(groupname)
        gpkg = dset.get_geopkg_object()
        gpkg.create_dataset_geopackage()
        dset.convert_vdatum()

    # for dset_name in ["BlueTopo_{0}N".format(n) for n in (14,15,16,18,19)]:
    #     dset = get_source_dataset_object(dset_name)
    #     dset.override_gdf_projection(verbose=True)

    # TEMP_move_CUDEM_egm2008_tiles()
    # args = define_and_parse_args()
    # for dset_name in ("TanDEMX", "BlueTopo", "CUDEM_AmericanSamoa",
    #                   "CUDEM_CONUS", "CUDEM_Guam", "CUDEM_Northern_Mariana",
    #                   "CUDEM_Puerto_Rico", "CUDEM_USVI"):
    #     dset = get_source_dataset_object(dset_name)
    #     print("==============", dset_name, "==============")
    #     dset.convert_vdatum(numprocs = args.numprocs)
    # TanDEMX.set_ndv()

    # GEBCO = get_source_dataset_object("GEBCO")
    # GEBCO.create_waffles_datalist()

    # FAB = get_source_dataset_object("FABDEM")

    # FAB.set_ndv(verbose=True, fail_if_different=False)


    # COP = get_source_dataset_object("CopernicusDEM")

    # COP.set_ndv(verbose=True)

    # Test out the NDV writing on one tile to begin.
    # print(COP.config.dem_ndv)
    # COP.set_ndv_individual_tile("/home/mmacferrin/Research/DATA/DEMs/CopernicusDEM/data/30m/COP30_hh/Copernicus_DSM_COG_10_N00_00_E006_00_DEM.tif",
    #                             COP.config.dem_ndv)

    # GB = get_source_dataset_object("global_lakes_globathy")
    # GB.create_waffles_datalist(resolution_s = 1)
