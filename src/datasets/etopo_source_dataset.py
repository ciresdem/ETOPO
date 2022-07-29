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

###############################################################################
# Import the project /src directory into PYTHONPATH, in order to import all the
# other modules appropriately.
import import_parent_dir; import_parent_dir.import_src_dir_via_pythonpath()
###############################################################################

import utils.configfile
import utils.progress_bar as progress_bar
import datasets.dataset_geopackage as dataset_geopackage

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

    def get_geodataframe(self, verbose=True):
        """Retrieve the geodataframe of the tile outlines. The geometries are polygons.
        If the dataframe does not exist where it says, it will be created.
        """
        geopkg = self.get_geopkg_object(verbose=verbose)
        return geopkg.get_gdf(verbose=verbose)

    def get_crs(self, as_epsg=True):
        """Get the CRS or EPSG of the coordinate reference system associated with this dataset."""
        gdf = self.get_geodataframe(verbose=False)
        if as_epsg:
            return gdf.crs.to_epsg()
        else:
            return gdf.crs

    def create_waffles_datalist(self, verbose=True):
        """Create a datalist file for the dataset, useful for cudem "waffles" processing.
        It will use the same name as the geopackage
        object, just substituting '.gpkg' for '.datalist'.
        """
        datalist_fname = self.get_datalist_fname()

        gdf = self.get_geodataframe(verbose=verbose)
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

    def get_datalist_fname(self):
        """Derive the source datalist filename from the geopackage filename.
        Just substitute .gpkg or .datalist
        """
        return os.path.splitext(self.config._abspath(self.config.geopackage_filename))[0] + ".datalist"

    def retrieve_all_datafiles_list(self, verbose=True):
        """Return a list of every one of the DEM tiff data files in this dataset."""
        gdf = self.get_geodataframe()
        return gdf['filename'].tolist()

    def retrieve_list_of_datafiles_within_polygon(self, polygon, polygon_crs, verbose=True):
        """Given a shapely polygon object, return a list of source data files that
        intersect that polygon (even if only partially)."""
        geopkg = self.get_geopkg_object(verbose=verbose)
        subset = geopkg.subset_by_polygon(polygon, polygon_crs)
        return subset["filename"].tolist()

    def vdatum_shift_original_tiles(self, input_tile_fname,
                                          output_tile_fname,
                                          output_vdatum):
        """If a source tile is not in the needed vertical datum, first shift it before
        regridding it."""
        # TODO: Finish this (if needed?)

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


    def generate_tile_datalist_entries(self, polygon, polygon_crs=None, verbose=True, weight=None):
        """Given a polygon (ipmortant, in WGS84/EPSG:4326 coords), return a list
        of all tile entries that would appear in a CUDEM datalist. If no source
        tiles overlap the polygon, return an empty list [].

        Each datalist entry is a 3-value string, as such:
        [filename/path] [format] [weight]
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

        if weight is None:
            weight = self.get_dataset_ranking_score()

        # Do a command-line "waffles -h" call to see datalist options. The datalist
        # type for raster files is "200"
        DTYPE_CODE = 200

        list_of_overlapping_files = self.retrieve_list_of_datafiles_within_polygon(polygon,
                                                                                   polygon_crs,
                                                                                   verbose=verbose)

        return ["{0} {1} {2}".format(fname, DTYPE_CODE, weight) \
                for fname in list_of_overlapping_files]


    def get_dataset_ranking_score(self, fname=None):
        """Given a polygon region, compute the quality (i.e. ranking) score of the dataset in that region.
        If the dataset contains no files in that region, return the 'default_ranking_score' of
        the dataset, provided in the constructor."""

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

if __name__ == "__main__":

    pass
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
#     print(get_source_dataset_object("CopernicusDEM"))
