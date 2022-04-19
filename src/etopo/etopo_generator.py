# -*- coding: utf-8 -*-

"""etopo_dataset.py -- parent code for managing, building and assessing/validating
   the ETOPO global topo-bathy DEM."""

import os
import imp
import shutil

#####################################################
# Code snippet to import the base directory into
# PYTHONPATH to aid in importing from all the other
# modules in other subdirs.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
#####################################################

import generate_empty_grids
import utils.configfile
import datasets.dataset_geopackage as dataset_geopackage

class ETOPO_Generator:
    # Tile naming conventions. ETOPO_2022_v1_N15E175.tif, e.g.
    fname_template_tif = r"ETOPO_2022_v1_{0:d}s_{1:s}{2:02d}{3:s}{4:03d}.tif"
    fname_template_netcdf = r"ETOPO_2022_v1_{0:d}s_{1:s}{2:02d}{3:s}{4:03d}.nc"

    # The base directory of the project is two levels up. Retrieve the absolute path to it on this machine.
    # This file resides in [project_basedir]/src/etopo
    project_basedir = os.path.abspath(os.path.join(os.path.split(__file__)[0], "..", ".."))

    etopo_config = utils.configfile.config(os.path.join(project_basedir, "etopo_config.ini"), )

    # Note, still need to add resolution "1s" or "15s" directory.
    empty_tiles_directory = etopo_config.etopo_empty_tiles_directory
    # intermediate_tiles_directory = etopo_config.etopo_intermediate_tiles_directory # Same as above
    finished_tiles_directory = etopo_config.etopo_finished_tiles_directory # Same as above
    etopo_gpkg_1s  = etopo_config.etopo_tile_geopackage_1s
    etopo_gpkg_15s = etopo_config.etopo_tile_geopackage_15s

    def __init__(self):
        """Read the configuration file, get the info about our grid locations."""
        self.datasets_dict = None

    def create_empty_grids(self, resolution_s=(1,15), verbose=True):
        """Create empty grid-cells for the ETOPO data product.
        resolution_s can be 1, 15, or a list/tuple with both values (1,15).
        """
        assert resolution_s in (1,15,(1,15),(15,1),[1,15],[15,1])

        # If it's a single int, make it a list so that the next block (looping over the list) universally works.
        if type(resolution_s) == int:
            resolution_s = [resolution_s]

        for res in resolution_s:
            resdir = os.path.join(ETOPO_Generator.empty_tiles_directory, str(res) + "s")
            if not os.path.exists(resdir):
                print("Directory", resdir, "does not exist.")
                assert os.path.exists(os.path.split(resdir)[0])
                os.mkdir(resdir)

            # In this case, the "tile_width_deg" is the same as the "resolution_s".
            # The 1-deg tiles are 1s resolution, 15-deg tiles are 15s resolution.
            # Makes each tile a 3600x3600 file.
            generate_empty_grids.create_empty_tiles(resdir,
                                                    fname_template_tif=ETOPO_Generator.fname_template_tif,
                                                    fname_template_netcdf=None, # TODO: Fill this in once the Netcdf outputs are finalized.
                                                    tile_width_deg=res,
                                                    resolution_s=res,
                                                    also_write_geopackage=True,
                                                    ndv = ETOPO_Generator.etopo_config.etopo_ndv,
                                                    verbose=verbose)

    def create_etopo_geopackages(self, verbose=True):
        """Create geopackages for all the tiles of the ETOPO dataset, at 1s and 15s
        resolutions. SHould run the "create_empty_grids() routine first."""
        # TODO: This code is out-of-date and out-of-sync with the DatasetGeopackage class definition.
        # Change this function to use the ETOPO_Geopackage API instead.
        for res in ("1s", "15s"):
            gtifs_directory = os.path.join(ETOPO_Generator.empty_tiles_directory, res)
            gpkg_fname = ETOPO_Generator.etopo_gpkg_1s if (res=="1s") \
                         else ETOPO_Generator.etopo_gpkg_15s


            dataset_geopackage.DatasetGeopackage(gpkg_fname).create_dataset_geopackage(\
                                           dir_or_list_of_files = gtifs_directory,
                                           geopackage_to_write = gpkg_fname,
                                           recurse_directory = False,
                                           file_filter_regex=r"\.tif\Z",
                                           verbose=verbose)

    def generate_tile_source_dlists(self, source = "all",
                                          active_only=True,
                                          # resolution=1,
                                          verbose=True):
        """For each of the ETOPO dataset tiles, generate a DLIST of all the
        source datasets that overlap that tile, along with their relative ranking.

        This DLIST will be used with the CUDEM waffles command to create the final
        ETOPO dataset from all available sources over each tile.

        If active_only (default), use only datasets listed as "active" in their configfiles.

        Resolution can be 1 or 15 (seconds).
        """
        # Get the dictionary of source datasets
        datasets_dict = self.fetch_etopo_source_datasets(active_only = active_only,
                                                         return_type=dict,
                                                         verbose=verbose)

        if source.strip().lower() == "all":
            datasets_list = sorted(datasets_dict.keys())
        else:
            datasets_list = [source]

        for dset_name in datasets_list:
            print("-------", dset_name, ":")
            dset_obj = datasets_dict[dset_name]
            dset_obj.create_waffles_datalist(verbose=verbose)

    def generate_etopo_source_master_dlist(self,
                                           active_only=True,
                                           verbose=True):
        """Create a master datalist that lists all the datasets from which ETOPO is derived.
        It will be a datalist of datalists from each of the etopo_source_dataset datalists."""
        datasets_dict = self.fetch_etopo_source_datasets(active_only = active_only,
                                                         return_type=dict,
                                                         verbose=verbose)

        dlist_lines = []
        DATALIST_DTYPE = -1 # The Waffles code for a datalist used recursively.
        for (dset_name, dset_obj) in datasets_dict.items():
            dset_dlist_fname = dset_obj.get_datalist_fname()
            dset_ranking_number = dset_obj.get_dataset_ranking_score()
            text_line = "{0} {1} {2}".format(dset_dlist_fname, DATALIST_DTYPE, dset_ranking_number)
            dlist_lines.append(text_line)

        dlist_text = "\n".join(dlist_lines)

        with open(self.etopo_config.etopo_sources_datalist, 'w') as f:
            f.write(dlist_text)
            f.close()
            if verbose:
                print(self.etopo_config.etopo_sources_datalist, "written.")

        return self.etopo_config.etopo_sources_datalist

    def fetch_etopo_source_datasets(self, active_only=True, verbose=True, return_type=dict):
        """Look through the /src/datasets/ directory, and get a list of the input datasets to use.
        This list will be saved to self.source_datasets, and will be instances of
        the datasets.source_dataset class, or sub-classes thereof.
        Each subclass must be called source_dataset_XXXX.py where XXX refers to the name
        of the dataset specified by the folder it's in. For instance, the CopernicusDEM
        is in the folder /src/datasets/CopernicusDEM/source_dataset_CopernicusDEM.py file.

        Any dataset that does not have a sub-class defined in such a file, will simply
        be instantiated with the parent class defined in source_dataset.py. Also if
        the dataset's 'is_active()' routeine is False, the dataset will be skipped.

        type_out can  be list, tuple, or dict. If dict, the keys will be the dataset names, the value the objects.
        For list and tuple, just a list of the objects is fine.
        """
        # The the list of dataset objects already exists, just return it.
        if self.datasets_dict:
            return self.datasets_dict

        # Get this directory.
        datasets_dir = os.path.join(ETOPO_Generator.project_basedir, 'src', 'datasets')
        assert os.path.exists(datasets_dir) and os.path.isdir(datasets_dir)

        # Get all the subdirs within this directory.
        subdirs_list = sorted([fn for fn in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, fn))])

        dataset_objects_dict = dict()
        # Loop through all the sub-directories, check requirements for an active dataset class.
        for dataset_name in subdirs_list:
            subdir = os.path.join(datasets_dir, dataset_name)
            dataset_class_name = "source_dataset_{0}".format(dataset_name)
            module_fname = os.path.join(subdir, dataset_class_name + ".py")

            if os.path.exists(module_fname):
                # Create the import path of the module, relative to the /src
                # directory (which is imported above)
                file, pathname, desc = imp.find_module(dataset_class_name, [subdir])
                module = imp.load_module(dataset_class_name, file, pathname, desc)

                # Create an object from the file.
                dataset_object = getattr(module, dataset_class_name)()
                if (not active_only) or dataset_object.is_active():
                    dataset_objects_dict[dataset_name] = dataset_object
                else:
                    # If the dataset is not currently active and we're only fetching the acive ones, just forget it.
                    continue

            # The the dataset doesn't have a class module, forget it.
            else:
                continue

        self.datasets_dict = dataset_objects_dict

        if verbose:
            print("Datasets active:", ", ".join([d for d in sorted(self.datasets_dict.keys())]))

        if return_type == dict:
            return self.datasets_dict
        else:
            return return_type(self.datasets_dict.values())

if __name__ == "__main__":
    EG = ETOPO_Generator()
    # EG.generate_tile_source_dlists(source="all",active_only=True,verbose=True)
    EG.generate_etopo_source_master_dlist()
    # EG.create_empty_grids()
    # EG.create_etopo_geopackages()
    # EG.create_intermediate_grids(source="FABDEM", resolution_s=1)
