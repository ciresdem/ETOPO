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

class ETOPO_Generator:
    # Tile naming conventions. ETOPO_2022_v1_N15E175.tif, e.g.
    fname_template_tif = r"ETOPO_2022_v1_{0:d}s_{1:s}{2:02d}{3:s}{4:03d}.tif"
    fname_template_netcdf = r"ETOPO_2022_v1_{0:d}s_{1:s}{2:02d}{3:s}{4:03d}.nc"

    # The base directory of the project is two levels up. Retrieve the absolute path to it on this machine.
    # This file resides in [project_basedir]/src/etopo
    project_basedir = os.path.abspath(os.path.join(os.path.split(__file__)[0], "..", ".."))

    etopo_configfile = utils.configfile.config(os.path.join(project_basedir, "config.ini"), )

    # Note, still need to add resolution "1s" or "15s" directory.
    empty_tiles_directory = etopo_configfile.etopo_empty_tiles_directory
    intermediate_tiles_directory = etopo_configfile.etopo_intermediate_tiles_directory # Same as above
    finished_tiles_directory = etopo_configfile.etopo_finished_tiles_directory # Same as above

    def __init__(self):
        """Read the configuration file, get the info about our grid locations."""
        self.datasets = None

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
                                                    ndv = ETOPO_Generator.etopo_configfile.etopo_ndv,
                                                    verbose=verbose)

    def create_intermediate_grids(self, source = "all",
                                        resolution_s=1,
                                        overwrite=False,
                                        verbose=True):
        """For the given source dataset (or all active datasets if source="all"),
        regride the source grids into the intermediate_tiles directory for each source
        dataset. Additionally, a "ranking" grid file will be generated for each intermediate
        tile, for use in the final gridding.

        resolution_s is the resolution of the destination grids to create.
        ETOPO currently supports 1 or 15.

        If 'overwrite', overwrite existing datasets. Defaults to False,
        which will skip grids already created and just create new ones.
        """
        datasets = self.create_etopo_source_dataset_objects(verbose=verbose)
        if source.strip().lower() != "all":
            # If we've selected only one dataset, use that.
            datasets = [dset for dset in datasets if dset.dataset_name == source]
            if len(datasets) == 0:
                if verbose:
                    print("In ETOPO_Generator::create_intermediate_grids():\n" +
                          "\tNo active dataset exists named '{0}'. Check the dataset".format(source) +
                          " spelling, and to see wheter the 'is_active' flag is set in" +
                          " {0}_config.ini.".format(source)
                          )
                return None
            # Should be no more than one dataset by any given name. Do a sanity check here.
            assert len(datasets) == 1

        for dataset_obj in datasets:
            # Create the intermediate grids for each dataset.
            dataset_obj.create_intermediate_grids(include_rankings = True,
                                                  resolution_s = resolution_s,
                                                  overwrite = overwrite,
                                                  verbose = verbose)

        return

    def delete_intermediate_files(self):
        """Delete all the intermediate tiles generated from source datasets."""
        tile_dir = self.intermediate_tiles_directory
        # We will clear out the "1s" and "15s" directories, but leave those put.
        subdirs = [os.path.join(tile_dir, "1s"), os.path.join(tile_dir, "15s")]

        # First, clear out any files that aren't in the designated subdirs directories.
        files_in_base_dir = [os.path.join(tile_dir, fn) for fn in os.listdir(tile_dir) if fn not in ("1s", "15s")]
        for path in files_in_base_dir:
            shutil.rmtree(path)

        for subdir in subdirs:
            if os.path.exists(subdir):
                # Delete all the files and directories in each subdir.
                files_in_subdir = [os.path.join(subdir, fn) for fn in os.listdir(subdir)]
                for path in files_in_subdir:
                    shutil.rmtree(path)
            else:
                # If for some reason the '1s' or '15s' directory doesn't exist, make it.
                os.mkdir(subdir)

        return

    def create_etopo_source_dataset_objects(self, verbose=True):
        """Look through the /src/datasets/ directory, and get a list of the input datasets to use.
        This list will be saved to self.source_datasets, and will be instances of
        the datasets.source_dataset class, or sub-classes thereof.
        Each subclass must be called source_dataset_XXXX.py where XXX refers to the name
        of the dataset specified by the folder it's in. For instance, the CopernicusDEM
        is in the folder /src/datasets/CopernicusDEM/source_dataset_CopernicusDEM.py file.

        Any dataset that does not have a sub-class defined in such a file, will simply
        be instantiated with the parent class defined in source_dataset.py. Also if
        the dataset's 'is_active()' routeine is False, the dataset will be skipped.
        """
        # The the list of dataset objects already exists, just return it.
        if self.datasets:
            return self.datasets

        # Get this directory.
        datasets_dir = os.path.join(ETOPO_Generator.project_basedir, 'src', 'datasets')
        assert os.path.exists(datasets_dir) and os.path.isdir(datasets_dir)

        # Get all the subdirs within this directory.
        subdirs_list = sorted([fn for fn in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, fn))])

        dataset_objects_list = []
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
                if dataset_object.is_active():
                    dataset_objects_list.append(dataset_object)
                else:
                    # If the dataset is not currently active, just forget it.
                    continue

            # The the dataset doesn't have a class module, forget it.
            else:
                continue

        self.datasets = dataset_objects_list

        if verbose:
            print("Datasets active:", ", ".join([d.dataset_name for d in self.datasets]))

        return self.datasets

if __name__ == "__main__":
    EG = ETOPO_Generator()
    EG.create_empty_grids()
