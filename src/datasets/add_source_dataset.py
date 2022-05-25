# -*- coding: utf-8 -*-

"""Code for adding a source dataset.
    STEPS: 1) Create a folder in this directory named after the dataset.
           2) Run populate_subset_folder(foldername), which will create a template
              subclass based on that dataset name, called "source_dataset_[foldername].py"
              It will also copy over a template configfile, titled "[foldername]_config.ini",
              which should be filled in by the user.
           3) Creat a blank __init__.py file in the directory so it can be used as a module.
"""

# Replace all instances of "FOOBAR" with the dataset name.
CLASS_TEMPLATE_TEXT = \
"""# -*- coding: utf-8 -*-

\"""Source code for the FOOBAR ETOPO source dataset class.\"""

import os

THIS_DIR = os.path.split(__file__)[0]

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.etopo_source_dataset as etopo_source_dataset

class source_dataset_FOOBAR(etopo_source_dataset.ETOPO_source_dataset):
    \"""Look in "src/datasets/etopo_source_dataset.py" to get base class definition.\"""
    def __init__(self,
                 configfile = os.path.join(THIS_DIR, "FOOBAR_config.ini" )):
        \"""Initialize the FOOBAR source dataset object.\"""

        super(source_dataset_FOOBAR, self).__init__("FOOBAR", configfile)
"""

import os
import shutil
import pandas
import dataset_geopackage

###############################################################################
# Import the project /src directory into PYTHONPATH, in order to import all the
# other modules appropriately.
import import_parent_dir; import_parent_dir.import_src_dir_via_pythonpath()
###############################################################################
import etopo.etopo_generator


def populate_dataset_module(dataset_name, overwrite=True):
    local_dir = os.path.split(__file__)[0]
    dataset_dir = os.path.join(local_dir, dataset_name)

    # Make the directory if it doesn't exist
    if not(os.path.exists(dataset_dir)):
        ## Create the dataset folder if it doesn't already exist.
        os.mkdir(dataset_dir)
        print("Directory {0} created.".format(dataset_dir))

    # Insert __init__.py into the directory if it doesn't exist.
    init_file = os.path.join(dataset_dir, "__init__.py")
    if (not os.path.exists(init_file)) or overwrite:
        # Create a new empty file and then close it.
        f = open(init_file, 'w')
        f.close()
        print(init_file, "created.")

    # Insert a copy of the import_parent_dir.py into the dataset directory.
    ipd_file = os.path.join(dataset_dir, "import_parent_dir.py")
    if (not os.path.exists(ipd_file)) or overwrite:
        ipd_source_file = os.path.join(local_dir, "import_parent_dir.py")
        with open(ipd_source_file, 'r') as fs:
            ipd_text = fs.read()
        # Since we're putting this in a sub-directory, must add one ".." addition
        # to the source directory text to get it to the "/src" base directory.
        ipd_text = ipd_text.replace('os.path.join(os.path.split(__file__)[0], ".."',
                                    'os.path.join(os.path.split(__file__)[0], "..", ".."')

        with open(ipd_file, 'w') as fd:
            fd.write(ipd_text)
        print(ipd_file, "created.")

    # Insert a copy of the derived dataset subclass file.
    subclass_file = os.path.join(dataset_dir, "source_dataset_{0}.py".format(dataset_name))
    subclass_text = CLASS_TEMPLATE_TEXT.replace("FOOBAR", dataset_name)
    if (not os.path.exists(subclass_file)) or overwrite:
        with open(subclass_file, 'w') as fd:
            fd.write(subclass_text)
            fd.close()
        print(subclass_file, "created.")

    # Insert a copy of the template [foldername]_config.ini text.
    configname_dest = os.path.join(dataset_dir, dataset_name + "_config.ini")
    configname_src = os.path.join(local_dir, "etopo_source_dataset_template_config.ini")
    assert os.path.exists(configname_src)
    if (not os.path.exists(configname_dest)) or overwrite:
        shutil.copyfile(configname_src, configname_dest)

        print(configname_dest, "created.")

def populate_all_source_datasets(overwrite=True):
    local_dir = os.path.split(__file__)[0]

    # Get all the sub-directories in this directory that don't begin with an underscore
    subdir_list = sorted([d for d in os.listdir(local_dir) if \
                   (os.path.isdir(os.path.join(local_dir, d)) and d[0] != "_")])

    for subdir in subdir_list:
        populate_dataset_module(subdir, overwrite=True)

def make_list_of_ranks_and_ids(output_csv=os.path.join(os.path.dirname(__file__), "ETOPO_dataset_ranks_and_ids.csv")):
    """Go through all the datasets. From their source data, list all the ranks, ids, vdatum_name, vdatum_number, and whether they're active or not."""
    datasets_dict = etopo.etopo_generator.ETOPO_Generator().fetch_etopo_source_datasets(active_only=False, verbose=True, return_type=dict)
    dset_names = sorted(datasets_dict.keys())
    is_active = [None] * len(dset_names)
    ranks = [None] * len(dset_names)
    ids = [None] * len(dset_names)
    vdatum_names = [None] * len(dset_names)
    vdatum_numbers = [None] * len(dset_names)

    for i,dset_name in enumerate(dset_names):
        dset = datasets_dict[dset_name]
        is_active[i] = dset.is_active()
        ranks[i] = dset.config.default_ranking_score
        ids[i] = dset.config.dataset_id_number
        vdatum_names[i] = dset.config.dataset_vdatum_name
        vdatum_numbers[i] = dset.config.dataset_vdatum_epsg

    df = pandas.DataFrame(data = {"name": dset_names,
                                  "is_active": is_active,
                                  "rank": ranks,
                                  "dset_id": ids,
                                  "vdatum_name": vdatum_names,
                                  "vdatum_epsg": vdatum_numbers})

    df.to_csv(output_csv)
    print(output_csv, "written with", len(dset_names), "entries.")


if __name__ == "__main__":
    # populate_all_source_datasets()
    # populate_dataset_module("CUDEM_CONUS")
    # populate_dataset_module("CUDEM_Hawaii")
    # populate_dataset_module("CUDEM_Guam")
    # populate_dataset_module("CUDEM_AmericanSamoa")
    # populate_dataset_module("CUDEM_PuertoRico")
    # populate_dataset_module("CUDEM_VirginIslands")
    # populate_dataset_module("CUDEM_PRVI")
    # populate_dataset_module("global_lakes")
    make_list_of_ranks_and_ids()
