# -*- coding: utf-8 -*-
"""map_finished_tiles_to_release_directory.py - creating symlinks to the final output locations in /data/ETOPO_2022_release."""

import os
import shutil

#####################################################
# Code snippet to import the base directory into
# PYTHONPATH to aid in importing from all the other
# modules in other subdirs.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
#####################################################
import utils.configfile
etopo_config = utils.configfile.config()

src_dir = etopo_config._abspath(etopo_config.etopo_finished_tiles_directory)
out_dir = os.path.abspath(os.path.join(etopo_config._abspath(etopo_config.etopo_finished_tiles_directory), "..", "ETOPO_2022_release"))

date_str = "2022.09.19"

file_mapping_dict = {os.path.join(src_dir, "60s", date_str, "ETOPO_2022_v1_60s_S90W180_2022.09.19.tif"):
                        os.path.join(out_dir, "gtif", "60s", "60s_surface_elev_gtif", "ETOPO_2022_60s.tif"),
                     os.path.join(src_dir, "60s", date_str, "ETOPO_2022_v1_60s_S90W180_bed.tif"):
                        os.path.join(out_dir, "gtif", "60s", "60s_bed_elev_gtif", "ETOPO_2022_60s_bed.tif"),
                     os.path.join()}