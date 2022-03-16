# -*- coding: utf-8 -*-

import sys
import os

def import_parent_dir_via_pythonpath():
    """Attempt to set the parent directory to the PYTHONPATH variable,
    to facilitate importing modules from the subdirectories of this parent directory."""
    parent_dir = os.path.abspath(os.path.join(os.path.split(__file__)[0], ".."))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
# import_parent_dir_via_pythonpath()

# import data_downloads.download_FABDEM
# import icesat2.icepyx_download
