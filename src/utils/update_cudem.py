# -*- coding: utf-8 -*-
"""update_cudem.py -- One-line utility for updating the git repository for CUDEM on my machine.
Handier than having to run it from the local git directory repeatedly.

Relies on my cudem repository to be in ~/git/cudem ."""

import os
import subprocess

cudem_dir = os.path.join(os.path.expanduser("~"), "git", "cudem")

subprocess.run(["git", "pull"], cwd=cudem_dir)
subprocess.run(["python", "install_CUDEM.py"], cwd=cudem_dir)
