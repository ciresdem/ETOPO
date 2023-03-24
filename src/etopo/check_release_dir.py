import os
import subprocess

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import utils.traverse_directory
import utils.configfile
etopo_config = utils.configfile.config()

def traverse_and_check_dir(dirname = os.path.join(etopo_config.project_base_directory, "data", "ETOPO_2022_release_compressed")):
    """Check all the .tar.gz, .tif, and .nc files for data integrity. Just a double-check here."""
    assert os.path.exists(dirname)

    file_list = utils.traverse_directory.list_files(dirname)
    good_file_count = 0
    bad_file_count = 0
    unhandled_file_count = 0

    for i,fname in enumerate(file_list):
        print("{0}/{1}".format(i + 1, len(file_list)), fname[(len(dirname)+1):], end=" ")

        f_ext = os.path.splitext(fname)[1]
        if f_ext in (".tif",".nc"):
            check_cmd = ["gdalinfo", fname]
        elif f_ext == ".gz":
            assert fname[-len(".tar.gz"):] == ".tar.gz"
            check_cmd = ["gunzip", "-t", fname]
        elif f_ext == ".xml":
            # Skip xmls for now, we'll delete later.
            pass
        else:
            print("ERROR UNHANDLED FILE TYPE.")
            unhandled_file_count += 1
            continue

        p = subprocess.run(check_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if p.returncode == 0:
            good_file_count += 1
            print("ok (0)")
        else:
            bad_file_count += 1
            print("BAD BAD BAD ({0})".format(p.returncode))

        if (((i+1)%100) == 0) or (i+1)==len(file_list):
            print("======", good_file_count, "good files,", bad_file_count, "bad files,", unhandled_file_count, "unhandled files. =======")

    xml_list = utils.traverse_directory.list_files(dirname, regex_match = r"\.xml\Z")
    if len(xml_list) > 0:
        print("Removing", len(xml_list), "XML files from the directory...", end="")
        for xmlf in xml_list:
            os.remove(xmlf)
        print(" Done.")

    return

if __name__ == "__main__":
    traverse_and_check_dir()