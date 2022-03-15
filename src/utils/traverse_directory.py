# -*- coding: utf-8 -*-

"""Function for traversing and getting all the files from a directory, including recursively into sub-directories."""

import re
import os

def list_files(dirname, regex_match = "\A\w*", ordered=True, include_base_directory = True):
    file_list = _list_files_recurse(dirname, regex_match=regex_match)

    if not include_base_directory:
        file_list = [fn[len(dirname):].lstrip("/\\") for fn in file_list]

    if ordered:
        file_list.sort()

    return file_list

def _list_files_recurse(dirname, regex_match = None):
    fpath_list = os.listdir(dirname)
    file_list = []

    for entryname in fpath_list:
        fpath = os.path.join(dirname, entryname)
        if os.path.isdir(fpath):
            file_list.extend(_list_files_recurse(fpath, regex_match=regex_match))
        elif (regex_match is None) or (re.search(regex_match, entryname) != None):
            file_list.append(fpath)
    return file_list

if __name__ == "__main__":
    pass
    # Just test this out:
    # dirname = "/home/mmacferrin/Research/DATA/DEMs/AW3D30/data/tiles"
    # regex_search = "\A\w*DSM\.tif"
    # file_list = list_files(dirname, regex_match=regex_search, ordered=True, include_directory=False)
    # print(len(file_list))
    # print(file_list[:10])
