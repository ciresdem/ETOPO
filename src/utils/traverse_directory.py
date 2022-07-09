# -*- coding: utf-8 -*-

"""Function for traversing and getting all the files from a directory, including recursively into sub-directories."""

import re
import os
import argparse

def list_files(dirname, regex_match = r"\A\w*", ordered=True, include_base_directory = True):
    file_list = _list_files_recurse(dirname, regex_match=regex_match)

    if not include_base_directory:
        file_list = [fn[len(dirname):].lstrip(os.sep) for fn in file_list]

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

def define_and_parse_args():
    parser = argparse.ArgumentParser(description="A utility for recursively finding (or deleting) files in a directory and sub-directories.")
    parser.add_argument("DIR", type=str, default=os.getcwd(), help="Directory to search within. Default: Current working directory.")
    parser.add_argument("-text", "-t", type=str, default=r"\A\w*", help="Regular expression to match.")
    parser.add_argument("--delete", "-d", action="store_true", default=False, help="Delete the files matching the search query. NOTE: Suggest to call first without this option to see what will be deleted, then re-call with -d.")

    return parser.parse_args()

if __name__ == "__main__":
    args = define_and_parse_args()
    fnames = list_files(args.DIR,
                        regex_match=args.text)
    for fn in fnames:
        if args.delete:
            print("Removing", fn)
            os.remove(fn)
        else:
            print(fn)
