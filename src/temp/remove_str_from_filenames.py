
import argparse
import os
import re

def rename_files(dirname, file_id_regex, remove_str):
    file_list = [fn for fn in os.listdir(dirname) if re.search(file_id_regex, fn) is not None]
    for fname in file_list:
        fname_new = fname.replace(remove_str, "")
        if fname == fname_new:
            continue
        print(fname, "->", fname_new)
        os.rename(os.path.join(dirname, fname), os.path.join(dirname, fname_new))

def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Change filenames in a directory by removing any instances of a sub-string, such as '_egm2008' or similar.")
    parser.add_argument("-dirname", default=".", help="Directory to look for filenames.")
    parser.add_argument("-file_id_regex", default=r"\.tif\Z", help="Regex of files to search for. Default r'\.tif\Z' (.tif files only).")
    parser.add_argument("-remove_str", default="_egm2008", help="String to remove from filenames. Default _egm2008.")

    return parser.parse_args()

if __name__ == "__main__":
    args = define_and_parse_args()
    rename_files(args.dirname, args.file_id_regex, args.remove_str)
