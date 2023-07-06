import os
import zipfile
import argparse
import sys

from sizeof_format import sizeof_fmt
import traverse_directory

z_algo_dict = {"deflate": zipfile.ZIP_DEFLATED,
               "bz2"    : zipfile.ZIP_BZIP2,
               "bzip2"  : zipfile.ZIP_BZIP2,
               "bzip"   : zipfile.ZIP_BZIP2,
               "stored" : zipfile.ZIP_STORED,
               "lzma"   : zipfile.ZIP_LZMA}

def zip_file_or_dir(file_or_dir,
                    match_str="",
                    outfile=None,
                    include_zips=False,
                    recurse=False,
                    reverse=False,
                    algorithm="lmza",
                    compress_level=5,
                    overwrite=False,
                    delete_originals=False,
                    verbose=True):
    """Zip all files in a directory matching the match_str, or just the one file."""
    if not os.path.exists(file_or_dir):
        raise FileNotFoundError(file_or_dir)

    # If it's a directory, get all the files in the directory.
    if os.path.isdir(file_or_dir):
        file_list = traverse_directory.list_files(file_or_dir, regex_match=match_str, depth=-1 if recurse else 0)
        if not include_zips:
            file_list = [fn for fn in file_list if os.path.splitext(fn)[1].lower() != ".zip"]

        if reverse:
            file_list.reverse()

    # If it's just one file, make a list of just that file to iterate over.
    else:
        file_list = [file_or_dir]

    if verbose and len(file_list) > 1:
        print(f"Found {len(file_list)} files to compress.")
        print("Deleting" if delete_originals else "Keeping", "original files.")


    zname_list = [None] * len(file_list)
    for i, fname in enumerate(file_list):

        base, ext = os.path.splitext(fname)
        if len(file_list) == 1 and outfile:
            zname = outfile
        else:
            zname = base + (".zip" if (ext.lower() != ".zip") else (ext + ".zip"))

        zname_list[i] = zname

        if verbose:
            print(("{0}/{1} ".format(i + 1, len(file_list)) if (len(file_list) > 1) else "") + os.path.basename(zname),
                  end="")
            sys.stdout.flush()

        if not overwrite and os.path.exists(zname):
            # First, check if it's a valid zipfile. If not, remove it.
            # (This could have happened if it was partially written the process was killed.)
            if not zipfile.is_zipfile(zname):
                os.remove(zname)
            else:
                if verbose:
                    print(" already exists.")
                continue

        try:
            with zipfile.ZipFile(zname,
                                 mode='w' if overwrite else 'x',
                                 compression=z_algo_dict[algorithm.strip().lower()],
                                 compresslevel=compress_level,
                                 allowZip64=True) as myzip:
                myzip.write(fname)
                myzip.close()
        except KeyboardInterrupt as e:
            if os.path.exists(zname) and not zipfile.is_zipfile(zname):
                os.remove(zname)
            raise e

        if verbose:
            orig_size = os.path.getsize(fname)
            zip_size = os.path.getsize(zname)
            compression = (orig_size - zip_size) / orig_size
            print(" written, {0} -> {1}, {2:0.2f}% compressed.".format(sizeof_fmt(orig_size, decimal_digits=2),
                                                                       sizeof_fmt(zip_size, decimal_digits=2),
                                                                       compression * 100.))

        # Verify the file is valid.
        is_zipfile_good = zipfile.is_zipfile(zname)
        if verbose and not is_zipfile_good:
            print("ERROR: {0} is NOT a valid Zipfile.".format(os.path.basename(zname)) + \
                  (" Not deleting original." if delete_originals else ""))

        if delete_originals and is_zipfile_good:
            os.remove(fname)
            if verbose:
                print("  " + os.path.basename(fname), "removed.")

    if len(zname_list) == 1:
        return zname_list[0]
    else:
        return zname_list


def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Zip a file, or all the files in a directory, separately.")
    parser.add_argument("FILE_OR_DIR", help="File or directory to zip.")
    parser.add_argument("-match_str", "-s", type=str, default="", help="Regex string to match files. Default: Find all files in directory. Unused unless a directory is given.")
    parser.add_argument("-outfile", "-o", default=None, help="Outfile to write. Default: use existing filename with .zip extension. Ignored if a directory is given.")
    parser.add_argument("-algorithm", "-a", type=str, default="deflate", help="Zip algorithm to use. Choices are 'delfate', 'bz2', 'lzma', 'stored' (no compression). Default 'deflate'.")
    parser.add_argument("-level", "-l", type=int, default=5, help="Compression level. Ignored for algorithms 'lzma' and 'stored'. Integer 0-9 for 'defalte', 1-9 for 'bz2'. Default 5.")
    parser.add_argument("--recurse", "-rc", default=False, action="store_true", help="Recurse into sub-directories. Default: Only look in local directory. Ignored unless a directory is given.")
    parser.add_argument("--include_zips", "-i", default=False, action="store_true", help="Include zipfiles (.zip), if they match the string. Default: Omit all existing .zip files. Only used if a directory is given.")
    parser.add_argument("--delete_originals", "-d", default=False, action="store_true", help="Delete the original file after the zipfile has been created and verified to be valid. Default: Keep them.")
    parser.add_argument("--reverse", "-rv", default=False, action="store_true", help="Process tiles in the reverse order. Useful if putting more than one process on a directory.")
    parser.add_argument("--overwrite", "-ov", default=False, action="store_true", help="Overwrite existing zipfiles. Default: Leave existing files there and ignore.")
    parser.add_argument("--quiet", "-q", default=False, action="store_true", help="Suppress output.")

    return parser.parse_args()

if "__main__" == __name__:
    args = define_and_parse_args()
    zip_file_or_dir(args.FILE_OR_DIR,
                    outfile=args.outfile,
                    match_str=args.match_str,
                    algorithm=args.algorithm,
                    compress_level=args.level,
                    include_zips=args.include_zips,
                    recurse=args.recurse,
                    reverse=args.reverse,
                    overwrite=args.overwrite,
                    delete_originals=args.delete_originals,
                    verbose=not args.quiet)