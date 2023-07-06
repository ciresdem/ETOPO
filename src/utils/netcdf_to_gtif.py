from osgeo import gdal
import argparse
import os
import subprocess

import traverse_directory


def convert_netcdf_to_gtiff(file_or_dir, search_str=r'\.nc\Z', varname="1", recurse=False, verbose=True):
    """Convert a netCDF or directory full of netCDFs to GeoTiffs.

    Will only produce one geotiff per netcdf.
    Return either a list of geotiffs created, or one string of the geotiff (if only one was created)."""
    flist = []
    if os.path.exists(file_or_dir):
        if os.path.isdir(file_or_dir):
            flist = traverse_directory.list_files(file_or_dir,
                                                  depth=-1 if recurse else 0,
                                                  regex_match=search_str)
        else:
            flist = [file_or_dir]
    else:
        if verbose:
            print(file_or_dir, "does not exist.")
            return

    outnames = []
    for i,fname in enumerate(flist):
        outfn = os.path.splitext(fname)[0] + ".tif"
        assert outfn != fname
        # Wrap the file and variable names in quotes if they contain any spaces (probably not, but good to check).
        fname_to_use = ('"' + fname + '"') if fname.find(" ") > -1 else fname
        varname_to_use = ('"' + varname + '"') if varname.find(" ") > -1 else varname
        # Construct the gdal_translate command.
        gdal_cmd = ["gdal_translate",
                    "NETCDF:{0}:{1}".format(fname_to_use, varname_to_use),
                    outfn]
        # Run the gdal_translate command.
        subprocess.run(gdal_cmd, capture_output=not verbose, cwd=os.path.dirname(fname))
        # Verify the output was written.
        if os.path.exists(outfn):
            outnames.append(outfn)
            if verbose:
                print(("{0}/{1} ".format(i+1, len(flist)) if (len(flist) > 1) else "") + outfn, "written.")

    if len(outnames) == 1:
        return outnames[0]
    else:
        return outnames

def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Convert a netCDF file to a GeoTiff, or a directory of netCDFs to GeoTiffs.")
    parser.add_argument("FILE_OR_DIR", type=str, help="netCDF file, or a directory of netCDF files.")
    parser.add_argument("-search_str", "-s", type=str, default=r"\.nc\Z",
                        help="Regex search string to find netCDFs. Default to r'\.nc\Z', to look for '.nc' at the end of the file.")
    parser.add_argument("-varname", "-v", type=str, default="1",
                        help="Variable name to pull from the netCDF. Name or number. Default '1' (for band 1 of a single-variable file).")
    parser.add_argument("--recurse", "-r", action="store_true", default=False,
                        help="If a directory is given, search recursively through subdirectories.")
    parser.add_argument("--quiet", '-q', action="store_true", default=False, help="Run silently.")
    return parser.parse_args()

if "__main__" == __name__:
    args = define_and_parse_args()
    convert_netcdf_to_gtiff(args.FILE_OR_DIR,
                            search_str=args.search_str,
                            varname=args.varname,
                            recurse=args.recurse,
                            verbose=not args.quiet)
