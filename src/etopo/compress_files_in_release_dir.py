import os
import subprocess
import numpy

#####################################################
# Code snippet to import the base directory into
# PYTHONPATH to aid in importing from all the other
# modules in other subdirs.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
#####################################################
import utils.traverse_directory
import utils.parallel_funcs
import utils.configfile
etopo_config = utils.configfile.config()

src_dir = os.path.join(etopo_config.project_base_directory, "data", "ETOPO_2022_release")
dst_dir = os.path.join(etopo_config.project_base_directory, "data", "ETOPO_2022_release_compressed")

# Command for deflating floating-point geotiffs
# gdal_translate -of GTiff -co COMPRESS=DEFLATE -co PREDICTOR=3 -co TILED=YES src.tif dst.tif
source_fp_tifs = utils.traverse_directory.list_files(src_dir, regex_match=r'((surface)|(geoid)|(bed))\.tif\Z', include_base_directory=False)
source_fp_args = ([["gdal_translate",
                    "-of", "GTiff",
                    "-co", "COMPRESS=DEFLATE",
                    "-co", "PREDICTOR=3",
                    "-co", "TILED=YES",
                    os.path.join(src_dir, fn), os.path.join(dst_dir, fn)] for fn in source_fp_tifs])

# Command for deflating integer geotiffs, same but without "PREDICTOR" arg
# gdal_translate -of GTiff -co COMPRESS=DEFLATE -co TILED=YES src.tif dst.tif
source_int_tifs = utils.traverse_directory.list_files(src_dir, regex_match=r'sid\.tif\Z', include_base_directory=False)
source_int_args = ([["gdal_translate",
                    "-of", "GTiff",
                    "-co", "COMPRESS=DEFLATE",
                    "-co", "TILED=YES",
                    os.path.join(src_dir, fn), os.path.join(dst_dir, fn)] for fn in source_int_tifs])

# Command for deflating netcdfs
# nccopy -d 5 -s src.nc dst.nc
source_nc = utils.traverse_directory.list_files(src_dir, regex_match=r'\.nc\Z', include_base_directory=False)
source_nc_args = ([["nccopy",
                    "-d", "5",
                    "-s", os.path.join(src_dir, fn),
                    os.path.join(dst_dir, fn)] for fn in source_nc])

# Make sure all target dirs are there.
for fn in (source_fp_tifs + source_int_tifs + source_nc):
    dirname = os.path.join(dst_dir, os.path.dirname(fn))
    if not os.path.exists(dirname):
        print(dirname)
        os.makedirs(dirname)


all_outfiles = [os.path.join(dst_dir, fn) for fn in (source_fp_tifs + source_int_tifs + source_nc)]
all_args = source_fp_args + source_int_args + source_nc_args

# FIX JUST THESE FILES, WHICH WERE INCORRECTLY TRANSLATED BEFORE.
# Look for the fucking files that are less than 18K in size. These are the broken ones.
# fix_mask = [(True if (os.path.getsize(fn) < 18000) else False) for fn in all_outfiles]
fix_mask = [(True if (fn.find("15s_N60W030_bed.nc") > -1) else False) for fn in all_outfiles]
all_outfiles = [fn for fn,mask_val in zip(all_outfiles, fix_mask) if mask_val]
all_args = [arg for arg,mask_val in zip(all_args, fix_mask) if mask_val]

print(len(all_outfiles), "files to fix.")
for fn in all_outfiles:
    print(fn)

    os.remove(fn)
# foobar

utils.parallel_funcs.process_parallel(subprocess.run,
                                      [[ar] for ar in all_args],
                                      kwargs_list={"capture_output": True},
                                      outfiles = all_outfiles,
                                      max_nprocs = 18)

