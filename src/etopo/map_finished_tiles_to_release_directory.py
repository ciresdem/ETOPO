# -*- coding: utf-8 -*-
"""map_finished_tiles_to_release_directory.py - creating symlinks to the final output locations in /data/ETOPO_2022_release."""

import os
import shutil
import argparse
import re
import subprocess

#####################################################
# Code snippet to import the base directory into
# PYTHONPATH to aid in importing from all the other
# modules in other subdirs.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
#####################################################
import utils.traverse_directory
import utils.sizeof_format
import utils.configfile
etopo_config = utils.configfile.config()

src_base_dir = etopo_config._abspath(etopo_config.etopo_finished_tiles_directory)
out_base_dir = os.path.abspath(os.path.join(etopo_config._abspath(etopo_config.etopo_finished_tiles_directory), "..", "ETOPO_2022_release"))

def provide_destination_folder(tile_path_orig, out_dir=out_base_dir):
    """Map an old dtirectory to a new one, using the tile filename.

    If it's a file we don't release (such a weight, or an inf, etc), return None."""
    tile_fname = os.path.basename(tile_path_orig)

    # We're not releasing the "weights" ("_w") files. Just skip them.
    if re.search("_w(?=[_\.])", tile_fname) is not None:
        return None

    tile_resolution = int(re.search("(?<=_)\d{1,2}(?=s_)", tile_fname).group())
    assert tile_resolution in (1, 15, 30, 60)
    if tile_resolution == 1:
        return None

    is_bed = tile_fname.find("_bed") >= 0
    is_sid = tile_fname.find("_sid") >= 0
    is_geoid = tile_fname.find("_geoid") >= 0
    file_ext = os.path.splitext(tile_path_orig)[1]
    if file_ext == ".tif":
        file_type = "gtif"
    elif file_ext == ".nc":
        file_type = "netcdf"
    else:
        return None

    # 30 and 60s tiles will not have output "sid" files. If this is one that was needlessly generated, return None
    if tile_resolution in (30,60) and is_sid:
        return None

    fout_dir = os.path.join(out_dir,
                            file_type,
                            "{0}s".format(tile_resolution),
                            "{0}s_{1}{2}_{3}".format(tile_resolution,
                                                     "geoid" if is_geoid else ("bed" if is_bed else "surface"),
                                                     "" if is_geoid else ("_sid" if is_sid else "_elev"),
                                                     file_type)
                            )

    return fout_dir

def convert_tile_fname(tilename: str,
                       also_strip_date: bool = True):
    "The lat/lon TileID numbers currently map to the southwest corner of the tile. Rename a tile so it maps to the northwest corner instead, without changing the file contents at all."""
    tile_fname = os.path.basename(tilename)
    if os.path.splitext(tile_fname)[1] not in (".nc", ".tif") or re.search("_w\.tif\Z", tile_fname) is not None:
        return None

    tile_ID_orig = re.search("(?<=_)[NS](\d{2})[EW](\d{3})(?=[_\.])", tile_fname).group()
    resolution_s = int(re.search("(?<=_)\d{1,2}(?=s_)", tile_fname).group())
    if resolution_s in (1,15):
        tile_lat: int = (1 if tile_ID_orig[0] == "N" else -1) * int(tile_ID_orig[1:3])
        tile_lat = tile_lat + resolution_s
        tile_ID_new = ("N" if tile_lat >= 0 else "S") + "{0:02d}".format(int(abs(tile_lat))) + tile_ID_orig[3:]

    else:
        assert ((resolution_s in (30,60)) and (tile_ID_orig == "S90W180"))
        tile_ID_new = "N90W180"

    tile_fname_new = tile_fname.replace(tile_ID_orig, tile_ID_new)

    if also_strip_date:
        tile_datestr_re = re.search("_(\d{4})\.(\d{2})\.(\d{2})", tile_fname_new)
        if tile_datestr_re is not None:
            tile_fname_new = tile_fname_new.replace(tile_datestr_re.group(), "")

    # Add "_surface" to surface tiles (both elev and sid).
    if tile_fname_new.find("_bed") == -1 and tile_fname_new.find("_geoid") == -1:
        try:
            assert re.search("[NS](\d{2})[EW](\d{3})((_sid)?)\.((tif)|(nc))\Z", tile_fname_new) is not None
        except AssertionError as e:
            print(tile_fname_new)
            raise e
        if tile_fname_new.find("_sid.tif") == -1:
            tile_fname_new = tile_fname_new.replace(".tif", "_surface.tif")
        else:
            tile_fname_new = tile_fname_new.replace("_sid.tif", "_surface_sid.tif")

    tile_directory_new = provide_destination_folder(tilename)

    if tile_directory_new is None:
        return None
    else:
        return os.path.join(tile_directory_new, tile_fname_new)

def create_output_directory_blank(out_base = out_base_dir,
                                  delete_old = True,
                                  skip_pdfs = True):
    if delete_old:
        for subitem in os.listdir(out_base):
            if skip_pdfs and os.path.splitext(subitem)[1] == ".pdf":
                continue
            rm_cmd = ["rm", "-rf", os.path.join(out_base, subitem)]
            subprocess.run(rm_cmd, capture_output=True)

    for ftype in ("gtif", "netcdf"):
        ftype_dir = os.path.join(out_base, ftype)
        if not os.path.exists(ftype_dir):
            os.mkdir(ftype_dir)

        for ressubdir in ("15s", "30s", "60s"):
            res_dir = os.path.join(ftype_dir, ressubdir)
            if not os.path.exists(res_dir):
                os.mkdir(res_dir)

            bed_elev_dir = os.path.join(res_dir, "{0}_bed_elev_{1}".format(ressubdir, ftype))
            if os.path.exists(bed_elev_dir):
                for fn in os.listdir(bed_elev_dir):
                    os.remove(os.path.join(bed_elev_dir, fn))
            else:
                os.mkdir(bed_elev_dir)

            surf_elev_dir = os.path.join(res_dir, "{0}_surface_elev_{1}".format(ressubdir, ftype))
            if os.path.exists(surf_elev_dir):
                for fn in os.listdir(surf_elev_dir):
                    os.remove(os.path.join(surf_elev_dir, fn))
            else:
                os.mkdir(surf_elev_dir)

            geoid_dir = os.path.join(res_dir, "{0}_geoid_{1}".format(ressubdir, ftype))
            if os.path.exists(geoid_dir):
                for fn in os.listdir(geoid_dir):
                    os.remove(os.path.join(geoid_dir, fn))
            else:
                os.mkdir(geoid_dir)

            if ressubdir == "15s":
                bed_sid_dir = os.path.join(res_dir, "{0}_bed_sid_{1}".format(ressubdir, ftype))
                if os.path.exists(bed_sid_dir):
                    for fn in os.listdir(bed_sid_dir):
                        os.remove(os.path.join(bed_sid_dir, fn))
                else:
                    os.mkdir(bed_sid_dir)

                surf_sid_dir = os.path.join(res_dir, "{0}_surface_sid_{1}".format(ressubdir, ftype))
                if os.path.exists(surf_sid_dir):
                    for fn in os.listdir(surf_sid_dir):
                        os.remove(os.path.join(surf_sid_dir, fn))
                else:
                    os.mkdir(surf_sid_dir)

    return

def generate_output_directory_and_files(src_subdir = None,
                                        delete_old : bool = True,
                                        skip_pdfs : bool = True,
                                        print_only : bool = False,
                                        summarize: bool = False):
    if not print_only:
        create_output_directory_blank(delete_old=delete_old, skip_pdfs=skip_pdfs)

    source_tiles = []
    for resolution_s in (15,30,60):
        src_dir = os.path.join(src_base_dir, "{0}s".format(resolution_s))
        if src_subdir:
            src_dir = os.path.join(src_dir, src_subdir)

        if os.path.exists(src_dir):
            source_tiles.extend(utils.traverse_directory.list_files(src_dir))

        geoid_dir = os.path.join(etopo_config.etopo_geoid_directory, "{0}s".format(resolution_s))
        source_tiles.extend(utils.traverse_directory.list_files(geoid_dir))

    src_fnames_to_use = []
    dest_fnames_to_use = []
    for src_fname in source_tiles:
        dest_fname = convert_tile_fname(src_fname)
        if dest_fname is None:
            continue
        src_fnames_to_use.append(src_fname)
        dest_fnames_to_use.append(dest_fname)

    for i, (src, dest) in enumerate(zip(src_fnames_to_use, dest_fnames_to_use)):
        if print_only:
            print("{0}/{1}".format(i+1, len(src_fnames_to_use)), src, "-->\n   ", dest)
        else:
            os.link(src, dest)

    if summarize:
        summarize_dest_dir_contents(out_base_dir)

    return

def summarize_dest_dir_contents(dest_dir : str) -> None:
    all_files = utils.traverse_directory.list_files(dest_dir)
    dir_f_counts = {}
    dir_f_sizes = {}
    for fn in all_files:
        dn = os.path.abspath(os.path.dirname(fn))
        fs = os.path.getsize(fn)
        if dn in dir_f_counts:
            dir_f_counts[dn] = dir_f_counts[dn] + 1
            dir_f_sizes[dn] = dir_f_sizes[dn] + fs
        else:
            dir_f_counts[dn] = 1
            dir_f_sizes[dn] = fs

    # Get a list of all the unique directories, sorted by length (shortest first)
    list_of_dirs = sorted(list(dir_f_counts.keys()))
    for dn in list_of_dirs:
        print(dn + ",", dir_f_counts[dn], "files,", utils.sizeof_format.sizeof_fmt(dir_f_sizes[dn]))
    print(len(all_files), "total files,", utils.sizeof_format.sizeof_fmt(sum(list(dir_f_sizes.values()))) + ".")
    return

def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Using symlinks, map the newly-generated etopo output tiles into the ETOPO_Release directory for handing off to NOAA.")
    parser.add_argument("-subdir", default=None, help="Sub-directory to find tiles. Usually a YYYY.MM.DD datestring.")
    parser.add_argument("--print_only", default=False, action="store_true", help="Just print the file mappings without actually generating the new links.")

    return parser.parse_args()

if __name__ == "__main__":
    args = define_and_parse_args()

    generate_output_directory_and_files(src_subdir = args.subdir,
                              delete_old = True,
                              skip_pdfs = True,
                              print_only = args.print_only,
                              summarize=True)
