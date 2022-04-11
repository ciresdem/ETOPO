# -*- coding: utf-8 -*-

import icepyx
import argparse
import pexpect
import os
import datetime
import sys
import ast
import numpy
# import subprocess

####################################3
# Include the base /src/ directory of thie project, to add all the other modules.
import import_parent_dir; import_parent_dir.import_src_dir_via_pythonpath()
####################################3
import utils.progress_bar as progress_bar
# Use config file to get the encrypted credentials.
import utils.configfile as configfile
my_config = configfile.config()

def icepyx_download(variables_list=[],
                    dataset_name = "ATL03",
                    region= [-180,-90,180,90],
                    local_dir=None,
                    dates=["2018-01-01","2022-01-01"],
                    overwrite=False,
                    child=False,
                    verbose=False,
                    crop_to_region=False,
                    print_files=False):
    """Download the needed ICESat-2 granules in the dataset given."""
    # Build an argparse namespace to pass to the parent process
    args = argparse.Namespace
    args.variables = variables_list
    args.dataset_name = dataset_name
    if (type(region) != str):
        if (len(region) > 1):
            args.region = ",".join([str(d) for d in region])
        else:
            args.region = str(region)
    else:
        args.region = region
    # The command-line interface doesn't seem to like arguments separated by commas (treats them as separate arguments unless encompassed by brackets.)
    if args.region.find(",") >= 0 and args.region[0] not in ("["):
        args.region = "[" + args.region + "]"
    # Strip out any spaces that may have made it through.
    args.region = args.region.replace(" ", "")
    # print(args.region)

    if (type(dates) != str) and (len(dates) > 1):
        args.dates = ",".join([str(d) for d in dates])
    else:
        args.dates = str(dates)
    args.overwrite = overwrite
    args.verbose = verbose
    args.local_dir = local_dir
    args.print_files = print_files
    args.crop_to_region = crop_to_region

    return icepyx_download_parent_process(args)

def build_child_process_command(args, print_files=False):
    proc_str = "{0} {1} --child".format(sys.executable, __file__)
    if args.dataset_name:
        proc_str += " -dataset_name " + args.dataset_name.replace(" ", "\ ")
    if args.region:
        proc_str += " -region " + args.region
    if args.local_dir:
        proc_str += " -local_dir " + args.local_dir.replace(" ", "\ ") # Make sure to handle spaces in the directory paths.
    if args.dates:
        proc_str += " -dates " + args.dates
    if args.overwrite:
        proc_str += " --overwrite"
    if args.verbose:
        proc_str += " --verbose"
    if args.print_files or print_files:
        proc_str += " --print_files"
    if args.crop_to_region:
        proc_str += " --crop_to_region"
    if (not args.variables is None) and (len(args.variables) > 0):
        proc_str += ' ' + " ".join(args.variables)

    return proc_str


def icepyx_download_parent_process(args):
    """Function to spawn and manage an icepyx/earthdata session in a subprocess.

    We use the pexpect subprocess in order to seamlessly login with our credentials,
    circumventing the manual login process if we already have it.
    """
    # Spawn a child process of the same script with the --child flag enabled.
    # Use pexpect in order to interact with it.
    # We will wait for a flag saying "Earthdata Login password: ",
    # and then give it the password from our locally-encrypted account credentials.
    proc_str = build_child_process_command(args, print_files=True)
    # print("PROC_STR:", proc_str)
    child = pexpect.spawn(proc_str)

    # Keep looping through every 0.01 seconds, print anything the child process
    # is printing, and return when the child process output ends.
    filenames = []
    logged_in = False
    while True:
        # If we are prompted for an Earthdata Login password,
        # get it from the credentials and keep going.
        if not logged_in:
            try:
                child.expect("Earthdata Login password: ", timeout=0.01)
                before_txt = child.before
                if args.verbose:
                    print(before_txt.decode("utf-8") )

                # Get the credentials saved in the encrypted NSIDC credential file, accessed through config.ini
                uname, pwd = my_config._read_credentials()
                child.sendline(pwd)
                del pwd

                if args.verbose:
                    print("Logging (again) into NASA EarthData account '{}'...".format(uname))

                del uname

                logged_in = True

            except pexpect.TIMEOUT:
                pass
            except pexpect.EOF:
                child.expect(pexpect.EOF, timeout=0.0)
                if args.verbose:
                    print(child.before.decode("utf-8"))
                return filenames

        # If the child process prints out the arguments,
        try:
            child.expect("==GRANULES_DOWNLOADED==", timeout=0.01)
            if args.verbose:
                print(child.before.decode("utf-8"))
            # Wait for the ending prompt.
            child.expect("==END_GRANULES_DOWNLOADED==")
            text_before = child.before.decode("utf-8")
            # Clip off the starting prompt if it's still in there (it shouldn't be.)
            if text_before.find("==GRANULES_DOWNLOADED==") >= 0:
                text_before = text_before[text_before.find("==GRANULES_DOWNLOADED=="):]
            if text_before.find("==END_GRANULES_DOWNLOADED==") >= 0:
                text_before = text_before[:text_before.find("==END_GRANULES_DOWNLOADED==")]

            filenames = [s.strip() for s in text_before.splitlines() if len(s.strip()) > 0]
            if args.print_files:
                print("\n".join(sorted(filenames)))

        except pexpect.TIMEOUT:
            pass
        except pexpect.EOF:
            child.expect(pexpect.EOF, timeout=0.0)
            if args.verbose:
                print(child.before.decode("utf-8"))
            return filenames

        # Just repeat each line printed by the child process.
        try:
            child.expect("\r\n", timeout=0.01)
            if args.verbose:
                print(child.before.decode("utf-8"))
            continue
        except pexpect.TIMEOUT:
            pass
        except pexpect.EOF:
            child.expect(pexpect.EOF, timeout=0.0)
            if args.verbose:
                print(child.before.decode("utf-8"))
            return filenames

    return filenames


def icepyx_download_child_process(args):
    """Child process handling an icepyx session.

    Parameters:
    -----------
        - dataset_name: "ATL03", "ATL06", or "ATL08". Defaults to "ATL03".

        - region: can be 1 of 3 types:
            - bounding box in WGS84 geographic coordinates: [xmin, ymin, xmax, ymax]
            - polygon in geographic coordinates: [(x1, y1), (x2, y2), (x3, y3), (x1, y1)]
            - path to a local shapefile: "/shape/file/location/aoi.shp"

        - date_range: 2-length list of dates to search: ['2020-01-01','2021-06-19'].
            - defaults to ['2018-01-01', today's date on your computer]

        - overwrite_if_existing: If True, if all the files already exist locally, don't overwrite, just return.
            (Currently there is not a way to easily subset the granules, when we figure this out
             in icepyx I will get a subset of the files only)

        - list_of_variables: A list of variable keys by which to subset the data. Separated by commas.

        - local_dir: Download them to this directory. If None specified, use the data
            directory outlined for that dataset in config.ini.
    """
    ## date_range ##
    # If date_range is None, default
    if args.dates is None:
        date_range = [my_config.atlas_sdp_epoch[:my_config.atlas_sdp_epoch.find("T")],
                      datetime.date.today().strftime("%Y-%m-%d")]
    elif type(args.dates) == str:
        try:
            date_range = ast.literal_eval(args.dates)
        except:
            date_range = args.dates.strip().strip("[](){}").replace("'","").replace('"',"").split(',')
    else:
        date_range = args.dates

    # Region:
    if args.region is None:
        region = [-180, -90, 180, 90]
    elif os.path.exists(args.region):
        # If it points to a shapefile, just leave it as is.
        region = args.region
    else:
        region = ast.literal_eval(args.region)

    # print("date_range", date_range)
    # print("region", region)

    # if len(args.variables) > 0:
    #     # Parse out by commas
    #     list_of_variables = atl_variable_finder.accrue_variables(args.variables)

    # Create the initial icepyx query object.
    query = icepyx.Query(args.dataset_name.upper(), region, date_range)

    # If we already have all the granules downloaded, just skip it all.
    granules_list = query.avail_granules(ids=True)[0]
    if (not args.overwrite) and \
        are_all_files_already_downloaded(granules_list, local_dir=args.local_dir):
            if args.verbose:
                print("All '{}' granules already present.".format(args.dataset_name))
            return

    # Log into NASA EarthData.
    uname, pwd = my_config._read_credentials()
    if (uname, pwd) == (None, None):
        # TODO: handle entering NSIDC credentials the first time. Then prompt to save it.
        raise FileNotFoundError("NSIDC credentials file '{}' not found.", my_config.nsidc_cred_file)

    else:
        # Password will be sent by the parent process. Just leave it for now.
        del pwd
        email = my_config.nsidc_account_email
        query.earthdata_login(uname, email)

    # Subset the data
    #   ... by variables
    if len(args.variables) > 0:
        query.order_vars.append(var_list=args.variables) # Whatever keywords/variable we want to get here.
        query.subsetparams(Coverage=query.order_vars.wanted)

    # Order the granules from NSIDC (set email=False to not get a fucking email
    # every time we run this.)
    query.order_granules(email=False, subset=args.crop_to_region)

    # Download the data.
    query.download_granules(args.local_dir, subset=args.crop_to_region)

    granules_downloaded = find_which_granules_are_present_on_local_system(granules_list, args.local_dir)
    if args.print_files:
        # Print these tags so that the parent process can easily parse the list of granules downloaded.
        print("==GRANULES_DOWNLOADED==")
        print("\n".join(granules_downloaded))
        print("==END_GRANULES_DOWNLOADED==")

    return granules_downloaded

def find_which_granules_are_present_on_local_system(granule_list, local_dir=None):
    """Given a list of granules to be downloaded, return a list of which files exist in the local directory.

    Often, Icepyx will list the granules to download, but in the spatial subsetting,
    not all of them will have data within the bounding box. Unfortunately it doesn't
    provide a list of which granules were downloaded and which weren't. Also,
    subsetted granules will have text added (such as "processed_") to the file name.

    This searches for the granules that were to be downloaded, and returns a list of the
    ones that actually were, and their actual filenames on the system.
    """
    if len(granule_list) == 0:
        return []
    # Flatten the list if it isn't already.
    elif len(granule_list) > 0 and type(granule_list[0]) in (list,tuple):
        granule_list = list(numpy.concatenate(granule_list).flat)

    granule_ids = [os.path.splitext(os.path.split(gr)[1])[0].upper() for gr in granule_list]
    if local_dir is None:
        local_dir = os.getcwd()

    local_dir_files = os.listdir(local_dir)
    local_dir_files_upper = [fname.upper() for fname in local_dir_files]

    fnames_found = []
    for gid in granule_ids:
         for dirfile, dirfile_upper in zip(local_dir_files,local_dir_files_upper):
             if dirfile_upper.find(gid) >= 0:
                 fnames_found.append(dirfile)


    return sorted(fnames_found)

def are_all_files_already_downloaded(granule_list, local_dir=None):
    """Check to see if all files for this icepyx request are already downloaded (or not).

    Looks in local_dir. If local_dir is None, checks in the dataset directory for that dataset in config.ini.
    NOTE: Assumes that all files are of the same dataset type, i.e. ATL03, ATL06, ATL08. Does not look in
    different folders.
    """
    # flatten the list if it isn't.
    if len(granule_list) == 0:
        return True
    elif len(granule_list) > 0 and type(granule_list[0]) in (list,tuple):
        granule_list = list(numpy.concatenate(granule_list).flat)

    granule_ids = [os.path.splitext(os.path.split(gr)[1])[0].upper() for gr in granule_list]
    if local_dir is None:
        # Get the dataset name.
        if granule_ids[0].find("ATL03") >= 0:
            local_dir = my_config.atl03_dir_raw
        elif granule_ids[0].find("ATL06") >= 0:
            local_dir = my_config.atl06_dir_raw
        elif granule_ids[0].find("ATL08") >= 0:
            local_dir = my_config.atl08_dir_raw
        else:
            raise ValueError("Granule name '{}' does not contain an 'ATL03', 'ATL06', or 'ATL08' identifier.")

    local_dir_files = [fname.upper() for fname in os.listdir(local_dir)]

    # Loop through all the granule_ids we're looking for
    for gid in granule_ids:

        found_gid = False
        # Loop through all the filenames in the directory, to find that ID.
        for dirfile in local_dir_files:
            if dirfile.find(gid) >= 0:
                found_gid = True
                break

        # If we don't find this granule_id anywhere in the available files, return False
        if not found_gid:
            return False

    # If we got to this point, we found each granule_id in a file name in the directory.
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="A utility for requesting and downloading ICESat-2 using icepyx.")

    parser.add_argument('variables', type=str, nargs="*",
                        help="List of variable keys by which to subset the data. Default to all the data (not subset).")
    parser.add_argument('-dataset_name', '-d', metavar="ATLXX", type=str, default="ATL03",
                        help="Dataset short name. Currently accepts ATL03, ATL06, or ATL08. Default ATL03.")
    parser.add_argument('-region', '-r', type=str, default='[-180,-90,180,90]',
                        help="Search area. Can be [xmin,ymin,xmax,ymax], a list of (x,y) points in a polygon, or the local path to a shapefile.")
    parser.add_argument('-local_dir', default=None,
                        help="Local directory to download the data.")
    parser.add_argument('-dates', default=None, type=str,
                        help="A pair of dates (separated by a comma) in which to search: YYYY-MM-DD,YYYY-MM-DD")
    parser.add_argument('--overwrite', '-o', default=False, action="store_true",
                        help="Overwrite existing granules even if they area all here. Default: skip if they are all present.")
    parser.add_argument('--child', default=False, action="store_true",
                        help="Make this the child icepyx process. Will prompt for password. (Not typically used by end-user.)")
    parser.add_argument('--verbose', default=False, action='store_true',
                        help="Verbose output.")
    parser.add_argument('--print_files', default=False, action='store_true',
                        help="Print a list of files downloaded to the local_dir.")
    parser.add_argument('--crop_to_region', default=False, action='store_true',
                        help="Crop the data to the given bounding box. Default: do not (will return all granules that overlap the bounding box, including data that doesn't).")
    return parser.parse_args()


def args_from_editor_EVEREST():
    # I can edit & use this function at will when I don't feel like doing it from command-line args.
    args = argparse.Namespace
    args.dataset_name = "ATL08"
    args.region = "[86,27,87,28]"
    args.local_dir = "../data/temp/everest_full"
    args.dates="2020-01-01,2020-12-31"
    args.verbose = True
    args.print_files = True
    args.variables = []
    args.crop_to_region = False
    args.child = False
    args.overwrite=False

    return args

def args_from_editor_NE_DEMS():
    # I can edit & use this function at will when I don't feel like doing it from command-line args.
    args = argparse.Namespace
    args.dataset_name = "ATL03"
    args.region = "[-71.25,41,-66.75,45.25]"
    args.local_dir = "../../DEMs/NCEI/ma_nh_me_deliverables20211007105912/ma_nh_me_deliverables/icesat2"
    args.dates="2020-04-01,2020-10-01"
    args.verbose = True
    args.print_files = True
    args.variables = []
    args.crop_to_region = False
    args.child = False
    args.overwrite=False

    return args

def args_from_editor():
    # I can edit & use this function at will when I don't feel like doing it from command-line args.
    args = argparse.Namespace
    args.dataset_name = "ATL08"
    args.region = "[19,0,20,1]"
    args.local_dir = "../../scratch_data/sample_tiles_list/Congo/icesat2"
    args.dates="2021-01-01,2022-01-01"
    args.verbose = True
    args.print_files = True
    args.variables = []
    args.crop_to_region = False
    args.child = False
    args.overwrite=False

    return args

if __name__ == "__main__":
    # Use the command-line arguments if running from a system shell (i.e. command-line or shell script).
    if progress_bar.is_run_from_command_line():
        args = parse_args()
    # Else, just use the arguments I've hard-coded in here.
    else:
        args = args_from_editor()

    # sys.exit(0)

    if args.child:
        files = icepyx_download_child_process(args)
    else:
        files = icepyx_download_parent_process(args)
        # print("\n".join(sorted(files)))
