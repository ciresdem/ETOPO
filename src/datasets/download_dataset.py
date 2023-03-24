# -*- coding: utf-8 -*-

import os
import re
import pexpect
import subprocess
import urllib

###############################################################################
# Import the project /src directory into PYTHONPATH, in order to import all the
# other modules appropriately.
import import_parent_dir; import_parent_dir.import_src_dir_via_pythonpath()
###############################################################################
import utils.traverse_directory

class DatasetDownloader_BaseClass:
    """A base class for downloading datasets from various websites, usually presented
    as a list of links on a web page (sometimes at an FTP site, sometimes with a whole host of sub-directories)
    This creates some basic tools for downloading datasets in parallel with an arbitrary number of subprocesses.
    """

    def __init__(self, website_url, local_data_dir):
        self.website_base_url = website_url
        self.local_data_dir = local_data_dir
        # It's handy to start this file with an underscore, that way we
        self.url_list = os.path.join(self.local_data_dir, "_URL_LIST.txt")

    def clean_up_extraneous_wget_files(self, data_dir=None, recurse=False):
        """The wget command, if you download a file that was already there, tends to re-download with a .1, .2 extension.
        Get rid of all those files, plus any 'wget-log' files."""
        if not data_dir:
            data_dir = self.local_data_dir

        filelist = os.listdir(data_dir)

        # Just a quick removal message. Better than repeating thie print
        # statement multiple times below (if I want to change it)
        def remove_msg(fname, numbytes):
            print(fname, "removed ({0} bytes).".format(numbytes))

        for fname in filelist:
            fpath = os.path.join(data_dir, fname)
            # If it's a directory, and if we've chosen to recurse, than recurse into it.
            if os.path.isdir(fpath):
                if recurse:
                    self.clean_up_extraneous_wget_files(data_dir=fpath, recurse=recurse)
                continue
            # Otherwise it's a file. Go from there.
            fsize = os.path.getsize(fpath)

            # Remove the various "wget-log" messages.
            if fname.find("wget-log") == 0:
                os.remove(fpath)
                remove_msg(fname, fsize)
                continue

            # If wget did a duplicate download, it gives that a ".1", ".2" extension. Get rid of these.
            result = re.search("\.(\d+)\Z", fname)
            if result != None:
                os.remove(fpath)
                remove_msg(fname, fsize)

    def create_list_of_links(self,
                             recurse=True,
                             file_regex=None,
                             subdir_regex=None,
                             write_to_file=True):
        """This is a base-class virtual function. Since each directory structure, protocol, etc,
        of each data source is fairly unique, it's impractical to come up with one single
        funciton to traverse remote directories for everything. Create a subclass and inheret
        this class to use the other helper functions, but create an overriding "create_list_of_links()"
        class method in the subclass."""
        # This function, as written, should traverse the subdirectories of whichever website
        # we are trying to download from, and create a textfile list of file
        # URLs to download, one file per line. The "download_file()" method
        # will traverse that list and download the data using 'wget' commands.
        raise NotImplementedError("Base class DatasetDownloader_BaseClass does not implement " + \
                                  "an instance of create_list_of_links() method. This must be " + \
                                  "generated and used within the sub-class.")

        return None

    def query_for_remote_filesize(self, url, data_dir):
        """Use the 'wget --spider [url]' command to query for a remote file size."""

        sizecheck_command = "wget --spider " + url
        # print(command)
        (command_output, exitstatus) = pexpect.run(sizecheck_command, cwd=data_dir, withexitstatus=True)

        if exitstatus == 0:
            # A typical query string looks like this:
            #    Length: 151092250 (144M) [image/tiff]
            # Here we're looking for the two words between 'Length: ' and the first "[". Those tell us the
            # exact bite size (fsize_remove) and the short string for the file size (fsize_small)
            try:
                fsize_remote, fsize_small = re.search("(?<=Length: )[\w\(\)\. ]+(?=\[)", command_output.decode('utf-8')).group().split()
            except AttributeError:
                # If the regex search fails, at least print the output so I can debug it.
                print("WARNING: String regex-match failed in query_for_remote_filesize()...")
                print(sizecheck_command)
                print(command_output.decode('utf-8'))
                return exitstatus, None, None
            fsize_small = fsize_small.strip("(),")
            fsize_remote = int(fsize_remote)


        return exitstatus, fsize_remote, fsize_small

    # NOTE: Before running this function, you should go into the sub-directory for the individual dataset and
    # use the create_list_of_links() function in there.
    def download_files(self,
                       N_subprocs=1,
                       include_speed_strings=False,
                       check_sizes_of_existing_files=True,
                       wget_extra_args=None):
        """wget_extra_args should be a single string with extra arguments to
        feed to the wget command. Nothing will be added if None.
        """

        file_of_urls = self.url_list
        data_dir = self.local_data_dir
        print(file_of_urls)
        if not os.path.exists(file_of_urls):
            self.create_list_of_links()

        urls_ALL = [line.strip() for line in open(file_of_urls,'r').readlines() if len(line.strip()) > 0]
        N = len(urls_ALL)

        urls_to_download = []
        wget_commands = []
        active_procs_list = []
        active_urls_list = []
        active_fsize_strings = []

        longest_update_strlen = 0

        # return
        # terminal_width = get_terminal_width()

        # First, get a list of all the commands to execute.
        print("Browsing existing files to look for needed downloads... ", end="")
        for i,url in enumerate( urls_ALL ):

            command = "wget " + ("" if (wget_extra_args is None) else (wget_extra_args + " ")) + url

            num_dirs_to_cut = 0
            if wget_extra_args is not None and wget_extra_args.find("--cut-dirs=") >= 0:
                num_dirs_to_cut = int(re.search("(?<=--cut-dirs\=)\d+", wget_extra_args).group())

            url_dirs = urllib.parse.urlparse(url).path.strip("/").split("/")
            subdirs = url_dirs[num_dirs_to_cut:]

            local_filepath = os.path.join(data_dir, *subdirs)

            # # THIS IS WRONG, THIS IS NOT HOW THE FILES ARE SAVED. LOOK IN SUBDIRS.
            # filename = os.path.split(url)[1]
            # local_filepath = os.path.join(data_dir, filename)

            # First check to see if the file already exists (did we already download it?)
            if os.path.exists(local_filepath):
                # print(True, i)
                # If we're checking the remote vs local file sizes to see if it already downloaded, try again.
                if check_sizes_of_existing_files:
                    # Run the 'wget --spider' command to query the remove files size, get its output.
                    # sizecheck_command = "wget --spider " + url
                    # # print(command)
                    # (command_output, exitstatus) = pexpect.run(sizecheck_command, cwd=data_dir, withexitstatus=True)
                    # print(command_output)
                    # print(exitstatus)
                    # print(command_output) #.decode('utf-8'))
                    # If the wget --spider command exited successfully.
                    # if exitstatus == 0:
                    #     # A typical query string looks like this:
                    #     #    Length: 151092250 (144M) [image/tiff]
                    #     # Here we're looking for the two words between 'Length: ' and the first "[". Those tell us the
                    #     # exact bite size (fsize_remove) and the short string for the file size (fsize_small)
                    #     fsize_remote, fsize_small = re.search("(?<=Length:\ )[\w\(\)\ ]+(?=\[)", command_output.decode('utf-8')).group().split()
                    #     fsize_small = fsize_small.strip("(),")
                    #     fsize_remote = int(fsize_remote)
                    exitstatus, fsize_remote, fsize_small = self.query_for_remote_filesize(url, data_dir)
                    if exitstatus == 0:
                        fsize_local = os.path.getsize(local_filepath)
                        # print(url, fsize_remote, fsize_local)
                        if fsize_remote == fsize_local:
                            # File already exists fully locally. Skip & move on.
                            # print("({0} of {1})".format(i+1, N), filename, "already exists.", fsize_small)
                            continue
                        else:
                            # If part of the file already exists, we can remove it by using the "--continue" flag.
                            command = command + " --continue"
                            urls_to_download.append(url)
                            active_fsize_strings.append(fsize_small)
                            wget_commands.append(command)

                    else:
                        # print("({0} of {1})".format(i+1, N), filename, "size query returned exit status {0}. Skipping.".format(exitstatus))
                        continue
                        urls_to_download.append(url)
                        active_fsize_strings.append(None)
                        wget_commands.append(command)
                else:
                    # print("({0} of {1})".format(i+1, N), filename, "already exists.")
                    # File already exists and we're not checking any new ones, just go on to the next.
                    continue
            else:
                # exitstatus, fsize_remote, fsize_small = self.query_for_remote_filesize(url, data_dir)
                # In some cases for CUDEM tiles, we moved them one or two folders up after download. Check there.
                file_found = False
                for numdirs_up in (1,2):
                    dirname, fname = os.path.split(local_filepath)
                    upper_filepath = os.path.abspath(os.path.join(dirname, "/".join([".."] * numdirs_up), fname))
                    if os.path.exists(upper_filepath):
                        file_found = True
                        break
                if file_found:
                    continue
                urls_to_download.append(url)
                active_fsize_strings.append(None)
                wget_commands.append(command)

        print("Done.")

        # print("\n".join(urls_to_download[0:10]))
        # print("\n".join(wget_commands[0:10]))
        # return
        assert len(wget_commands) == len(urls_to_download)

        if len(urls_to_download) < N:
            print("{0} of {1} complete. {2} remaining to download.".format(N - len(urls_to_download), N, len(urls_to_download)))
            N = len(urls_to_download)

        # Now, start looping through and adding processes to the queue until everything is executed.
        status_strings = []
        num_finished = 0
        try:
            while (len(wget_commands) > 0) or (len(active_procs_list) > 0):

                # First, loop through the active processes and see if any can be removed.
                # Build a string of the remaining ones' status.
                # This loop will be skipped the first time if we haven't added any processes yet.
                for i,(current_proc,
                       fsize_str,
                       current_url,
                       status_str) \
                    in \
                    enumerate(zip(active_procs_list,
                                  active_fsize_strings,
                                  active_urls_list,
                                  status_strings)):
                    # The current process is alive. Get its status.
                    if fsize_str in (None,""):
                        _, _, fsize_str = self.query_for_remote_filesize(current_url, data_dir)
                        active_fsize_strings[i] = fsize_str

                    if current_proc.isalive():
                        retval = current_proc.expect(["\r", pexpect.EOF, pexpect.TIMEOUT], timeout=0.001)
                        if retval == 0:
                            # Get the 'before' string, retrieve the status code.
                            before = current_proc.before.decode('utf-8')
                            try:
                                # Update the status string for this process.
                                percent_str = re.search(r"(\d+)%", before).group()
                                #
                                if include_speed_strings:
                                    speed_str = re.search(r"(\d+)[KMG ]B\/s", before).group()
                                    status_strings[i] = "{0} {1} {2}".format(fsize_str,
                                                                             speed_str,
                                                                             percent_str)
                                else:
                                    status_strings[i] = "{0} {1}".format(fsize_str,
                                                                         percent_str)

                            except:
                                pass
                        else:
                            assert retval in (1,2)
                            # These will be removed the next time this loop is entered.
                            pass
                    else:
                        # Remove it from the queue, print a confirmation message.
                        # Find the process in active_procs_list
                        proc_idx = active_procs_list.index(current_proc)
                        # active_procs_list.remove(current_proc)
                        active_procs_list = active_procs_list[:proc_idx] + active_procs_list[proc_idx+1:]
                        # active_fsize_strings.remove(fsize_str)
                        active_fsize_strings = active_fsize_strings[:proc_idx] + active_fsize_strings[proc_idx+1:]
                        # status_strings.remove(status_str)
                        status_strings = status_strings[:proc_idx] + status_strings[proc_idx+1:]
                        # active_urls_list.remove(current_url)
                        active_urls_list = active_urls_list[:proc_idx] + active_urls_list[proc_idx+1:]
                        fname = os.path.split(current_url)[1]
                        num_finished += 1
                        str_to_print = "\r({0} of {1}) {2} complete. ({3})".format(
                                    num_finished,
                                    N,
                                    fname,
                                    fsize_str)
                        print(str_to_print + " " * (max(0,longest_update_strlen-len(str_to_print))))

                # Print the status strings to the screen.
                status_update_str = "\r" + " | ".join(status_strings)
                print(status_update_str + " "*(max(0,longest_update_strlen-len(status_update_str))), end="")
                longest_update_strlen = max(longest_update_strlen, len(status_update_str))

                # Second, check if any new processes can be started.
                # and whether there is room for them on the queue.
                # If so, kick them off and add them to the list.
                while (len(active_procs_list) < N_subprocs) and (len(wget_commands) > 0):
                    # Kick off another process, add it to the queue
                    # Get the file size from it.
                    new_wget_command = wget_commands.pop(0)
                    new_url = urls_to_download.pop(0)

                    proc = pexpect.spawn(new_wget_command, cwd=data_dir)

                    try:
                        proc.expect("B/s")
                    except:
                        # What to do here?
                        pass

                    # At times there can be weird mismatch in file sizes.
                    # If it says it's "already fully retrieved", just skip this step and move onto the next.
                    before = proc.before.decode("utf-8")
                    if before.find("The file is already fully retrieved") > 0:
                        num_finished += 1
                        str_to_print = "\r({0} of {1}) {2} already fully retrieved.".format(num_finished, N, os.path.split(new_url)[1])
                        print(str_to_print + " "*(max(longest_update_strlen - len(str_to_print), 0)))
                        continue

                    # Append this process to the list of active processes and strings
                    active_procs_list.append(proc)
                    active_urls_list.append(new_url)
                    status_strings.append("")
                    # Now, retrive the file size from the wget opening screen. Also, add to the various status strings.
                    try:
                        size_strings = re.search("(?<=Length: )[\w\(\)\.\, ]+(?=\[)", before).group().split()
                        active_fsize_strings.append(size_strings[1].strip("(),"))
                    except:
                        active_fsize_strings.append("")

        finally:
            # Clean up some of the extra crap that wget leaves behind.
            print()
            self.clean_up_extraneous_wget_files()
            return

    def unzip_downloaded_files(self,
                               to_subdirs=True,
                               recurse_directory=True,
                               fname_regex_filter = "\.zip\Z",
                               overwrite=True,
                               verbose=True):
        """Unzip the .zip or .tar.gz files that have just been downloaded.

        If 'to_subdirs' is selected, unpack them to directories named after the files. If not, just put the files in
        the directory already.
        """
        data_dir = self.local_data_dir
        if recurse_directory:
            fnames = utils.traverse_directory.list_files(self.local_data_dir, regex_match=fname_regex_filter)
        else:
            fnames = sorted(os.listdir(data_dir))
        if fname_regex_filter:
            fnames = [fn for fn in fnames if re.search(fname_regex_filter, fn) != None]

        for fname in fnames:
            base, ext = os.path.splitext(fname)
            ext = ext.lower()
            if ext == ".zip":
                command = "unzip"
            elif ext == ".gz":
                base, ext2 = os.path.splitext(base)
                assert ext2.lower() == ".tar"
                command = "tar -xf"

            command = command + " " + os.path.join(data_dir, fname)

            subdir = os.path.join(data_dir, base)
            if to_subdirs:
                if os.path.exists(subdir):
                    if not os.path.isdir(subdir):
                        print("Cannot unzip", fname, "to", base + ". A file exists with that name.")
                        continue
                else:
                    os.mkdir(subdir)

                command = command + " --directory " + subdir

            # if overwrite:
            #     command = command + " -o"
            # else:
            #     command = command + " -n"

            if verbose:
                print(command, end="")

            proc = subprocess.run(command.split(), cwd=data_dir)

            if verbose:
                if proc.returncode == 0:
                    print( " ... success." )
                else:
                    print( " ... ERROR: Return code", proc.returncode)

if __name__ == "__main__":
    pass
    # dirname = os.path.abspath(os.path.join(os.path.split(__file__)[0], "../../../DEMs/TanDEM-X/data/90mdem/DEM"))
    # print(dirname)
    # dl = DatasetDownloader_BaseClass("TanDEMX", dirname)
    # dl.unzip_downloaded_files(to_subdirs=False, overwrite=False)
