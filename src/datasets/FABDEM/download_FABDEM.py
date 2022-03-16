# -*- coding: utf-8 -*-

import re
import os
import urllib.request
import pexpect
import argparse
# import time
# import fcntl, termios, struct, sys, psutil

FABDEM_html_url = r'https://data.bris.ac.uk/data/dataset/25wfy0f9ukoge2gs7a5mqpq2j7'

FABDEM_file_template = r'https://data.bris.ac.uk/datasets/25wfy0f9ukoge2gs7a5mqpq2j7/[\w-]+?_FABDEM_V1-0.zip'.replace(".","\.")

FABDEM_local_data_dir = os.path.abspath(os.path.join(os.path.split(__file__)[0], "../../../../DEMs/FABDEM/data"))
URL_list_file = os.path.join(FABDEM_local_data_dir, "_URL_LIST.txt")
# print(os.path.exists(FABDEM_local_data_dir))


def create_list_of_links(file_to_save = URL_list_file,
                         url=FABDEM_html_url,
                         regex_template=FABDEM_file_template):
    f = urllib.request.urlopen(url)
    website_html = str(f.read())
    f.close()

    # print(website_html)[0:500]
    list_of_links = re.findall(regex_template, website_html)
    if len(list_of_links) == 0:
        print("No matching links found. Exiting.")
        return

    # Make sure we just save unique ones, no duplicates.
    # set() objects only save unique values. So it's a good way to do this. list --> set --> list, sorted
    list_of_links = sorted(list(set(list_of_links)))

    with open(file_to_save, 'w') as fout:
        fout.write("\n".join(list_of_links))
        fout.close()
        print(len(list_of_links), "file URLs written to", os.path.abspath(file_to_save))

def clean_up_extraneous_wget_files(data_dir=FABDEM_local_data_dir):
    """The wget command, if you download a file that was already there, tends to re-download with a .1, .2 extension.
    Get rid of all those files, plus any 'wget-log' files."""
    filelist = os.listdir(data_dir)
    for fname in filelist:
        fpath = os.path.join(data_dir, fname)
        fsize = os.path.getsize(fpath)

        def remove_msg(fname, numbytes):
            print(fname, "removed ({0} bytes).".format(numbytes))

        if fname.find("wget-log") == 0:
            os.remove(fpath)
            remove_msg(fname, fsize)
            continue

        result = re.search("\.(\d+)\Z", fname)
        if result != None:
            os.remove(fpath)
            remove_msg(fname, fsize)

def download_files(file_of_urls=URL_list_file,
                   data_dir = FABDEM_local_data_dir,
                   N_subprocs=11,
                   check_sizes_of_existing_files=True):
    if not os.path.exists(file_of_urls):
        create_list_of_links(file_to_save=file_of_urls)

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

        command = "wget " + url
        filename = os.path.split(url)[1]
        local_filepath = os.path.join(data_dir, filename)

        # First check to see if the file already exists (did we already download it?)
        if os.path.exists(local_filepath):
            # print(True, i)
            # If we're checking the remote vs local file sizes to see if it already downloaded, try again.
            if check_sizes_of_existing_files:
                # Run the command, get its output.
                (command_output, exitstatus) = pexpect.run("wget --spider " + url, cwd=data_dir, withexitstatus=True)
                # print(command_output) #.decode('utf-8'))
                # If the wget --spider command exited successfully.
                if exitstatus == 0:
                    fsize_remote, fsize_small = re.search("(?<=Length: )[\w\(\)\.\, ]+(?= \[application\/zip\])", command_output.decode('utf-8')).group().split()
                    fsize_remote = int(fsize_remote)
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
                        # fsize_strings.append(fsize_small)
                        wget_commands.append(command)

                else:
                    # print("({0} of {1})".format(i+1, N), filename, "size query returned exit status {0}. Skipping.".format(exitstatus))
                    continue
                    urls_to_download.append(url)
                    # fsize_strings.append(None)
                    wget_commands.append(command)
            else:
                # print("({0} of {1})".format(i+1, N), filename, "already exists.")
                continue
        else:
            urls_to_download.append(url)
            # fsize_strings.append(None)
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
                if current_proc.isalive():
                    retval = current_proc.expect(["\r", pexpect.EOF, pexpect.TIMEOUT], timeout=0.001)
                    if retval == 0:
                        # Get the 'before' string, retrieve the status code.
                        before = current_proc.before.decode('utf-8')
                        try:
                            # Update the status string for this process.
                            percent_str = re.search(r"(\d+)%", before).group()
                            # speed_str = re.search(r"(\d+)[KMG ]B\/s", before).group()
                            status_strings[i] = "{0} {1}".format(fsize_str,
                                                                      # speed_str,
                                                                 percent_str)
                        except:
                            pass
                    else:
                        assert retval in (1,2)
                        # These will be removed the next time this loop is entered.
                        pass
                else:
                    # Remove it from the queue, print a confirmation message.
                    active_procs_list.remove(current_proc)
                    active_fsize_strings.remove(fsize_str)
                    status_strings.remove(status_str)
                    active_urls_list.remove(current_url)
                    fname = os.path.split(current_url)[1]
                    num_finished += 1
                    str_to_print = "\r({0} of {1}) {2} complete. ({3})".format(
                                num_finished,
                                N,
                                fname,
                                fsize_str)
                    print(str_to_print + " " * (max(0,longest_update_strlen-len(str_to_print))))

            status_update_str = "\r" + " | ".join(status_strings)
            print(status_update_str, end="")
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
                    size_strings = re.search("(?<=Length: )[\w\(\)\.\, ]+(?= \[application\/zip\])", before).group().split()
                    active_fsize_strings.append(size_strings[1].strip("(),"))
                except:
                    active_fsize_strings.append("")

                # return
    finally:
        # Clean up some of the extra crap that wget leaves behind.
        print()
        clean_up_extraneous_wget_files()
        return

    # # Run while there still exist acive processes, and/or commands still to run.
    # while (len(active_procs_list) > 0) or (len(urls_to_download) > 0):
    #     # First, go through all the active processes and see if any have finished.
    #     process_messages = []
    #     for active_proc, active_url in zip(active_procs_list, active_urls_list):
    #         indexval = active_proc.expect(["eta", pexpect.EOF, pexpect.TIMEOUT], timeout=0.01) #Each output line from wget ends with an "eta" statement.
    #         # Just pluck the last one if there's multiple lines of it.
    #         process_messages.append(active_proc.before.split("\n")[-1])

            # if index pexpect.TIMEOUT:
            #     process_messages.append(None)
            # except pexpect.EOF:
            #     # The process ended. Awesome.
            #     fname_base =  os.path.split(active_url)[1]
            #     local_fname = os.path.join(data_dir,)
            #     if os.path.exists(local_fname):
            #         print(os.path.split(local_fname))

            # For ones that have finished. Print "... downloaded" to the screen, with a newline
                # verify that the local file exists.
            # Check if they have died for reasons unknown (?)

        # Then if room in the active processes exists, kick off another one.
            # For each file, first check if it already exists locally. Skip it if it does.
        # Loop through and check the download status of each process.

def import_and_parse_args():
    parser = argparse.ArgumentParser(description="A utility for parallelized downloads of the U Bristol FABDEM data product.")
    parser.add_argument("-N", type=int, default=10, help="Number of parallel processes.")
    parser.add_argument("-outdir", type=str, default=None, help="Directory to output the files. (Defaults to {0})".format(FABDEM_local_data_dir))
    return parser.parse_args()

if __name__ == "__main__":
    # print(FABDEM_file_template)
    # result = re.search("https://data.bris.ac.uk/datasets/25wfy0f9ukoge2gs7a5mqpq2j7/[\w-]+?_FABDEM_V1-0.zip".replace(".","\."), "https://data.bris.ac.uk/datasets/25wfy0f9ukoge2gs7a5mqpq2j7/N00E000-N10E010_FABDEM_V1-0.zip")
    # print(result)
    # create_list_of_links()
    args = import_and_parse_args()

    if args.outdir == None:
        download_files(N_subprocs=args.N)
    else:
        URL_list_file = os.path.join(args.outdir, "_URL_LIST.txt")
        download_files(file_of_urls = URL_list_file,
                       data_dir=args.outdir,
                       N_subprocs=args.N)
