# -*- coding: utf-8 -*-

"""free_unused_ram.py -- A little utility for calling command-line processes to free
up RAM. My ubuntu machine, when I'm processing large datasets, has an annoying habit
of caching file data and hanging onto it. 1-2 hours into a process, my RAM is full
and the machine slows to a crawl. This calls a bash command to free kup that memory
to keep things running okay. NOTE: Don't overuse, it does temporarily slow down access
to files being used that were previously cached, because they need to be re-read
into cache. But it's helpful to keep memory-usage to a reasonable level."""

# import subprocess
import pexpect
import time
import getpass
import psutil
import argparse
import datetime

###############################################
import import_parent_dir
import_parent_dir.import_parent_dir_via_pythonpath()
###############################################
import utils.configfile
my_config = utils.configfile.config()

def free_unused_ram_once(verbose=True):
    # This command frees up unused memory (primarly disk cache) that Ubuntu is holding onto.
    # I'm finding it expecially helpful for my code where processing large
    sync_args = 'sudo sync'
    drop_cache_args ='sudo sysctl -w vm.drop_caches=3'

    if verbose:
        print(sync_args)

    childproc = pexpect.spawn(sync_args, encoding='utf-8')

    i = childproc.expect(["password[\w\s]+:", pexpect.EOF], timeout=None)
    if i == 0:
        uname, pwd = my_config._read_credentials(fname_attribute="user_cred_file", verbose=verbose)

        if verbose and (uname, pwd) == (None, None):
            print(childproc.before if childproc.before != None else "", end="")
            print(childproc.match.group(), end="")
            print(childproc.after if childproc.after != None else "", end="")
            # Get a password prompt
            uname = getpass.getuser()
            pwd = getpass.getpass()
            # Save the credentials for later use.
            my_config._save_credentials(uname, pwd, fname_attribute="user_cred_file", warn_if_exists=verbose, verbose=verbose)

        # Send the password.
        childproc.sendline(s=pwd)
        del pwd
        del uname

        # After that, and a bit of outout text, we should soon see the EOF
        childproc.expect(pexpect.EOF, timeout=None)
        if verbose:
            print(childproc.before, end="")
        childproc.close()
    elif i == 1:
        # Reach EOF
        if verbose:
            print(childproc.before, end="")
        childproc.close()

    if verbose:
        print(drop_cache_args)
    childproc = pexpect.spawn(drop_cache_args, encoding='utf-8')

    i = childproc.expect(["password[\w\s]+:", pexpect.EOF], timeout=None)
    if i == 0:
        uname, pwd = my_config._read_credentials(fname_attribute="user_cred_file", verbose=verbose)

        if verbose and (uname, pwd) == (None, None):
            print(childproc.before if childproc.before != None else "", end="")
            print(childproc.match.group(), end="")
            print(childproc.after if childproc.after != None else "", end="")
            # Get a password prompt
            uname = getpass.getuser()
            pwd = getpass.getpass()
            # Save the credentials for later use.
            my_config._save_credentials(uname, pwd, fname_attribute="user_cred_file", warn_if_exists=verbose, verbose=verbose)

        # Send the password.
        childproc.sendline(s=pwd)
        del pwd
        del uname

        # After that, and a bit of outout text, we should soon see the EOF
        childproc.expect(pexpect.EOF, timeout=None)
        if verbose:
            print(childproc.before, end="")
        childproc.close()
    elif i == 1:
        # Reach EOF
        if verbose:
            print(childproc.before, end="")
        childproc.close()

    return
    # The command to run is: sudo sync && sudo sysctl -w vm.drop_caches=3
    # However, to run sudo, I need to get the password. It's insecure but I
    # can use my confg.py code for this and save the credentials to an encrypted file.
    # The use pexpect to enter the passworkd when needed here.

def free_memory_every_N_seconds(S=600, only_if_over_pct=None, verbose=True):
    """Continuously loop, freeing up memory every N minutes.

    NOTE: This funcion does not return. It shiould be run as a subprocess and
    then kill()'ed when it is no longer needed.
    """
    while True:
        time.sleep(S)

        if only_if_over_pct is None:
            free_unused_ram_once(verbose=verbose)
        else:
            free_memory_if_over_percent_usage(P=only_if_over_pct, verbose=verbose)

def free_memory_if_over_percent_usage(P=78, verbose=True):
    """If current member usage is at or over P percent of total RAM, free some up."""
    ram_usage_pct = psutil.virtual_memory().percent

    if ram_usage_pct >= P:

        if verbose:
            print("RAM at {0:0.1f} %. Freeing on {1}.".format(
                ram_usage_pct,
                datetime.datetime.now().astimezone().strftime("%a %Y-%m-%d %H:%M:%S %Z")
                 ))

        free_unused_ram_once(verbose=False)

        if verbose:
            ram_usage_pct_new = psutil.virtual_memory().percent
            print("\tRAM now at {0:0.1f} %.".format(ram_usage_pct_new))

    return

def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Utility for monitoring memory and freeing unneeded cache if getting too full.")
    parser.add_argument("-every_seconds", '-s', type=int, default=-1, help="Monitor every S seconds.")
    parser.add_argument("-cutoff_pct", '-p', type=float, default=-1.0, help="Only if virtual memory usage over P percent.")
    parser.add_argument("--quiet", "-q", default=False, action="store_true", help="Run quietly.")
    return parser.parse_args()

if __name__ == "__main__":
    args = define_and_parse_args()
    try:
        if args.every_seconds != -1:
            free_memory_every_N_seconds(S=args.every_seconds,
                                        only_if_over_pct = None if (args.cutoff_pct == -1) else args.cutoff_pct,
                                        verbose=not args.quiet)

        elif args.cutoff_pct != -1:
            free_memory_if_over_percent_usage(P=args.cutoff_pct, verbose=not args.quiet)
        else:
            free_unused_ram_once(verbose=not args.quiet)
    except KeyboardInterrupt:
        import sys
        sys.exit(0)
