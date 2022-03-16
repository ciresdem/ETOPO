# -*- coding: utf-8 -*-

import re
import os
import urllib.request, urllib.parse
import argparse

# Add this line to give this file access to the whole code tree from the base directory.
import import_parent_dir; import_parent_dir.import_parent_dir_via_pythonpath()

import datasets.download_dataset

ArcticDEM_html_url = r'https://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic/v3.0/{subdir}/'
# These are subdirectories of the above URL

ArcticDEM_datasets = ["1km", "2m", "10m", "32m", "100m", "500m"]


ArcticDEM_local_data_dir = os.path.abspath(os.path.join(os.path.split(__file__)[0], "../../../../DEMs/ArcticDEM/data/{subdir}"))
URL_list_file = os.path.join(ArcticDEM_local_data_dir, "_URL_LIST.txt")

# Find links to sub-directories in the html file, and then explore those.
ArcticDEM_subdir_regex = "(?<=<a href=\")(?P<subdir>[\d_]+\/)(?=\"\>(?P=subdir)<\/a>)"
ArcticDEM_file_regex = "(?<=<a href=\")(?P<fname>[\w\.]+)(?=\"\>(?P=fname)<\/a>)"

class ArcticDEM_Downloader (datasets.download_dataset.DatasetDownloader_BaseClass):
    def __init__(self, dataset_id):
        # Initialized the parent class.
        html_link = ArcticDEM_html_url.replace("{subdir}", dataset_id)
        local_dir = ArcticDEM_local_data_dir.replace("{subdir}", dataset_id)
        # Initiate the instantiation of the base class. This saves the link and local_dir as well.
        super().__init__(html_link,local_dir)

    def create_list_of_links(self,
                             url=None,
                             recurse=True,
                             file_regex=ArcticDEM_file_regex,
                             subdir_regex=ArcticDEM_subdir_regex,
                             write_to_file=True):
        if url is None:
            url=self.website_base_url

        f = urllib.request.urlopen(url)
        website_html = str(f.read())
        f.close()

        if write_to_file:
            print("Parsing directory structure at", url, "to find ArcticDEM files to download. This may take a bit.")

        # print(website_html)[0:500]
        list_of_files = [urllib.parse.urljoin(url, fname) for fname in re.findall(file_regex, website_html)]

        # Find lists of links in sub-directories.
        if recurse:
            list_of_subdirs = re.findall(subdir_regex, website_html)
            for subdir in list_of_subdirs:
                subdir_path = urllib.parse.urljoin(url + ("" if (url[-1] == "/") else "/"), subdir)
                list_of_files.extend(self.create_list_of_links(url=subdir_path,
                                                               recurse=recurse,
                                                               file_regex=file_regex,
                                                               subdir_regex=subdir_regex,
                                                               write_to_file=False))

        # Make sure we just save unique ones, no duplicates.
        # set() objects only save unique values. So it's a good way to do this. list --> set --> list, sorted
        list_of_files = sorted(list(set(list_of_files)))

        if write_to_file:
            with open(self.url_list, 'w') as fout:
                fout.write("\n".join(list_of_files))
            print(self.url_list, "written with", len(list_of_files), "links.")

        return list_of_files


def import_and_parse_args():
    parser = argparse.ArgumentParser(description="A utility for parallelized downloads of the U Bristol FABDEM data product.")
    parser.add_argument("-dataset_id", "-d", type=str, default="10m", help="ID of the dataset resolution: Choices are " + ", ".join(ArcticDEM_datasets) + ". Default 10m.")
    parser.add_argument("-N", type=int, default=1, help="Number of parallel processes. Default 1.")
    parser.add_argument("--create_list", action="store_true", default=False, help="Create the output list of all files to download.")
    parser.add_argument("--unzip", action="store_true", default=False, help="Unzip all the files you have just downloaded.")
    return parser.parse_args()

if __name__ == "__main__":
    # print(FABDEM_file_template)
    # result = re.search("https://data.bris.ac.uk/datasets/25wfy0f9ukoge2gs7a5mqpq2j7/[\w-]+?_FABDEM_V1-0.zip".replace(".","\."), "https://data.bris.ac.uk/datasets/25wfy0f9ukoge2gs7a5mqpq2j7/N00E000-N10E010_FABDEM_V1-0.zip")
    # print(result)
    # create_list_of_links()
    args = import_and_parse_args()

    downloader = ArcticDEM_Downloader(dataset_id=args.dataset_id)

    if args.create_list:
        downloader.create_list_of_links()

    if args.unzip:
        downloader.unzip_downloaded_files(to_subdirs=True,
                                          fname_regex_filter="\.tar\.gz\Z",
                                          overwrite=True,
                                          verbose=True)

    if (not args.create_list) and (not args.unzip):
        downloader.download_files(N_subprocs=args.N)
