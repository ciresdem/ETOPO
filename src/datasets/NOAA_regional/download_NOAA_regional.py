# -*- coding: utf-8 -*-

import os
import urllib.request
import re
import numpy
import shutil

##############################################################################
# Code for importing the /src directory so that other modules can be accessed.
import import_parent_dir
import_parent_dir.import_src_dir_via_pythonpath()
##############################################################################

import datasets.download_dataset
import source_dataset_NOAA_regional



class NOAA_Regional_Downloader(datasets.download_dataset.DatasetDownloader_BaseClass):

    def __init__(self):
        self.noaa_dset_obj = source_dataset_NOAA_regional.source_dataset_NOAA_regional()
        super().__init__(self.noaa_dset_obj.config.website_url, self.noaa_dset_obj.config.source_datafiles_directory)

    def create_list_of_links(self,
                             recurse=False,
                             file_regex="(?<=\>)\w+_(\d+)(m?)\.nc",
                             subdir_regex=None,
                             write_to_file=True):
        web_lines = [line.strip() for line in
                         str(urllib.request.urlopen(self.website_base_url, data=None).read().decode("ascii")).splitlines()
                         if (len(line.strip()) > 0)]

        # Get the list of file names.
        # file_names_list = [re.search(file_regex, line).group() for line in web_lines if re.search(file_regex, line) is not None]
        # Get the list of URLs.
        file_urls_list = [(self.noaa_dset_obj.config.website_url
                           + re.search("(?<=(\<a href\=\'catalog\.html))\?dataset=(\w+)/(\w+)_(\d+)(m?)\.nc", line).group()).replace("/catalog/regional/catalog.html?dataset=regionalDatasetScan/","/fileServer/regional/")
                          for line in web_lines if re.search(file_regex, line) is not None]


        if write_to_file:
            with open(self.url_list, 'w') as f:
                f.write("\n".join(file_urls_list))
                f.close()

            print(self.url_list, "written with", len(file_urls_list), "urls.")


if __name__ == "__main__":
    noaa = NOAA_Regional_Downloader()
    # noaa.create_list_of_links()
    # noaa.download_files(N_subprocs=5,
    #                     include_speed_strings=True,
    #                     )
