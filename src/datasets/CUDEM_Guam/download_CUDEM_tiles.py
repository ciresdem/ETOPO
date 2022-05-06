# -*- coding: utf-8 -*-

import os
import ast
import urllib
import re

###############################################################################
# Import the project /src directory into PYTHONPATH, in order to import all the
# other modules appropriately.
import import_parent_dir; import_parent_dir.import_src_dir_via_pythonpath()
###############################################################################
import utils.traverse_directory
import utils.configfile
import datasets.download_dataset as download_dataset

class CUDEM_Downloader(download_dataset.DatasetDownloader_BaseClass):
    """A downloader class for downloading subsets of the CUDEM dataset. CUDEM tiles
    come in various vertical datums, necessitating having them in separate source_dataset
    groups. Each group in this case has 2 sets of tiles which are defined in the
    "CUDEM/data/CUDEM_source_urls.txt" file.
    """
    def __init__(self, CUDEM_groupname="Guam"):
        self.config = utils.configfile.config(os.path.join(os.path.dirname(__file__),"CUDEM_{0}_config.ini".format(CUDEM_groupname)))
        local_data_dir = self.config._abspath(self.config.source_datafiles_directory)

        self.url_lookup_file = self.config._abspath(self.config.source_urls_lookup_file)
        with open(self.url_lookup_file, 'r') as f:
            txt = f.read()
            # print(txt)
            self.urls_tuple = ast.literal_eval(txt)[CUDEM_groupname]
            f.close()

        # print(self.urls_tuple)

        # I can initialize this twice, later with the second website URL
        super().__init__(self.urls_tuple[0], local_data_dir)

    def create_list_of_links(self,
                             file_regex="ncei(\w+)v1\.tif",
                             write_to_file=True):
        """This is a base-class virtual function. Since each directory structure, protocol, etc,
        of each data source is fairly unique, it's impractical to come up with one single
        funciton to traverse remote directories for everything. Create a subclass and inheret
        this class to use the other helper functions, but create an overriding "create_list_of_links()"
        class method in the subclass."""

        url_list_total = []
        urlfile_regex = "urllist(\d+)\.txt"
        for web_url in self.urls_tuple:
            ### Thankfully, each URL contains a link to a "urllistXXXX.txt file, which makes this a lot easier.
            # Get the webpage html, read it, decode to ascii
            html_text = urllib.request.urlopen(web_url, data=None).read().decode("ascii")
            # print()
            # print(web_url)
            # print(html_text)
            urllist_fname = re.search(urlfile_regex, html_text).group()
            # print(urllist_fname)
            url_list_url = urllib.parse.urljoin(web_url, urllist_fname)
            url_list_urls = [line.strip() for line in str(urllib.request.urlopen(url_list_url, data=None).read().decode("ascii")).splitlines() if len(line.strip()) > 0]

            # Read all the tile-name URLs into a unique set.
            tile_urls_list = sorted(list(set([url for url in url_list_urls if re.search(file_regex, url) != None])))
            print(web_url, "has", len(url_list_urls), "URLs in", urllist_fname + ",", len(tile_urls_list), "unique tiles.")
            # print("\n".join(tile_urls_list))
            # print()
            url_list_total.extend(tile_urls_list)

        if write_to_file:
            with open(self.url_list, 'w') as f:
                f.write("\n".join(url_list_total))
            print(self.url_list, "written with", len(url_list_total), "tile URLs.")

    def download(self,N_subprocs=5):
        wget_args = "-np -r -nH -L -e robots=off --no-check-certificate --cut-dirs=3"
        self.download_files(N_subprocs=N_subprocs, include_speed_strings=True, wget_extra_args=wget_args)

if __name__ == "__main__":
    CU = CUDEM_Downloader()
    # CU.create_list_of_links()
    CU.download()
