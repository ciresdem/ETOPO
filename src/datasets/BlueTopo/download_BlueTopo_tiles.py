# -*- coding: utf-8 -*-

import os
import ast
import urllib.request, urllib.parse
import re
import geopandas

###############################################################################
# Import the project /src directory into PYTHONPATH, in order to import all the
# other modules appropriately.
import import_parent_dir; import_parent_dir.import_src_dir_via_pythonpath()
###############################################################################
import utils.traverse_directory
import utils.configfile
import datasets.download_dataset as download_dataset

# TODO: Look into the post-Sandy tiles, see if we have them all. Updates are here: https://www.ngdc.noaa.gov/mgg/inundation/sandy/data/tiles/

class BlueTopo_Downloader(download_dataset.DatasetDownloader_BaseClass):
    """A downloader class for downloading subsets of the CUDEM dataset. CUDEM tiles
    come in various vertical datums, necessitating having them in separate source_dataset
    groups. Each group in this case has 2 sets of tiles which are defined in the
    "CUDEM/data/CUDEM_source_urls.txt" file.
    """
    def __init__(self):
        configfile = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                     "BlueTopo_config.ini"))
        # print(configfile)
        self.config = utils.configfile.config(configfile)
        local_data_dir = self.config._abspath(self.config.source_datafiles_directory)

        remote_data_dir = r'https://noaa-ocs-nationalbathymetry-pds.s3.amazonaws.com/BlueTopo/'

        # self.url_lookup_file = self.config._abspath(self.config.source_urls_lookup_file)
        # with open(self.url_lookup_file, 'r') as f:
        #     txt = f.read()
        #     # print(txt)
        #     self.urls_tuple = ast.literal_eval(txt)[CUDEM_groupname]
        #     f.close()

        # print(self.urls_tuple)

        # I can initialize this twice, later with the second website URL
        super().__init__(remote_data_dir, local_data_dir)

    def read_latest_tile_scheme_gpkg(self, verbose=True):
        # In the BlueTopo tile scheme folder, get the latest GPKG in that directory. Read and return the table as a DataFrame
        tile_gpkg_dir = self.config._abspath(self.config.tile_scheme_local_directory)
        # Get the list of all the Geopackage files in the tile scheme directory.
        tile_gpkg_list = [os.path.join(tile_gpkg_dir, fn) for fn in os.listdir(tile_gpkg_dir) if os.path.splitext(fn)[1].lower() == ".gpkg"]
        # Sort ascending by file name.
        tile_gpkg_list.sort()

        latest_gpkg_fname = tile_gpkg_list[-1]
        if verbose:
            print("Reading", os.path.basename(latest_gpkg_fname))
        df = geopandas.read_file(latest_gpkg_fname)
        # print(df)
        # print(df.columns)
        return df

    def create_list_of_links(self, write_to_file=True, verbose=True):
        # Read the latest "BlueTopo-Tile_Scheme" .gpkg file, get all the tile URLs in there.
        # If 'exclude_existing', exclude the tiles that already exist.
        # return the list of tiles to download.
        df = self.read_latest_tile_scheme_gpkg(verbose=verbose)

        urls_series = [url for url in df['GeoTIFF_Link'].tolist() if url is not None]

        if write_to_file:
            urls_fname = self.url_list

            f = open(urls_fname, 'w')
            f.write("\n".join(urls_series))

            if verbose:
                print(urls_fname, "written with {0} urls.".format(len(urls_series)))

        return urls_series
        # if exclude_existing:
        #     urls_series_existing = []
        #     local_dirpath = self.local_data_dir
        #     remote_dirpath = self.website_base_url
        #
        #     for url in urls_series:
        #         local_path = url.replace(remote_dirpath, local_dirpath + ("/" if (remote_dirpath[-1] == "/") else ""))
        #
        #         if os.path.exists(local_path)
        #         print(url, "-->", local_path)
        #
        # print(urls_series)
        # print(len(urls_series), "distinct links.")

        # url_list_total = []
        # for web_url in self.urls_tuple:
        #     ### Thankfully, each URL contains a link to a "urllistXXXX.txt file, which makes this a lot easier.
        #     # Get the webpage html, read it, decode to ascii
        #     html_text = urllib.request.urlopen(web_url, data=None).read().decode("ascii")
        #     # print()
        #     # print(web_url)
        #     # print(html_text)
        #     urllist_fname = re.search(urlfile_regex, html_text).group()
        #     # print(urllist_fname)
        #     url_list_url = urllib.parse.urljoin(web_url, urllist_fname)
        #     url_list_urls = [line.strip() for line in str(urllib.request.urlopen(url_list_url, data=None).read().decode("ascii")).splitlines() if len(line.strip()) > 0]
        #
        #     # Read all the tile-name URLs into a unique set.
        #     tile_urls_list = sorted(list(set([url for url in url_list_urls if re.search(file_regex, url) != None])))
        #     print(web_url, "has", len(url_list_urls), "URLs in", urllist_fname + ",", len(tile_urls_list), "unique tiles.")
        #     # print("\n".join(tile_urls_list))
        #     # print()
        #     url_list_total.extend(tile_urls_list)
        #
        # if write_to_file:
        #     with open(self.url_list, 'w') as f:
        #         f.write("\n".join(url_list_total))
        #     print(self.url_list, "written with", len(url_list_total), "tile URLs.")

    def download(self,N_subprocs=5):
        wget_args = "-np -r -nH -L -e robots=off --no-check-certificate --cut-dirs=3"
        self.download_files(N_subprocs=N_subprocs, include_speed_strings=True, wget_extra_args=wget_args)

if __name__ == "__main__":
    BT = BlueTopo_Downloader()
    BT.create_list_of_links()

    BT.download_files(N_subprocs=5)
