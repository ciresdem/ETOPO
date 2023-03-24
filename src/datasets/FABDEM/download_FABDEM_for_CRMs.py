
## Use fetches to get all the FABDEM v1_2 tiles overlapping the CRMs.
## Written 2023.03.08

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

class FABDEM_Downloader(download_dataset.DatasetDownloader_BaseClass):
    """A downloader class for downloading subsets of the CUDEM dataset. CUDEM tiles
    come in various vertical datums, necessitating having them in separate source_dataset
    groups. Each group in this case has 2 sets of tiles which are defined in the
    "CUDEM/data/CUDEM_source_urls.txt" file.
    """
    def __init__(self):
        configfile = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                     "FABDEM_config.ini"))
        # print(configfile)
        self.config = utils.configfile.config(configfile)
        local_data_dir = self.config._abspath(self.config.source_datafiles_directory)

        remote_data_dir = None

        super().__init__(remote_data_dir, local_data_dir)

    # def create_list_of_links(self, write_to_file=True, verbose=True):
    #     # Read the latest "BlueTopo-Tile_Scheme" .gpkg file, get all the tile URLs in there.
    #     # If 'exclude_existing', exclude the tiles that already exist.
    #     # return the list of tiles to download.
    #     df = self.read_latest_tile_scheme_gpkg(verbose=verbose)
    #
    #     urls_series = [url for url in df['GeoTIFF_Link'].tolist() if url is not None]
    #
    #     if write_to_file:
    #         urls_fname = self.url_list
    #
    #         f = open(urls_fname, 'w')
    #         f.write("\n".join(urls_series))
    #
    #         if verbose:
    #             print(urls_fname, "written with {0} urls.".format(len(urls_series)))
    #
    #     return urls_series


if __name__ =="__main__":
    F = FABDEM_Downloader()
    F.download_files(N_subprocs=8)