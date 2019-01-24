"""
Download the dataset from KITTI if the dataset isn't available in the data directory
"""

import os
import shutil
import zipfile
from urllib.request import urlretrieve

from tqdm import tqdm


class DownloadProgress(tqdm):
    """
    Show the progress of the download
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


class DataSanity:

    def __init__(self, data_dir):
        """
            Download if the data is not available
        :param data_dir: data directory in which the dataset will be downloaded and extracted
        """
        self.data_dir = data_dir
        self.data_road_filename = 'data_road.zip'
        self.data_road_path = os.path.join(data_dir, 'data_road')

    def dispatch(self):
        """
        Determine if the dataset has to be downloaded and download if it has to be.
        """

        if not os.path.exists(self.data_road_path):
            # minor cleanup on the dataset directory
            if os.path.exists(self.data_road_path):
                shutil.rmtree(self.data_road_path)
            os.makedirs(self.data_road_path)

            # Download Kitti dataset
            print('Downloading KITTI Dataset...Please wait...')
            with DownloadProgress(unit='B', unit_scale=True, miniters=1) as pbar:
                urlretrieve(
                    'https://zenodo.org/record/1154821/files/potsdam.zip?download=1',
                    os.path.join(self.data_road_path, self.data_road_filename),
                    pbar.hook)
            print("...Download Completed...")
            # Extract the dataset
            print('...Extracting Dataset...')
            zip_ref = zipfile.ZipFile(os.path.join(self.data_road_path, self.data_road_filename), 'r')
            zip_ref.extractall(self.data_dir)
            zip_ref.close()

            # Remove zip file to save space
            os.remove(os.path.join(self.data_road_path, self.data_road_filename))
        else:
            print("Retrieving Data...")
