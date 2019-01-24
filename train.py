#! /usr/bin/env python

"""
Lane Detection

    # Train a new model using kitti
    Usage: python3 train.py  --conf=./config.json

"""

import argparse
import json

from src.DataHandler import DataSanity
from src.frontend import Segment

# define command line arguments
argparser = argparse.ArgumentParser(
    description='Train and validate Kitti Road Segmentation Model')

argparser.add_argument(
    '-c',
    '--conf', default="config.json",
    help='path to configuration file')


def _main_(args):
    """
    :param args: command line argument
    """

    # parse command line argument
    config_path = args.conf

    # open and load the config json
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    # parse the json to retrieve the training configuration
    backend = config["model"]["backend"]
    input_size = (config["model"]["im_width"], config["model"]["im_height"])
    classes = config["model"]["classes"]
    data_dir = config["train"]["data_directory"]

    # Trigger the the dataset downloader if the dataset is not present
    DataSanity(data_dir).dispatch()

    # define the model and train
    segment = Segment(backend, input_size, classes)
    segment.train(config["train"])


if __name__ == '__main__':
    # parse the arguments
    args = argparser.parse_args()
    _main_(args)
