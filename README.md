# Kitti- Road Segmentation
> Lane Segmentation using several architectures.


[![Python Version][travis-image]][travis-url]
[![Build Status][travis-image]][travis-url]
[![License][license-image]][license-url]


It contains the code for both training and segmentation of lane lines using Deep Learning. Currently the supported architectures are ENET, UNET, Modified VGG.


## Features

- [x] The training code is very much scalable towards any new architecture.
- [x] All changes made in the config file will effect in the training process so that the training logic can be without hassle.
- [x] The training configuartion are easily tunable through the config file provided.

## Requirements

- The training module has been built using Pycharm 2018.1.4.
- The System requirement’s are 2.7 GHz Intel Core i5 with atleast 8 GB of RAM.

## Installation

#### OpenCV
You can use [Anaconda](https://conda.io/) to install `opencv` with the following command line.:

```
conda install -c conda-forge opencv
```

#### Image Augmentation
You can use [PIP](https://pypi.org/project/pip/) to install the module `imgaug` with the following command line.:

```
pip install imgaug
```

#### tensorflow
You can use [PIP](https://pypi.org/project/pip/) to install `tensorflow` with the following command line or please go through their official installation guideline[https://www.tensorflow.org/install/pip]:

```
pip install tensorflow
```


#### Keras
You can use [PIP](https://pypi.org/project/pip/) to install `keras` with the following command line or please go through their official installation guideline[https://keras.io/#installation]:

```
pip install keras
```

## Usage example

Run the following script to dispatch the trainer.


```
python3 train.py  --conf=./config.json
```

## Contribute

I would love for you to contribute to **KITT-Road Segmentation**, check the ``LICENSE`` file for more info.

## Meta

Stanly Moses – [@Linkedin](https://in.linkedin.com/in/stanlymoses) – stanlyhardy@yahoo.com

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/StanlyHardy/KITTI-Road-Segmentation](https://github.com/StanlyHardy/)

[python-image]:https://img.shields.io/badge/Made%20with-Python-1f425f.svg
[python-url]: https://docs.python.org/3/
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[license-image]: https://img.shields.io/badge/License-MIT-blue.svg
[license-url]: LICENSE

