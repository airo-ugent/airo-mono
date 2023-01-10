![Synthetic data generation box keypoints example](https://i.imgur.com/ZpH0grX.jpg)

# airo-blender

A Python package on top of the [Blender Python API](https://docs.blender.org/api/current/index.html) to
generate synthetic data for robotic manipulation.

To get started, please see the [tutorials](docs/tutorials) directory after completing the installation.


## Installation
> Currently this is a "development" installation, where we assume you have this directory cloned locally. We develop and test this package on Ubuntu.

### Downloading Blender
Download the latest version of [Blender](https://www.blender.org/download/), currently `3.4.1`.
We provide the subdirectory [blender](blender) as a convenient but optional place to save and extract Blender. See the directory's [README.md](blender/README.md) for the expected directory structure after download.

### Installing the `airo_blender` Python package
Then, run the following commands from the `airo-blender` directory:
```
./blender/blender-3.4.1-linux-x64/3.4/python/bin/python3.10 -m ensurepip
./blender/blender-3.4.1-linux-x64/3.4/python/bin/pip3 install -e .
```
The first command installs `pip` for the Blender Python.
The second command installs the `airo_blender` package into it. The `-e` option


## Philosophy
This package is meant to be as lightweight and opt-in as possible.
We deliberately stick close to the Blender API and keep data explicit.
This means that we for example don't wrap the Blender datatypes with custom Python classes.

Alternatives to this library are:
* [BlenderProc](https://github.com/DLR-RM/BlenderProc)
* [kubric](https://github.com/google-research/kubric)
