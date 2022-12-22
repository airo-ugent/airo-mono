import setuptools
from setuptools import find_packages
import pathlib
import os
root_folder = pathlib.Path(__file__).parents[1]

setuptools.setup(
    name="airo_spatial_algebra",
    version="0.0.1",
    description="code for working with SE3 poses,transforms,... for robotic manipulation at the Ghent University AI and Robotics Lab",
    author="Thomas Lips",
    author_email="thomas.lips@ugent.be",
    install_requires=["numpy", "scipy", "spatialmath-python"],
    packages=["airo_spatial_algebra", "airo_typing"],
    package_dir={"airo_typing": os.path.join(root_folder, "airo-typing","airo_typing")},
    package_data={"airo_spatial_algebra": ["py.typed"]},
)
