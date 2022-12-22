import pathlib

import setuptools
from setuptools import find_packages

root_folder = pathlib.Path(__file__).parents[1]
setuptools.setup(
    name="airo_spatial_algebra",
    version="0.0.1",
    description="code for working with SE3 poses,transforms,... for robotic manipulation at the Ghent University AI and Robotics Lab",
    author="Thomas Lips",
    author_email="thomas.lips@ugent.be",
    install_requires=[
        "numpy",
        "scipy",
        "spatialmath-python",
        f"airo_typing @ file://localhost/{root_folder}/airo-typing",
    ],
    packages=find_packages(),
    package_data={"airo_spatial_algebra": ["py.typed"]},
)
