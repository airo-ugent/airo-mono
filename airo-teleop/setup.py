import pathlib

import setuptools

root_folder = pathlib.Path(__file__).parents[1]
setuptools.setup(
    name="airo_teleop",
    version="2024.1.0",
    description="teleoperation functionality for manually controlling manipulators and grippers using gaming controllers etc. at the Ghent University AI and Robotics Lab",
    author="Thomas Lips",
    author_email="thomas.lips@ugent.be",
    install_requires=["pygame", "click", "loguru", "airo-typing", "airo-spatial-algebra", "airo-robots"],
    packages=setuptools.find_packages(exclude=["test"]),
)
