import pathlib

import setuptools

root_folder = pathlib.Path(__file__).parents[1]
setuptools.setup(
    name="airo_teleop",
    version="2025.7.0",
    description="Teleoperation functionality for manually controlling manipulators and grippers using gaming controllers etc. at the Ghent University AI and Robotics Lab",
    author="Thomas Lips",
    author_email="thomas.lips@ugent.be",
    install_requires=[
        "pygame",
        "click",
        "loguru",
        "airo-typing==2025.4.0",
        "airo-spatial-algebra==2025.4.0",
        "airo-robots==2025.4.0",
    ],
    packages=setuptools.find_packages(exclude=["test"]),
)
