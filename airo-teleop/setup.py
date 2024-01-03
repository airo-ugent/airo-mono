import pathlib

import setuptools

root_folder = pathlib.Path(__file__).parents[1]
setuptools.setup(
    name="airo_teleop",
    version="0.0.1",
    description="teleoperation functionality for manually controlling manipulators and grippers using gaming controllers etc. at the Ghent University AI and Robotics Lab",
    author="Thomas Lips",
    author_email="thomas.lips@ugent.be",
    install_requires=["pygame", "click", "loguru"],
    extras_require={
        "external": [
            f"airo_typing @ file://localhost/{root_folder}/airo-typing",
            f"airo_spatial_algebra @ file://localhost/{root_folder}/airo-spatial-algebra",
            f"airo_robots @file://localhost/{root_folder}/airo-robots",
        ]
    },
    packages=setuptools.find_packages(),
)
