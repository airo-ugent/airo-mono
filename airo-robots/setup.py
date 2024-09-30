import pathlib

import setuptools

root_folder = pathlib.Path(__file__).parents[1]
setuptools.setup(
    name="airo_robots",
    version="2024.1.0",
    description="Interfaces, hardware implementations of those interfaces and other functionalities to control robot manipulators and grippers at the Ghent University AI and Robotics Lab",
    author="Thomas Lips",
    author_email="thomas.lips@ugent.be",
    install_requires=[
        "numpy",
        "ur-rtde>=1.5.7",  # cf https://github.com/airo-ugent/airo-mono/issues/52
        "click",
        "airo-typing",
        "airo-spatial-algebra",
        "airo-tulip==0.0.9",
    ],
    packages=setuptools.find_packages(),
    package_data={"airo_robots": ["py.typed"]},
)
