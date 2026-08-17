import pathlib

import setuptools

root_folder = pathlib.Path(__file__).parents[1]
setuptools.setup(
    name="airo_robots",
    version="2026.7.0",
    description="Interfaces, hardware implementations of those interfaces and other functionalities to control robot manipulators and grippers at the Ghent University AI and Robotics Lab",
    author="Thomas Lips",
    author_email="thomas.lips@ugent.be",
    install_requires=[
        "numpy>=2.0",
        "click",
        "loguru",
        "airo-typing>=2026.1.0",
        "airo-spatial-algebra>=2026.1.0",
    ],
    packages=setuptools.find_packages(),
    package_data={"airo_robots": ["py.typed"]},
    extras_require={
        "realman": ["Robotic_Arm"],
        "ur": ["ur-rtde>=1.6.0"],  # cf https://github.com/airo-ugent/airo-mono/issues/52
        # The FANUC driver also drives the Robotiq 2F-85 that is mounted on our FANUC, over the
        # controller's registers, so this extra covers Robotiq2F85Fanuc as well.
        "fanuc": ["airo-fanuc>=0.1.0"],
        "schunk": ["bkstools"],
        "kelo": ["airo-tulip>=0.4.0"],
    },
)
