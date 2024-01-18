import pathlib

import setuptools

root_folder = pathlib.Path(__file__).parents[1]
setuptools.setup(
    name="airo_spatial_algebra",
    version="2024.1.0",
    description="code for working with SE3 poses,transforms,... for robotic manipulation at the Ghent University AI and Robotics Lab",
    author="Thomas Lips",
    author_email="thomas.lips@ugent.be",
    install_requires=["numpy", "scipy", "spatialmath-python", "airo-typing"],
    packages=setuptools.find_packages(exclude=["test"]),
    # include py.typed to declare type information is available, see
    # https://mypy.readthedocs.io/en/stable/installed_packages.html#making-pep-561-compatible-packages
    package_data={"airo_spatial_algebra": ["py.typed"]},
)
