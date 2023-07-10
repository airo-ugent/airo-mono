import pathlib

import setuptools

root_folder = pathlib.Path(__file__).parents[1]
setuptools.setup(
    name="airo_camera_toolkit",
    version="0.0.1",
    description="Interfaces and common functionality to work with RGB(D) cameras for robotic manipulation at the Ghent University AI and Robotics Lab",
    author="Thomas Lips",
    author_email="thomas.lips@ugent.be",
    install_requires=[
        "numpy",
        "matplotlib",
        "opencv-contrib-python==4.7.0.72",  # opencv has a tendency to make breaking changes
        "rerun-sdk",
        "click==8.1.3",  # 8.1.4 breaks mypy
        "loguru",
        "pyrealsense2",
    ],
    extras_require={
        "external": [
            f"airo_typing @ file://localhost/{root_folder}/airo-typing",
            f"airo_spatial_algebra @ file://localhost/{root_folder}/airo-spatial-algebra",
            f"airo_robots @ file://localhost/{root_folder}/airo-robots",
            f"airo_dataset_tools @ file://localhost/{root_folder}/airo-dataset-tools",
        ]
    },
    packages=setuptools.find_packages(),
    package_data={"airo_camera_toolkit": ["py.typed"]},
)
