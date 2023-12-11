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
        "opencv-contrib-python==4.8.1.78",  # We need opencv contrib for the aruco marker detection, but when some packages install (a different version of) opencv-python-headless, this breaks the contrib version. So we install both here to make sure they are the same version.
        "opencv-python-headless==4.8.1.78",  # Lock to match contrib version.
        "matplotlib",
        "rerun-sdk>=0.11.0",
        "open3d",
        "click==8.1.3",  # 8.1.4 breaks mypy
        "loguru",
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
    entry_points={
        "console_scripts": [
            "airo-camera-toolkit = airo_camera_toolkit.cli:cli",
        ],
    },
    package_data={"airo_camera_toolkit": ["py.typed"]},
)
