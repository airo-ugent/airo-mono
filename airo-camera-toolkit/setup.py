import pathlib

import setuptools

root_folder = pathlib.Path(__file__).parents[1]
setuptools.setup(
    name="airo_camera_toolkit",
    version="2024.1.0",
    description="Interfaces and common functionality to work with RGB(D) cameras for robotic manipulation at the Ghent University AI and Robotics Lab",
    author="Thomas Lips",
    author_email="thomas.lips@ugent.be",
    install_requires=[
        "numpy<2.0",
        "opencv-contrib-python==4.8.1.78",  # We need opencv contrib for the aruco marker detection, but when some packages install (a different version of) opencv-python-headless, this breaks the contrib version. So we install both here to make sure they are the same version.
        "opencv-python-headless==4.8.1.78",  # Lock to match contrib version.
        "matplotlib",
        "rerun-sdk>=0.11.0",
        "click",
        "open3d",
        "loguru",
        "airo-typing",
        "airo-spatial-algebra",
        "airo-dataset-tools",
    ],
    extras_require={"hand-eye-calibration": ["airo-robots"]},
    packages=setuptools.find_packages(exclude=["test"]),
    entry_points={
        "console_scripts": [
            "airo-camera-toolkit = airo_camera_toolkit.cli:cli",
        ],
    },
    package_data={"airo_camera_toolkit": ["py.typed"]},
)
