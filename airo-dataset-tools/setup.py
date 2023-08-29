import pathlib

import setuptools
from setuptools import find_packages

root_folder = pathlib.Path(__file__).parents[1]
setuptools.setup(
    name="airo-dataset-tools",
    version="0.0.1",
    author="Victor-Louis De Gusseme",
    author_email="victorlouisdg@gmail.com",
    description="TODO",
    install_requires=[
        "numpy",
        "pydantic<2.0.0",  # pydantic 2.0.0 has a lot of breaking changes
        "opencv-contrib-python==4.7.0.72",
        "opencv-python-headless==4.7.0.72",  # opencv headless gets installed by some packages, so install it manually to match versions with opencv-contrib-python
        "pycocotools",
        "xmltodict",
        "tqdm",
        "fiftyone",  # visualization
        "Pillow",
        "types-Pillow",
        "albumentations",
    ],
    extras_require={
        "external": [
            f"airo_typing @ file://localhost/{root_folder}/airo-typing",
            f"airo_spatial_algebra @ file://localhost/{root_folder}/airo-spatial-algebra",
        ]
    },
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "airo-dataset-tools = airo_dataset_tools.cli:cli",
        ],
    },
    package_data={"airo_dataset_tools": ["py.typed"]},
)
