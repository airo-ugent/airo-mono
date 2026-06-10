import pathlib

import setuptools

root_folder = pathlib.Path(__file__).parents[1]
setuptools.setup(
    name="airo-dataset-tools",
    version="2026.5.0",
    author="Victor-Louis De Gusseme",
    author_email="victorlouisdg@gmail.com",
    description="Scripts for loading and converting datasets for the Ghent University AI and Robotics Lab",
    install_requires=[
        "numpy>=2.0",
        "pydantic>2.0.0",  # pydantic 2.0.0 has a lot of breaking changes
        # opencv-contrib-python and opencv-python-headless both install a cv2 module and conflict
        # with each other — whichever is installed last wins. fiftyone (in the [fiftyone] extra)
        # pulls in opencv-python-headless. To ensure contrib wins after installing the extra, run:
        #   pip install --force-reinstall opencv-contrib-python==4.10.0.84
        "opencv-contrib-python==4.10.0.84",
        "pycocotools",
        "xmltodict",
        "tqdm",
        "Pillow",
        "types-Pillow",
        "albumentations",
        "click",
        "airo-typing>=2026.1.0",
        "airo-spatial-algebra>=2026.1.0",
    ],
    extras_require={
        "fiftyone": ["fiftyone"],
    },
    packages=setuptools.find_packages(exclude=["test"]),
    entry_points={
        "console_scripts": [
            "airo-dataset-tools = airo_dataset_tools.cli:cli",
        ],
    },
    package_data={"airo_dataset_tools": ["py.typed"]},
)
