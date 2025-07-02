import pathlib

import setuptools

root_folder = pathlib.Path(__file__).parents[1]
setuptools.setup(
    name="airo-dataset-tools",
    version="2025.7.0",
    author="Victor-Louis De Gusseme",
    author_email="victorlouisdg@gmail.com",
    description="Scripts for loading and converting datasets for the Ghent University AI and Robotics Lab",
    install_requires=[
        "numpy<2.0",
        "pydantic>2.0.0",  # pydantic 2.0.0 has a lot of breaking changes
        "opencv-contrib-python==4.8.1.78",  # See airo-camera-toolkit setup.py for explanation
        "opencv-python-headless==4.8.1.78",  # See airo-camera-toolkit setup.py for explanation
        "pycocotools",
        "xmltodict",
        "tqdm",
        "fiftyone",  # visualization
        "Pillow",
        "types-Pillow",
        "albumentations",
        "click",
        "airo-typing==2025.4.0",
        "airo-spatial-algebra==2025.4.0",
    ],
    packages=setuptools.find_packages(exclude=["test"]),
    entry_points={
        "console_scripts": [
            "airo-dataset-tools = airo_dataset_tools.cli:cli",
        ],
    },
    package_data={"airo_dataset_tools": ["py.typed"]},
)
