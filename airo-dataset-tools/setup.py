import setuptools
from setuptools import find_packages

setuptools.setup(
    name="airo-dataset-tools",
    version="0.0.1",
    author="Victor-Louis De Gusseme",
    author_email="victorlouisdg@gmail.com",
    description="TODO",
    install_requires=[
        "numpy",
        "pydantic",
        "pycocotools",
        "xmltodict",
        "tqdm",
        "fiftyone",  # visualization
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "airo-dataset-tools = airo_dataset_tools.cli:cli",
        ],
    },
    package_data={"airo_dataset_tools": ["py.typed"]},
)
