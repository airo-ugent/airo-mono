import pathlib

import setuptools

setuptools.setup(
    name="airo_typing",
    version="2026.10.1",
    description="Python type definitions for use in the python packages at the Ghent University AI and Robotics Lab",
    long_description=(pathlib.Path(__file__).parent / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Thomas Lips",
    author_email="thomas.lips@ugent.be",
    python_requires=">=3.11",
    install_requires=["numpy>=2.0"],
    packages=setuptools.find_packages(exclude=["test"]),
    package_data={"airo_typing": ["py.typed"]},
)
