import setuptools
from setuptools import find_packages

setuptools.setup(
    name="airo-core",
    version="0.0.1",
    description="core utilities, interfaces and base class for robotic manipulation at the Ghent University AI and Robotics Lab",
    author="Thomas Lips",
    author_email="thomas.lips@ugent.be",
    install_requires=["numpy", "scipy", "spatialmath-python"],
    extra_requires={
        "develop": ["pre-commit", "mypy", "pytest", "pytest-xdist", "pytest-cov"]  # multi-core testing  # coverage
    },
    packages=find_packages(),
    package_data={"airo_core": ["py.typed", "version.txt"]},
)
