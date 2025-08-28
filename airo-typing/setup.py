import setuptools

setuptools.setup(
    name="airo_typing",
    version="2025.8.0",
    description="Python type definitions for use in the python packages at the Ghent University AI and Robotics Lab",
    author="Thomas Lips",
    author_email="thomas.lips@ugent.be",
    install_requires=["numpy>=2.0"],
    packages=setuptools.find_packages(exclude=["test"]),
    package_data={"airo_typing": ["py.typed"]},
)
