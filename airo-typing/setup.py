import setuptools

setuptools.setup(
    name="airo_typing",
    version="2024.1.0",
    description="python type definitions for use in the python packages at the Ghent University AI and Robotics Lab",
    author="Thomas Lips",
    author_email="thomas.lips@ugent.be",
    install_requires=["numpy"],
    packages=setuptools.find_packages(exclude=["test"]),
    package_data={"airo_typing": ["py.typed"]},
)
