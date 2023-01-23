import setuptools

setuptools.setup(
    name="airo_blender",
    version="0.0.1",
    description="Synthetic data generation in Blender for robotic manipulation",
    author="Victor-Louis De Gusseme",
    author_email="victorlouisdg@gmail.com",
    install_requires=[
        "numpy",
        "scipy",
        "xmltodict",  # used to parse urdf files to python dictionaries
    ],
    packages=["airo_blender"],
)