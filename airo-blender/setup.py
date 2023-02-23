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
        "blender-asset-tracer",  # used to find the tags of assets
        "tqdm",  # used to display progress bars
        "pydantic",  # for building dataset formats
        "opencv-python",
    ],
    packages=["airo_blender"],
)
