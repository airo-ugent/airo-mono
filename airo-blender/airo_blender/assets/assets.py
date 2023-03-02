import os
import time
from pathlib import Path

import bpy
from blender_asset_tracer import blendfile
from blender_asset_tracer.blendfile import iterators
from tqdm import tqdm

asset_codes = (b"OB", b"MA", b"WO", b"GR", b"NT", b"AC")
asset_types = ("objects", "materials", "worlds", "collections", "node_groups", "actions")
code_to_type = {code: asset_type for code, asset_type in zip(asset_codes, asset_types)}


def load_asset(name: str, relative_path: str, library: str, type: str, **kwargs) -> bpy.types.ID:
    """Load a single asset from a .blend file given the name of the asset and the path to its blend file relative to
    the directory of its library .

    Refs:
        https://blender.stackexchange.com/questions/244971
        https://developer.blender.org/T87235

    Args:
        name: The name of the asset to load in its .blend file.
        relative_path: The path to the .blend file with the asset, relative to the directory of its library.
        library: The name of the library to load the asset from.
        type: The type of the asset to load. TODO: If None, several types will be tried in a fixed order.
        **kwargs: Unused keyword arguments, this allows user to call the function by unpacking a dictionary.

    Returns:
        The datablock of the loaded asset.
    """
    datablocks_before_load = set(getattr(bpy.data, type))

    asset_libraries = bpy.context.preferences.filepaths.asset_libraries
    if library is not None:  # TODO later? If None, all available libraries will be searched.
        if library in asset_libraries:
            library_path = asset_libraries[library].path
            asset_absolute_path = os.path.join(library_path, relative_path)

    # polyhaven_path = bpy.context.preferences.filepaths.asset_libraries["Poly Haven"].path

    with bpy.data.libraries.load(asset_absolute_path, assets_only=True) as (datablocks_source, datablocks_destination):
        if name not in getattr(datablocks_source, type):
            print(f"No asset with name {name} found in {asset_absolute_path}")

        getattr(datablocks_destination, type).append(name)

    datablock_after_load = set(getattr(bpy.data, type))
    new_datablocks = list(datablock_after_load - datablocks_before_load)

    return new_datablocks[0]


def available_assets() -> list[dict]:
    """Discovers all available assets in the asset libraries. Also loads the tags of each asset.

    Warning: this function is slow because it has to open and read all .blend files in the asset libraries.
    On my laptop, it takes about 1 minutes to process all Poly Haven assets.

    Hopefully we can replace this function with native bpy version in the future. See:
    https://blender.community/c/rightclickselect/GM7Q

    Returns:
        list[dict]: For each asset a dictionary with its tags and the information needed to load it.
    """
    start = time.time()

    asset_infos = []
    asset_libraries = bpy.context.preferences.filepaths.asset_libraries
    for asset_library in asset_libraries:
        print(f"Scanning .blend files from asset library: {asset_library.name}")
        asset_paths = [asset_path for asset_path in Path(asset_library.path).glob("**/*.blend")]

        for asset_path in tqdm(asset_paths):
            bf = blendfile.open_cached(asset_path)
            for code in asset_codes:
                for block in bf.find_blocks_from_code(code):
                    asset_metadata = block.get_pointer((b"id", b"asset_data"), default=None)
                    if not asset_metadata:
                        continue
                    tags = asset_metadata.get_pointer((b"tags", b"first"))
                    tag_names = [asset_tag[b"name"].decode("utf8") for asset_tag in iterators.listbase(tags)]
                    asset_info = {
                        "name": block.id_name.decode("utf-8")[2:],  # The first two characters are the code
                        "library": asset_library.name,
                        "relative_path": os.path.relpath(asset_path, asset_library.path),
                        "type": code_to_type[block.code],
                        "tags": tag_names,
                    }
                    asset_infos.append(asset_info)
            blendfile.close_all_cached()

    end = time.time()
    print(f"Found {len(asset_infos)} assets in {end - start:.2f} seconds")
    return asset_infos


def available_assets_without_tags() -> list[dict]:
    """Discovers all available assets in the asset libraries.

    Note: this function is about 10x faster than available_assets().

    Returns:
        list[dict]: For each asset a dictionary with the information nedded to load it.
    """
    start = time.time()

    asset_infos = []
    asset_libraries = bpy.context.preferences.filepaths.asset_libraries
    for asset_library in asset_libraries:
        print(f"Scanning .blend files from asset library: {asset_library.name}")
        asset_paths = [str(asset_path) for asset_path in Path(asset_library.path).glob("**/*.blend")]
        for asset_path in tqdm(asset_paths):
            with bpy.data.libraries.load(asset_path, assets_only=True) as (datablocks_source, _):
                for asset_type in asset_types:
                    assets_of_type = getattr(datablocks_source, asset_type)
                    for asset in assets_of_type:
                        asset_info = {
                            "name": asset,
                            "library": asset_library.name,
                            "relative_path": os.path.relpath(asset_path, asset_library.path),
                            "type": asset_type,
                        }
                        asset_infos.append(asset_info)

    end = time.time()
    print(f"Found {len(asset_infos)} assets in {end - start:.2f} seconds")
    return asset_infos
