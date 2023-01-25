# Tutorial 2 - Getting assets :gift:


## 2.1 Introduction :books:
Assets can be many things, models, materials, environment textures, lights etc.
They are a powerful way to improve the fidelity of your synthetic data.
However, working with assets can be **messy**.
There are many different tools for developing 3D assets, and as such also many different conventions, features and file formats.

Since version 3.0, Blender improved the workflow for storing and using assets significantly by introducing the [Asset Browser](https://docs.blender.org/manual/en/latest/editors/asset_browser.html).
The idea is that you download a "raw" 3D asset from the internet, e.g. a mesh as an `.obj` file, or a [PBR material](https://marmoset.co/posts/physically-based-rendering-and-you-can-too/) as a set of `.png` or `.jpg` texture maps.

Then you manual import this asset into Blender.
When you're lucky, this is painless, e.g. you import an `.obj` and it results in a correctly oriented and scaled Blender object.
Often however you will have to do some tweaking to get the asset looking right.
For PBR materials, you often get a zip of texture maps, and you need to build a simple shader graph combine into a Blender material.

Finally when you're satisfied with how the asset looks, you can right-click it in the Outliner to `Mark as Asset`.
From that point on, the "raw" asset has become a **Blender asset**.

If you then save your `.blend` file into an **Asset Library**, which is a specific directory, by default `~/Documents/Blender/Assets/`, Blender will automatically discover it.
Then you can simply drag-and-drop any Blender asset from the Asset Browser into your scene.

## 2.2 Poly Haven :sunrise:
[Poly Haven](https://polyhaven.com/) is one of the best providers of high-quality free 3D assets.
It also has great integration with Blender through it's addon, which downloads all Poly Haven assets directly as Blender assets.
They encourage you to purchase it, but they also made it publically available on [github](https://github.com/Poly-Haven/polyhavenassets).

In this rest of this tutorial will show how to load the the Poly Haven assets from Python.
So before continuing, please install the addon and download the Poly Haven assets.

After finishing the installation and downloads, you should be able to throw together a scene like the one below by dragging and dropping a few assets.

![Poly Haven example scene](https://i.imgur.com/zShf7Od.jpg)

## 2.3 Loading Blender assets from Python :inbox_tray:
### 2.3.1 Loading a known asset :deciduous_tree:
Now we have some Blender assets, it's time to load them from Python.
If you know an asset's name, source file, type and library, you can load it like so:

```python
import bpy
import airo_blender as ab

# Identify the asset to load
asset_name = "woods"
asset_relative_path = "woods/woods.blend"
asset_library = "Poly Haven"
asset_type = "worlds"

woods = ab.load_asset(asset_name, asset_relative_path, asset_type, asset_library)
print(f'Imported asset "{woods.name}" with type {type(woods)}')

# Set the world to the loaded HDRI
bpy.context.scene.world = woods
```

After running the above code, if you set the `Viewport Shading` to `Rendered`, you should see that the world background was changed to a forest.

### 2.3.2 Creating a snapshot of the available assets :camera:
For synthetic data generation, we generally don't want to load a specific asset.
Often we want to load a few random assets from all the available assets.

Sadly, Blender does offer a fast way to query the available assets (e.g. as they are listed in the Asset Browser) from Python.
However this functionality will probably be added in the [future](https://blender.community/c/rightclickselect/GM7Q).

To work around this limitation, we will create a `asset_snapshot.json` file that contains an overview of the available assets.
Having this file has the additional benefit that we can version it and share it with collaborators.
This way we can check whether everyone has the same assets available locally, and make our synthetic data generation **reproducable**.

Below a script to create such a snapshot:

```python
import airo_blender as ab
import json

all_assets = ab.available_assets()
polyhaven_assets = [asset for asset in all_assets if asset["library"] == "Poly Haven"]

asset_snapshot = {"assets": polyhaven_assets}

with open("asset_snapshot.json", "w") as file:
    json.dump(asset_snapshot, file, indent=4)

```
You can also find it in this directory in `create_asset_snapshot.py` and run it like so:
```
blender -P create_asset_snapshot.py
```
This should result in a json file that looks like this:
```json
{
    "assets": [
        {
            "name": "river_small_rocks",
            "library": "Poly Haven",
            "relative_path": "river_small_rocks/river_small_rocks.blend",
            "type": "materials",
            "tags": [
                "terrain",
                "rock",
                "natural",
                "rough",
                "uneven",
                "stones",
                "pebbles",
                "river",
                "pathway",
                "debris",
                "scattered",
                "ground",
                "broken",
                "costal"
            ]
        }, ... # The other Poly Haven assets
    ]
}
```

We can then rewrite our first script that loaded the `woods` HDRI like so:

```python
import bpy
import airo_blender as ab
import json

with open("asset_snapshot.json", "r") as file:
    assets = json.load(file)["assets"]

worlds = [asset for asset in assets if asset["type"] == "worlds"]
woods_info = [asset for asset in worlds if asset["name"] == "woods"][0]
print(woods_info)

woods = ab.load_asset(**woods_info)
bpy.context.scene.world = woods
```


### 2.3.3 Asset snapshots and reproducability