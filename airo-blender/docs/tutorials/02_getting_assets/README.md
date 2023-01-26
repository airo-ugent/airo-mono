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

## 2.2 Poly Haven :sunrise_over_mountains:
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

### 2.3.2 Creating a snapshot of the available assets :bookmark_tabs:
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
> Note that use we can use Python dictionary unpacking to call our `ab.load_asset(**asset_info)`.
> This is because we chose the same name for keys of dictionary and the arguments of the function.
> Additionally we allowed that additional `kwargs` are passed to the function, like the `tags`, which the function does not use.

## 2.4 Filling a scene with coherent assets :city_sunset:
### 2.4.1 Filtering assets :croissant:
We now have access to more than 1000 assets.
However, to build synthetic scenes for the tasks we're interested in, we want to have control over their occurance.

Say we want to create a croissant detector.
Lucky for us, Poly Haven has just the right model!

We would think we could load it like so:
```python
croissant_info = [asset for asset in assets if asset["name"] == "croissant"][0]
croissant = ab.load_asset(**croissant_info)
```
However there's a problem.
If you look at your Blender scene after running this code, you won't see the croissant yet.
To understand why, we need to clarify some Blender concepts first in the following section.

### 2.4.2 Collections, collection instances and objects :mag:
Poly Haven shares all their models as **Collections**, instead as plain **Objects**.
They do this because Blender has a special type of object, a **Collection Instance**.

The difference between these 3 is the following:

* **Collection**: a group of objects. Can contain multiple object types (e.g. meshes and lights). Can be used as a "master copy" / orginal version that's centered in the world and not directly used in a Scene.
* **Collection Instance**: can be thought of as "shallow" copy of the orignal. You can move, rotate and scale this model, but you cannot edit the shape of its underlying geometry. A Collection instance is implemented as an Empty object, with its `.instance_type` property set to `COLLECTION` and its `.instance_collection` set to the chosen collection.
*  "Real" **Object**: This is basically any of the [Blender object types](https://docs.blender.org/manual/en/latest/scene_layout/object/types.html) (meshes, cameras, lights etc.) except Empties with their `.instance_type` set to `COLLECTION`. Real objects' data can be directly edited.

> Instancing is not the only way to share data between objects.
> Every type of object has associated [object data](https://docs.blender.org/manual/en/latest/scene_layout/object/introduction.html).
> An object is essentially a name and transform (location, rotation, scale), linked to object data.
> This object data can be completely or partially shared by [linking](https://docs.blender.org/manual/en/latest/scene_layout/object/editing/duplicate_linked.html).

> In the Blender you can [instance a collection](https://docs.blender.org/manual/en/latest/scene_layout/object/properties/instancing/collection.html) with `Add > Collection Instance`.
> You then get an Empty object that reference the collection you chose.
> If you want more freedom to edit the created instance, you can click `Object > Apply > Make Instances Real`. The Empty object now gets remove and the objects of collection get [duplicated](https://docs.blender.org/manual/en/latest/scene_layout/object/editing/duplicate.html).
> Note that duplicated objects are not completely separate by default, some data-blocks like materials remain shared.

For the croissant, simply transforming it sufficient, so we'll want to create a collection instance.
We can create a croissant instance like so:
```python
bpy.ops.object.collection_instance_add(collection=croissant_collection.name)
croissant = bpy.context.object
```

TODO render of croissant here

### 2.4.3 Adding randomization :game_die:
A single croissant is a bit sad, so lets add a whole bunch of them:
```python
for _ in range(20):
    bpy.ops.object.collection_instance_add(collection=croissant_collection.name)
    croissant = bpy.context.object

    # Randomize location in the z=0 plane
    x, y = np.random.uniform(-0.5, 0.5, size=2)
    croissant.location = x, y, 0

    # Randomize rotation around the z-axis
    rz = np.random.uniform(0, 2 * np.pi)
    croissant.rotation_euler = 0, 0, rz
```
When you run your script now, you should see a whole bunch of floating croissants.
Let's add a random table to rest the croissant on:

```python
def table_filter(asset_info: dict) -> bool:
    if asset_info["type"] != "collections":
        return False

    if "table" not in asset_info["tags"]:
        return False

    not_tables = ["desk_lamp_arm_01", "CoffeeCart_01", "wicker_basket_01"]
    if asset_info["name"] in not_tables:
        return False

    return True


tables = [asset for asset in assets if table_filter(asset)]
random_table = np.random.choice(tables)

table_collection = ab.load_asset(**random_table)
bpy.ops.object.collection_instance_add(collection=table_collection.name)
table = bpy.context.object
```
Now you should see a table in your scene, however your croissants are still scattered on the floor.
Let's change that by using the [bounding box](https://blender.stackexchange.com/a/8470/161432) of the table.
It's a bit tricky because Blender doesn't supply us with the bounding boxes of collections of collections instances.
So to find the bounding box of our instance, we have to iterate over mesh objects in collection it references, and calculate the bounding box of those bounding boxes.
```python
# Bounding box of the table
min_negative_corner = np.array([np.inf, np.inf, np.inf])
max_positive_corner = np.array([-np.inf, -np.inf, -np.inf])

for object in table_collection.objects:
    print(object.name)
    bounding_box = np.array(object.bound_box)
    negative_corner = bounding_box.min(axis=0)
    positive_corner = bounding_box.max(axis=0)

    min_negative_corner = np.minimum(min_negative_corner, negative_corner)
    max_positive_corner = np.maximum(max_positive_corner, positive_corner)

x_min, y_min, z_min = min_negative_corner
x_max, y_max, z_max = max_positive_corner
```
However, we also provide a convenience function in `airo_blender` that handles (globally) axis-aligned bounding boxes for objects, collection instances and collections.
```python
# TODO implement
```

# 2.5 Conclusion
Now you know how to load Blender assets from your Asset Libraries into your scenes from Python.
You also know how to filter these assets, to bring structure and coherence to your scenes.

However it's up to you how far you want to take this.
It's still an open question how much coherence and context matter for synthetic data generation.
For example, you might be able to train a pretty good croissant detector by cutting and pasting croissants on random images from the internet.
I guess it all depends on which task you are trying to learn.