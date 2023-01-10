# Tutorial 1 - Getting to know the Python API :notebook_with_decorative_cover:
Welcome to the first tutorial of this series.
In this tutorial we will get to know the Blender Python API and render your first ray-traced synthetic image!

We will cover:
* Opening Blender
* Using the Python console
* Running Python scripts and some basic command
* Rendering an image
* Enabling Blender's ray tracer: Cycles
* Saving additional annotations, e.g. the center of an object

> The full Blender Python API docs can be [here](https://docs.blender.org/api/current/index.html).

## 1.1 Opening Blender :art:
Lets start by opening up Blender. You can do this by running the `blender` executable from your terminal, e.g:
```
./airo-blender/blender/blender-3.4.1-linux-x64/blender
```
Because we will be running the `blender` executable often, so I recommend adding its directory to your system `PATH`.
Please see [here](../../adding_blender_to_bashrc.md) for an explanation.
Then you will be able to open Blender by simply running `blender`.

## 1.2 The Python Console :snake:
Now Blender is opened, look for and click on the `Scripting` tab in the top bar.
This simply changes Blender's UI layout to be more convenient for scripting.

On the left, a `Python Console` should have appeared.

### 1.2.1 Your first command
Time to run your first command! In the console type:
```python
bpy.data.objects["Cube"].location.z += 0.5
```
Then press `Enter`.
This should have moved the default cube up by 0.5 meters.
If you missed it, you can repeat it by pressing the `up-arrow` on your keyboard followed by `Enter`.

## 1.3 Running Python scripts :scroll:
The Python Console is handy for quickly testing out commands, but for more complex tasks we want to put our code in a Python script.
Make a new python file `tutorial_1.py` e.g. in a new directory `airo-blender-tutorials` somewhere on your pc.
Put the following code in that file:
```python
import bpy

bpy.data.objects["Cube"].location.z += 5.0
```
Save the file and in its directory run:
```
blender -P tutorial_1.py
```
This should launch Blender again and show the default scene with the cube moved up by 5 meters.

> The `-P` option in the command above tells Blender to execute the python script. Without it, blender will try, unsuccessfully, to open it as a `.blend` file.


Now we can start editing this file and build a more complex scene.

### 1.3.1 Building a simple scene
You've probably already noticed that when you open Blender, there's always a default scene already present.
This scene contains 3 objects, a camera, a light and a cube.

To add a cylinder to the scene, add this line to your script and rerun:
```python
bpy.ops.mesh.primitive_cylinder_add()
```
This should create a cylinder named `"Cylinder"` in your scene.

Let's move the cube and cylinder around a bit to make a basic scene:
```python
import bpy

# Create the cylinder
bpy.ops.mesh.primitive_cylinder_add()

# We can assign the blender Objects to variables for easy access
cube = bpy.data.objects["Cube"]
cylinder = bpy.data.objects["Cylinder"]

# Playing with the objects' properties
cube.scale = (2.0, 2.0, 0.1)
cube.location.z = 0.05
cylinder.location.z = 1.0
cylinder.rotation_euler.x = 3.14 / 2.0
```

### 1.3.2 Adding a material
We have a simple scene now, unfortunately, it's pretty gray and boring right now.
To make it more interesting we need to add materials to our objects.

Adding a red material to the cylinder can be done using these commands:
```python
red = (1.0, 0.0, 0.0, 1.0)
material = bpy.data.materials.new(name="Cylinder Material")
material.use_nodes = True
bdsf = material.node_tree.nodes["Principled BSDF"]
bdsf.inputs["Base Color"].default_value = color
cylinder.data.materials.append(material)
```
You might be thinking right now: "Wow that's a lot of code for such a simple use case!", and I would agree with you.
The Blender Python API has a tendency of being very very modular, to the point where it can start feeling clunky.
That's where the `airo_blender` comes in.
One of the things we provide is functions to aid 3D scene construction.
In this case, we can replace the snippet above with the code below:
```python
import airo_blender as ab

red = (1.0, 0.0, 0.0, 1.0)
ab.add_material(cyclinder, red)
```
> Note: by default Blender shows your scene in a fast to render "Solid" shading mode.
> To see what your material will look like in the final render, change it to "Rendered" mode.
> (Click the right-most little sphere in the top-right corner of the viewport.)

### 1.3.3 A nicer background
By default, the Blender "world" has a pretty dark gray background.
Let's set that to a brighter color.

```python
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs["Color"].default_value = (0.02, 0.0, 1.0, 1.0)
```

## 1.4 Rendering an image :camera:
Rendering an image can be done simply by adding this line at the end of your scripts:
```python
bpy.ops.render.render()
```
When running your script Blender will now create your scene and then render it into an image.
To view the rendered image, click on the `Rendering` tab in the top bar.

By now your code should look something like this:
```python
import bpy
import airo_blender as ab

# Create the cylinder
bpy.ops.mesh.primitive_cylinder_add()

# We can assign the blender Objects to variables for easy access
cube = bpy.data.objects["Cube"]
cylinder = bpy.data.objects["Cylinder"]

# Playing the objects' properties
cube.scale = (2.0, 2.0, 0.1)
cube.location.z -= 0.1
cylinder.scale = (0.5, 0.5, 1.0)
cylinder.location.z = 0.5
cylinder.rotation_euler.x = 3.14 / 2.0

# Adding a nice material
red = (1.0, 0.0, 0.0, 1.0)
ab.add_material(cylinder, red)

# Making the background brighter
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs["Color"].default_value = (1.0, 0.9, 0.7, 1.0)

# Rendering the scene into an image
bpy.ops.render.render()
```

## 1.5 Enabling Blender's phyiscally-based path tracer: Cycles :high_brightness:
The rendered image should already look ok, but you might notice that some lighting effects are missing.
This is because by default Blender uses its real-time renderer EEVEE.
To tell Blender to use Cycles takes only one line of code:
```python
bpy.context.scene.render.engine = 'CYCLES'
```
Additionally, we want to tell Blender how much noise we can tolerate in the rendered output:
```python
bpy.context.scene.cycles.adaptive_threshold = 0.1
```

## 1.6 Saving additional annotations :floppy_disk:

## The End :tada:
Congratulations, you've reached the end of the first tutorial and hopefully generated your first piece of synthetic data!
You now also understand the basics of synthetic data generation in Blender.
As you've seen, Blender's Python API is pretty explicit and straightforward.

In the following tutorials, we will teach you how to add assets to your scene, add randomization and work with keypoint
annotations!

