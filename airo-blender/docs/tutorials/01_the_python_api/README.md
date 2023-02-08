# Tutorial 1 - Getting to know the Python API :snake:
Welcome to the first tutorial of this series.
In this tutorial we will get to know the Blender Python API and render your first ray-traced synthetic image!

We will cover:
* Opening Blender
* Using the Python console
* Running Python scripts and some basic command
* Rendering an image
* Enabling Blender's ray tracer: Cycles
* Saving additional annotations, e.g. the pose of an object

> The full Blender Python API docs can be [here](https://docs.blender.org/api/current/index.html).

## 1.1 Opening Blender :art:
Let's start by opening up Blender. You can do this by running the `blender` executable from your terminal, e.g:
```
./airo-blender/blender/blender-3.4.1-linux-x64/blender
```
Because we will be running the `blender` executable often, I recommend adding its directory to your system `PATH`.
Please see [here](../../adding_blender_to_bashrc.md) for an explanation.
Then you will be able to open Blender by simply running `blender`.

## 1.2 The Python Console :pager:
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

> :pencil: The `-P` option in the command above tells Blender to execute the python script. Without it, blender will try, unsuccessfully, to open it as a `.blend` file.


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

> :gem: `bpy.ops.mesh.primitive_cylinder_add()` is one of Blender's many operators.
> Blender's UI is actually defined in Python, which means that you can do almost everything in your scripts that you can do in the UI!
> To figure you which operators Blender is running, you can change on of the open *Editors* to an [*Info Editor*](https://docs.blender.org/manual/en/latest/editors/info_editor.html).
>
> Additionally, in `Edit > Preferences > Interface`, check the two boxes `Developer Extras` and `Python Tooltips`.
> If you then hover over a property in the UI, e.g. the Location X of the default cube, Blender will show `bpy.data.objects["Cube"].location[0]` in the tooltip.

### 1.3.2 Adding a material
We have a simple scene, unfortunately it's pretty gray and boring right now.
To make it more interesting we need to add materials to our objects.

Adding a red material to the cylinder can be done using these commands:
```python
red = (1.0, 0.0, 0.0, 1.0) # Blender requires a tuple with 4 values
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

red = (1.0, 0.0, 0.0) # We assume alpha=1 when the tuple has 3 values
ab.add_material(cyclinder, red)
```
> :pencil: Blender by default shows your scene in a fast-to-render "Solid" shading mode.
> To see what your material will look like in the final render, change it to "Rendered" mode.
> (Click the [right-most little sphere in the top-right corner of the viewport](https://docs.blender.org/manual/en/latest/editors/3dview/introduction.html#header-region).)

### 1.3.3 A nicer background
By default, the Blender "world" has a pretty dark gray background.
Let's set that to a brighter color.

```python
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs["Color"].default_value = (0.02, 0.0, 1.0, 1.0)
```
>If you've enable tooltips as described in a previous note, you can find this path in the UI by going to the *Properties Editor*.
> Then click the World icon and then hover over the *Color*.

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
red = (1.0, 0.0, 0.0)
ab.add_material(cylinder, red)

# Making the background brighter
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs["Color"].default_value = (1.0, 0.9, 0.7, 1.0)

# Rendering the scene into an image
bpy.ops.render.render()
```

## 1.5 Enabling Blender's physically-based path tracer: Cycles :high_brightness:
The rendered image should already look ok, but you might notice that some lighting effects are missing.
This is because by default Blender uses its real-time renderer EEVEE.
To tell Blender to use Cycles takes only one line of code:
```python
bpy.context.scene.render.engine = 'CYCLES'
```
Additionally, we want to tell Blender how many rays we want to cast per pixel.
(If we don't, Cycles will render a very high-quality image by default.)
More rays will result in a less noisy image, but takes longer to render:
```python
bpy.context.scene.cycles.samples = 64
```
> :information_source: It's still an open question how much sample count influences sim2real transfer of models trained on synthetic images.
> Using less samples is computationally attractive, but does it degrade model performance?
> Or can we consider render noise a from of harmless (of even helpfull) data augmentation?

To set the image resolution, you can simply:
```python
bpy.context.scene.render.resolution_x = 1024
bpy.context.scene.render.resolution_y = 512
```


Finally, if we set the `render.filepath` and turn on the `write_still` option in the render command:

```python
bpy.context.scene.render.filepath = "red_cylinder.jpg"
bpy.ops.render.render(write_still=True)
```

Blender will save our render to disk (relative to where you started the blender executable).

Your script should look something like this by now:
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

# Telling Blender to render with Cycles, and how many rays we want to cast per pixel
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = 64

bpy.context.scene.render.resolution_x = 1024
bpy.context.scene.render.resolution_y = 512

bpy.context.scene.render.filepath = "red_cylinder.jpg"

# Rendering the scene into an image
bpy.ops.render.render(write_still=True)
```

And this is what the rendered image should look like:
![Red cylinder rendered with Cycles](https://i.imgur.com/G5QuqgC.png)

## 1.6 Saving an object's pose :floppy_disk:
In this final section of the tutorial, we'll save the cylinder's pose to disk.
We can access a Blender object's transform via its `.matrix_world` attribute.
However, for the cylinder this will currently give us the wrong matrix.
The reason is that we've scaled the cylinder, and this scale is also included in this matrix.

To get an object's pose (i.e. rotation and translation, but not scale), the simplest way is to "apply" its scale.
Applying a transform in Blender means "moving" the transform from the object-level, down into the vertex coordinates.

To apply the scale for all objects, we can run:
```python
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
```

Afterwards, the cylinder's `.matrix_world` contains the pose we expect:
```python
pose = cylinder.matrix_world
print(pose)
```
The output in the terminal where you started Blender:
```python
<Matrix 4x4 (1.0000, 0.0000,  0.0000, 0.0000)
            (0.0000, 0.0008, -1.0000, 0.0000)
            (0.0000, 1.0000,  0.0008, 0.5000)
            (0.0000, 0.0000,  0.0000, 1.0000)>
```
As an example, we'll save this pose similar to how the [BOP format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md) does it, where they save:
* `cam_R_m2c` - 3x3 rotation matrix R_m2c (saved row-wise).
* `cam_t_m2c` - 3x1 translation vector t_m2c.
* lengths are in millimeters
* `m2c` is short for "model to camera"

This can be achieve with the following code:
```python
camera = bpy.context.scene.camera

# Find the model to camera transform
# Use use the Drake notation here, X_ab means the transform from frame b to frame a
X_wm = cylinder.matrix_world   # world to model
X_wc = camera.matrix_world     # world to camera
X_mc = X_wm.inverted() @ X_wc  # model to camera

translation, rotation, scale = X_mc.decompose()

import numpy as np
cam_t_m2c = list(1000.0 * translation)
cam_R_m2c = list(1000.0 * np.array(rotation.to_matrix()).flatten())

import json
data = {
    "cam_R_m2c": cam_R_m2c,
    "cam_t_m2c": cam_t_m2c,
}

with open('cylinder_pose.json', 'w') as fp:
    json.dump(data, fp)
```

When running the script now, you should get a `cylinder_pose.json` file with following contents:
```json
{
    "cam_R_m2c": [
        685.9207153320312,
        -324.0134119987488,
        651.5581607818604,
        0.579443818423897,
        895.6385850906372,
        444.78219747543335,
        -727.6760339736938,
        -304.707795381546,
        614.5248413085938
    ],
    "cam_t_m2c": [
        7358.8916015625,
        4452.79296875,
        6929.3388671875
    ]
}
```
The full script that implements this tutorial can be found [here](tutorial_1.py).


## The End :tada:
Congratulations, you've reached the end of the first tutorial and hopefully generated your first piece of synthetic data!
You now also understand the basics of synthetic data generation in Blender.
As you've seen, Blender's Python API is pretty explicit and straightforward.

In the following tutorials, we will teach you how to add assets to your scene, add randomization and work with keypoint
annotations!
