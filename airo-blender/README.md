![Synthetic data generation box keypoints example](https://i.imgur.com/ZpH0grX.jpg)

# airo-blender

A Python package on top of the [Blender Python API](https://docs.blender.org/api/current/index.html) to
generate synthetic data for robotic manipulation.

To get started, please see the [tutorials](docs/tutorials) directory after completing the installation. :notebook_with_decorative_cover:


## Installation
> Currently this is a "development" installation, where we assume you have this directory cloned locally. We develop and test this package on Ubuntu.

### Downloading Blender
Download the latest version of [Blender](https://www.blender.org/download/), currently `3.4.1`.
We provide the subdirectory `blender/` as a convenient but optional place to save and extract Blender. See the directory's [README.md](blender/README.md) for the expected directory structure after download.

### Installing the `airo_blender` Python package
Then, run the following commands from the `airo-blender` directory:
```
./blender/blender-3.4.1-linux-x64/3.4/python/bin/python3.10 -m ensurepip
./blender/blender-3.4.1-linux-x64/3.4/python/bin/pip3 install -e .
```
The first command installs `pip` for the Blender Python.
The second command installs the `airo_blender` package into it. The `-e` option


## Philosophy
This package is meant to be as lightweight and opt-in as possible.
We deliberately stick close to the Blender API and keep data explicit.
This means that we for example don't wrap the Blender datatypes with custom Python classes.

Alternatives to this library are:
* [BlenderProc](https://github.com/DLR-RM/BlenderProc)
* [kubric](https://github.com/google-research/kubric)

### Why don't we use the alternatives?
Both libraries provide a lot of great functionality.
However, we choose not to use them because they introduce **too many new concepts** on top of Blender.

For example, to make a light in Kubric (from [helloworld.py](https://github.com/google-research/kubric/blob/main/examples/helloworld.py)), you do:
```python
scene += kb.DirectionalLight(name="sun", position=(-1, -0.5, 3), look_at=(0, 0, 0), intensity=1.5)
```
In blenderproc (from [basic/main.py](https://github.com/DLR-RM/BlenderProc/blob/main/examples/basics/basic/main.py)) you would do:
```python
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([5, -5, 5])
light.set_energy(1000)
```
However, this use case can be handled perfectly fine by the Blender Python API itself e.g:
```python
bpy.ops.object.light_add(type='POINT', radius=1, location=(0, 0, 0))
```
Introducing these additional abstractions on top of Blender creates uncertainty.
Where does the light data live and how does it sync with the Blender scene? Am I still allowed to edit the Blender scene directly, or do I have to do it the "Kubric/Blenderproc way"?
Kubric and Blenderproc both push a very specific workflow.
They try hard to hide/replace Blender, instead of extending it.
As a consequence, these libraries feel very much **"all or nothing"**.
By hiding what is going on, these libraries make it harder to experiment and add new features.

In airo-blender, we prefer explaining the Blender Python API over hiding it.
In the tutorials we show our workflow and the functions we use.
We try to operate on the Blender data as statelessly as possible, with simple functions.
As a result you can easily adopt only the parts you like.
