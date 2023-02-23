# Tutorial 3 - Creating a COCO Towel keypoints datasets :dvd:
In the last tutorial we saw how to build up a Blender in scene.
In this tutorial we build on that knowledge to build a more complex scene, but we won't elaborate too much.
The focus here is on explaining how we can use these scenes to generate entire datasets.

In our cloth folding research, we rely on the detection of keypoints on the border of the cloth.
In this tutorial we will show how we can create a COCO-format dataset of towel corner keypoints.

## 3.1 Setting up a random towel scene :game_die:
Synthetic data generation is all about defining **rules**.
In this tutorial we're generating data for a household robot that has to fold towels.

The rules we will be encoding in the generation of our scene are:
* Towel are perfectly flat and rectangular
* Towel dimensions are between 30 cm and 1 m.
* Towel lie on tables.
* Towel have a uniform, random, (RGB) color.
* The robot will look down at the towel from a small height.

These rules are not completely representative of the real world scenes in which the robot will operate.
However, they will serve as a useful starting for the synthetic data generation.
We can then improve our data generation by evaluating the sim2real performance of models trained on it.
<!-- > One can rightfully ask how much effort you should put into these rules. -->

You can take a look in [`tutorial_3.py`](./tutorial_3.py) to see how we translated the above rules into Python.
You might wonder why we build the towel geometry ourselves from vertices, edges and faces, instead of scaling a Plane.
The reason is that this will generalize to more complex shapes such a shirts and pants.

## 3.2 The COCO dataset format for keypoints
There's no real standard format for exchange of keypoints yet.
The most popular format I'm aware of is [COCO](https://cocodataset.org/#format-data), so that's what we will be implementing here.

