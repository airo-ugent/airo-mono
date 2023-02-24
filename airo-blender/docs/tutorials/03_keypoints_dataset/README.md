Tutorial 3 - Creating a COCO Towel keypoints datasets :dvd:
===========================================================
In the last tutorial we saw how to build up a Blender in scene.
In this tutorial we build on that knowledge to build a more complex scene, but we won't elaborate too much.
The focus here is on explaining how we can use these scenes to generate entire datasets.

In our cloth folding research, we rely on the detection of keypoints on the border of the cloth.
In this tutorial we will show how we can create a COCO-format dataset of towel corner keypoints.

3.1 Setting up a random towel scene :game_die:
----------------------------------------------
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

3.2 The COCO dataset format for keypoints
-----------------------------------------
There's no real standard format for exchange of keypoints yet.
The most popular format I'm aware of is [COCO](https://cocodataset.org/#format-data), so that's what we will be implementing here.
<!-- COCO keypoints are mostly used for human pose estimation. -->

Comply to a dataset format can be buggy and tedious work.
For that reason, we implemented a [pydantic](https://docs.pydantic.dev/) parser for COCO.
You can find it in `airo_blender/coco_parser.py`.
An example is the class defined for images:
```python
class CocoImage(BaseModel):
    license: Optional[LicenseID]
    file_name: Relativepath
    height: int
    width: int
    id: ImageID
```
In our data generation script we can now import this class and easily fill it in:
```python
from airo_blender.coco_parser import CocoImage

coco_image = CocoImage(file_name=relative_image_path, height=image_height, width=image_width, id=random_seed)
print(coco_image)
```
This will print:
```
license=None file_name='00000000.jpg' height=512 width=512 id=0
```

### 3.2.1 Keypoints

Besides the `CocoImage` class, we similarly have a `CocoKeypointAnnotation`:
```python
class CocoKeypointAnnotation(BaseModel):
    id: AnnotationID
    image_id: ImageID
    category_id: CategoryID
    segmentation: Segmentation
    area: Area
    bbox: BoundingBox
    iscrowd: IsCrowd

    num_keypoints: Optional[int]
    keypoints: List[float]
```

## 3.3 Usage
In the `03_keypoints_dataset` directory:
```
blender -P tutorial_3.py -- --seed 42
```
This will generate a folder `00000042` which contains all the output for a single sample, e.g. the render, segmentation mask, coco annotations etc.

To generate a dataset of 50 images:
```
mkdir dataset0
cd dataset0
blender -b -P ../dataset.py -- --dataset_size 50
```
You can then visualize this dataset using FiftyOne:
```
python fiftyone_coco.py dataset0/annotations.json
```