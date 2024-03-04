# airo-mono
Welcome to `airo-mono`! This repository provides ready-to-use Python packages to accelerate the development of robotic manipulation systems.

**Key Motivation:**
  * 🚀 **Accelerate Experimentation:** Reduce the time spent on repetitive coding tasks, enabling faster iteration from research idea to demo on the robots.
  * 😊 **Collaboration:** Promote a shared foundation of well-tested code across the lab, boosting reliability and efficiency and promoting best practices along the way.


Want to learn more about our vision? Check out the in-depth explanation [here](docs/about_this_repo.md)

## Overview

### Packages 📦
The airo-mono repository employs a monorepo structure, offering multiple Python packages, each with a distinct focus:

| Package                                          | Description                                               | Owner          |
| ------------------------------------------------ | --------------------------------------------------------- | -------------- |
| 📷 [`airo-camera-toolkit`](airo-camera-toolkit)   | RGB(D) camera, image, and point cloud processing          | @tlpss         |
| 🏗️ [`airo-dataset-tools`](airo-dataset-tools)     | Creating, loading, and manipulating datasets              | @victorlouisdg |
| 🤖 [`airo-robots`](airo-robots)                   | Simple interfaces for controlling robot arms and grippers | @tlpss         |
| 📐 [`airo-spatial-algebra`](airo-spatial-algebra) | Transforms and SE3 pose conversions                       | @tlpss         |
| 🎮 [`airo-teleop`](airo-teleop)                   | Intuitive teleoperation of robot arms                     | @tlpss         |
| 🛡️ [`airo-typing`](airo-typing)                   | Type definitions and conventions                          | @tlpss         |

**Detailed Information:** Each package contains its own dedicated README outlining:
  - A comprehensive overview of the provided functionality
  - Package-specific installation instructions (if needed)
  - Rationale behind design choices (if applicable)


**Code Ownership:** The designated code owner for each package is your go-to resource for:
  - Understanding features, codebase, and design decisions. 🤔
  - Proposing and contributing new package features. 🌟


**Command Line Functionality:** Some packages offer command-line interfaces (CLI).
Run `package-name --help` for details. Example: `airo-dataset-tools --help`

### Sister repositories 🌱
Repositories that follow the same style as `airo-mono` packages, but are not part of the monorepo (for various reasons):

| Repository                                                     | Description                                     |
| -------------------------------------------------------------- | ----------------------------------------------- |
| 🎥 [`airo-blender`](https://github.com/airo-ugent/airo-blender) | Synthetic data generation with Blender          |
| 🛒 [`airo-models`](https://github.com/airo-ugent/airo-models)   | Collection of robot and object models and URDFs |
| 🐉 `airo-drake`                                                 | Integration with Drake (coming soon)            |
| 🧭 `airo-planner`                                               | Motion planning interfaces (coming soon)        |

### Usage & Philosophy 📖
We believe in *keep simple things simple*. Starting a new project should\* be as simple as:
```bash
pip install airo-camera-toolkit airo-robots
```
And writing a simple script:
```python
from airo_camera_toolkit.cameras.zed.zed2i import Zed2i
from airo_robots.manipulators.hardware.ur_rtde import URrtde
from airo_robots.grippers.hardware.robotiq_2f85_urcap import Robotiq2F85

robot_ip_address = "10.40.0.162"

camera = Zed2i()
robot = URrtde(ip_address=robot_ip_address)
gripper = Robotiq2F85(ip_address=robot_ip_address)

image = camera.get_rgb_image()
grasp_pose = select_grasp_pose(image)  # example: user provides grasp pose
robot.move_linear_to_tcp_pose(grasp_pose).wait()
gripper.close().wait()
```

> \* we are still simplifying the installation process and the long imports

### Projects using `airo-mono` 🎉
Probably the best way to learn what `airo-mono` has to offer, is to look at the projects it powers:

| Project                                                                     | Description                                                                                                 |
| --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| 👕 [`cloth-competition`](https://github.com/Victorlouisdg/cloth-competition) | airo-mono is the backbone of the [ICRA 2024 Cloth Competition](https://airo.ugent.be/cloth_competition/) 🏆! |

## Installation 🔧

### Option 1: Local clone 📥

#### 1.1 Conda method
Make sure you have a version of conda e.g. [miniconda](https://docs.anaconda.com/free/miniconda/) installed.
To make the conda environment creation faster, we recommend configuring the [libmamba solver](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) first.

Then run the following commands:
```bash
git clone https://github.com/airo-ugent/airo-mono.git
cd airo-mono
conda env create -f environment.yaml
```
This will create a conda environment called `airo-mono` with all packages installed. You can activate the environment with `conda activate airo-mono`.

#### 1.2 Pip method
While we prefer using conda, you can also install the packages simply with pip:

```bash
git clone https://github.com/airo-ugent/airo-mono.git
cd airo-mono
pip install -e airo-robots -e airo-dataset-tools -e airo-camera-toolkit
```

### Option 2: Installation from Github 🌐
> ℹ️ This method will be deprecated in the future, as we are moving to PyPI for package distribution. [Direct references](https://peps.python.org/pep-0440/#direct-references) are not allowed in projects that are to be published on PyPI.

You can also install the packages from this repository directly with pip. This is mainly useful if you want to put `airo-mono` packages as dependencies in your `environment.yaml` file:
```yaml
name: my-airo-project
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pip:
      - "airo-typing @ git+https://github.com/airo-ugent/airo-mono@main#subdirectory=airo-typing"
      - "airo-spatial-algebra @ git+https://github.com/airo-ugent/airo-mono@main#subdirectory=airo-spatial-algebra"
      - "airo-camera-toolkit @ git+https://github.com/airo-ugent/airo-mono@main#subdirectory=airo-camera-toolkit"
      - "airo-dataset-tools @ git+https://github.com/airo-ugent/airo-mono@main#subdirectory=airo-dataset-tools"
```

### Option 3: Git submodule 🚇
You can add this repo as a submodule and install the relevant packages afterwards with regular pip commands. This allows to seamlessly make contributions to this repo whilst working on your own project or if you want to pin on a specific version.

In your repo, run:
```bash
git submodule init
git submodule add https://github.com/airo-ugent/airo-mono@<commit>
cd airo-mono
```
You can now add the packages you need to your requirements or environment file, either in development mode or through a regular pip install.
More about submodules can be found [here](https://git-scm.com/book/en/v2/Git-Tools-Submodules). Make sure to install the packages in one pip command such that pip can install them in the appropriate order to deal with internal dependencies.

### Option 4: Installation from PyPI 📦
> 🚧 Not available yet, but coming soon.

Install the packages from PyPI.
```
pip install airo-camera-toolkit airo-dataset-tools airo-robots
```

## Developer guide 🛠️
### Setup
Create and activate the conda environment, then run:
```
pip install -r dev-requirements.txt
pre-commit install
```

### Coding style 👮
We use [pre-commit](https://pre-commit.com/) to automatically enforce coding standards before every commit.

Our [.pre-commit-config.yaml](.pre-commit-config.yaml) file defines the tools and checks we want to run:
  - **Formatting**: Black, isort, and autoflake
  - **Linting**: Flake8

**Typing:** Packages can be typed (optional, but strongly recommended). For this, mypy is used. Note that pre-commit curretnly does not run mypy, so you should run it manually with `mypy <package-dir>`, e.g. `mypy airo-camera-toolkit`.

**Docstrings:** Should be in the [google docstring format](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

### Testing 🧪
  - **Framework:** Pytest for its flexibility.
  - **Organization:** Tests are grouped by package, the CI pipeline runs them in isolation
  - **Minimum:** Always include [at least one test per package](https://github.com/pytest-dev/pytest/issues/2393).
  - **Running Tests:** `make pytest <airo-package-dir>` (includes coverage report).
  - **Hardware Testing:** Cameras and robots have scripts available for manual testing. These scripts provide simple sanity checks to verify connections and functionality.


### Continuous Integration (CI) ⚙️
We use GitHub Actions to run the following checks:

| Workflow                                        | Runs When                                      |
| ----------------------------------------------- | ---------------------------------------------- |
| [pre-commit](.github/workflows/pre-commit.yaml) | Every push                                     |
| [mypy](.github/workflows/mypy.yaml)             | pushes to `main`, PRs and  the `ci-dev` branch |
| [pytest](.github/workflows/pytest.yaml)         | pushes to `main`, PRs and  the `ci-dev` branch |


**Package Test Isolation:** We use [Github Actions matrices](https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs) to run tests for each package in its own environment. This ensures packages correctly declare their dependencies. However, creating these environments adds overhead, so we'll explore caching strategies to optimize runtime as complexity increases.


### Creating a new package ✨

To quickly setup up Python projects you can use this [cookiecutter template](https://github.com/tlpss/research-template). In the future we might create a similar one for `airo-mono` packages. For now here are the steps you have to take:

1. **Create directory structure:**
```
airo-package/
    ├─ airo_package/
    │   └─ code.py
    ├─ test/
    │   └─ test_some_feature.py
    ├─ README.md
    └─ setup.py
```
2. **Integrate with CI:** update the CI workflow matrices to include your package.
3. **Update Documentation:**  add your package to this README
4. **Installation:** add package to the `environment.yaml`
and `scripts/install-airo-mono.sh`.

### Command Line Interfaces 💻
For convenient access to specific functions, we provide command-line interfaces (CLIs). This lets you quickly perform tasks like visualizing COCO datasets or starting hand-eye calibration without the need to write Python scripts to change arguments (e.g., changing a data path or robot IP address).


  - **Framework:** [Click](https://click.palletsprojects.com/en/8.1.x/) for composable CLIs.
  - **Exposure:** We use Setuptools [`console_scripts`](https://setuptools.pypa.io/en/latest/userguide/entry_point.html) to make commands available.
  - **Organization:** User-facing CLI commands belong in a top-level `cli.py` file.
  - **Separation:** Keep CLI code distinct from core functionality for maximum flexibility.
  - **Example:** [`airo_dataset_tools/cli.py`](airo-dataset-tools/airo_dataset_tools/cli.py) and [`airo-dataset-tools/setup.py`](airo-dataset-tools/setup.py).
  - **Developer Focus:** Scripts' `__main__()` functions can still house developer-centric CLIs. Consider moving user-friendly ones to the package CLI.

### Versioning & Releasing 🏷️

As a first step towards PyPI releases of the `airo-mono` packages, we have already started versioning them.
Read more about it in [docs/versioning.md](docs/versioning.md).

### Design choices ✏️
- **Minimalism:** Before coding, explore existing libraries. Less code means easier maintenance.
- **Properties:** Employ Python properties ([@property](https://docs.python.org/3/howto/descriptor.html#properties)) for getters/setters. This enhances user interaction and unlocks powerful code patterns.
- **Logging**: Use [loguru](https://loguru.readthedocs.io/en/stable/) for structured debugging instead of print statements.
- **Output Data:** Favor native datatypes or NumPy arrays for easy compatibility. For more complex data, use dataclasses as in [airo-typing](airo-typing).

#### Management of local dependencies in a Monorepo
> **TODO:** simplify this explanation and move it to the setup or installation section.

An issue with using a monorepo is that you want to have packages declare their local dependencies as well. But before you publish your packages or if you want to test unreleased code (as usually), this creates an issue: where should pip find these local package? Though there exist more advanced package managers such as Poetry, ([more background on package and dependency management in python](https://ealizadeh.com/blog/guide-to-python-env-pkg-dependency-using-conda-poetry/)
) that can handle this, we have opted to stick with pip to keep the barier for new developers lower.


This implies we simply add local dependencies in the setup file as regular dependencies, but we have to make sure pip can find the dependencies when installing the pacakges.There are two options to do so:
1. You make sure that the local dependencies are installed before installing the package, either by running the pip install commands along the dependency tree, or by running all installs in a single pip commamd: `pip install <pkg1>  <pkg2> <pkg3>`
2. you create distributions for the packages upfront and then tell pip where to find them (because they won't be on PyPI, which is where pip searches by default): `pip install --find-link https:// or /path/to/distributions/dir`


Initially, we used a direct link to point to the path of the dependencies, but this created some issues and hence we now use this easier approach. see [#91](https://github.com/airo-ugent/airo-mono/issues/91) for more details.
