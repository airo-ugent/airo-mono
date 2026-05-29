# airo-mono
Welcome to `airo-mono`! This repository provides ready-to-use Python packages to accelerate the development of robotic manipulation systems.

**Key Motivation:**
  * 🚀 **Accelerate Experimentation:** Reduce the time spent on repetitive coding tasks, enabling faster iteration from research idea to demo on the robots.
  * 😊 **Collaboration:** Promote a shared foundation of well-tested code across the lab, boosting reliability and efficiency and promoting best practices along the way.


Want to learn more about our vision? Check out the in-depth explanation [here](docs/about_this_repo.md)

## Table of Contents

- [Overview](#overview)
  - [Packages](#packages-)
  - [Sister repositories](#sister-repositories-)
  - [Usage & Philosophy](#usage--philosophy-)
  - [Getting started](#getting-started-)
  - [Projects using `airo-mono`](#projects-using-airo-mono-)
- [Installation](#installation-)
  - [Option 1: Installation from PyPI](#option-1-installation-from-pypi-)
  - [Option 2: Local clone](#option-2-local-clone-)
    - [2.1 uv method (recommended)](#21-uv-method-recommended)
    - [2.2 Conda method](#22-conda-method)
  - [Option 3: Git submodule](#option-3-git-submodule-)
- [Developer guide](#developer-guide-)
  - [Setup](#setup)
  - [Coding style](#coding-style-)
  - [Testing](#testing-)
  - [Continuous Integration (CI)](#continuous-integration-ci-)
  - [Creating a new package](#creating-a-new-package-)
  - [Command Line Interfaces](#command-line-interfaces-)
  - [Versioning & Releasing](#versioning--releasing-)
  - [Design choices](#design-choices-)
    - [Management of local dependencies in a Monorepo](#management-of-local-dependencies-in-a-monorepo)

## Overview

### Packages 📦
The airo-mono repository employs a monorepo structure, offering multiple Python packages, each with a distinct focus:

| Package                                          | Description                                               | Owner          |
| ------------------------------------------------ | --------------------------------------------------------- | -------------- |
| 📷 [`airo-camera-toolkit`](airo-camera-toolkit)   | RGB(D) camera, image, and point cloud processing          | @m-decoster    |
| 🏗️ [`airo-dataset-tools`](airo-dataset-tools)     | Creating, loading, and manipulating datasets              | @victorlouisdg |
| 🤖 [`airo-robots`](airo-robots)                   | Simple interfaces for controlling robot arms and grippers | @tlpss         |
| 📐 [`airo-spatial-algebra`](airo-spatial-algebra) | Transforms and SE3 pose conversions                       | @tlpss         |
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

| Repository                                                      | Description                            |
|-----------------------------------------------------------------| -------------------------------------- |
| 🎥 [`airo-blender`](https://github.com/airo-ugent/airo-blender) | Synthetic data generation with Blender |
| 🐉 [`airo-drake`](https://github.com/airo-ugent/airo-drake)     | Integration with Drake                 |
| 🔛 [`airo-ipc`](https://github.com/airo-ugent/airo-ipc)         | Inter-process communication library |
| 🛒 [`airo-models`](https://github.com/airo-ugent/airo-models)   | Collection of robot and object models and URDFs |
| 🧭 [`airo-planner`](https://github.com/airo-ugent/airo-planner) | Motion planning interfaces      |
| 🎮 [`airo-teleop`](https://github.com/airo-ugent/airo-teleop/)  | Intuitive teleoperation of robot arms     |
| 🚗 [`airo-tulip`](https://github.com/airo-ugent/airo-tulip)     | Driver for the KELO mobile robot platform |
| 🦾 [`ur-analytic-ik`](https://github.com/Victorlouisdg/ur-analytic-ik)     | Analytic IK calculations for UR manipulators |

### Usage & Philosophy 📖
We believe in *keep simple things simple*. Starting a new project is as simple as:
```bash
pip install airo-camera-toolkit airo-robots
```
And writing a simple script:
```python
from airo_camera_toolkit.cameras.zed.zed import Zed
from airo_robots.manipulators.hardware.ur_rtde import URrtde
from airo_robots.grippers.hardware.robotiq_2f85_urcap import Robotiq2F85

robot_ip_address = "10.40.0.162"

camera = Zed()
robot = URrtde(ip_address=robot_ip_address)
gripper = Robotiq2F85(ip_address=robot_ip_address)

camera.grab_images()
image = camera.retrieve_rgb_image()
grasp_pose = select_grasp_pose(image)  # example: user provides grasp pose
robot.move_linear_to_tcp_pose(grasp_pose).wait()
gripper.close().wait()
```

### Getting started 🚀
To get started with `airo-mono`, check out our [getting started guide](docs/getting_started.md) which provides examples and explanations of the main functionalities provided by the packages.

### Projects using `airo-mono` 🎉
Probably the best way to learn what `airo-mono` has to offer, is to look at the projects it powers:

| Project                                                                     | Description                                                                                                 |
| --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| 🧩 [ITF World 2026](https://airo.ugent.be/itfworld/) | airo-mono powered our demos at ITF World 2026 |
| 🍹 [ITF World 2025](https://airo.ugent.be/news/202505_itf/) | airo-mono powered our demos at ITF World 2025 |
| 👕 [`cloth-competition`](https://github.com/Victorlouisdg/cloth-competition) | airo-mono is the backbone of the [ICRA 2024 Cloth Competition](https://airo.ugent.be/cloth_competition/) 🏆! |
| 🧪 [ITF World 2024](https://airo.ugent.be/news/itf2024/) | airo-mono powered our demo at ITF World 2024 |

## Installation 🔧

### Option 1: Installation from PyPI 📦

Install the packages from PyPI.
```
pip install airo-camera-toolkit airo-dataset-tools airo-robots
```

### Option 2: Local clone 📥

We support two ways to set up a local development environment. **uv is recommended** but conda is also supported.

#### 2.1 uv method (recommended)
Install [uv](https://docs.astral.sh/uv/) (see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)), then:

```bash
git clone https://github.com/airo-ugent/airo-mono.git
cd airo-mono
uv sync
```

This creates a `.venv` with all packages installed in editable mode. Activate it with `source .venv/bin/activate`, or prefix commands with `uv run` (e.g. `uv run pytest`).

To pull in optional hardware extras: `uv sync --extra realsense --extra recording` (see `[project.optional-dependencies]` in `airo-camera-toolkit/pyproject.toml`).

#### 2.2 Conda method
Make sure you have a version of conda e.g. [miniconda](https://docs.anaconda.com/free/miniconda/) installed. To make the conda environment creation faster, we recommend configuring the [libmamba solver](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) first.

Then run:
```bash
git clone https://github.com/airo-ugent/airo-mono.git
cd airo-mono
conda env create -f environment.yaml
conda activate airo-mono
pip install -r dev-requirements.txt   # if you want the dev tools
```
This creates a conda environment called `airo-mono` with all packages installed in editable mode. Note that `environment.yaml` and `dev-requirements.txt` are kept in sync with the uv `[project]` and `[dependency-groups] dev` tables in `pyproject.toml`.

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

## Developer guide 🛠️
### Setup
After setting up your environment (see [Option 2: Local clone](#option-2-local-clone-)), install the pre-commit hooks:
```
pre-commit install   # or: uv run pre-commit install
```
Dev tools (mypy, pytest, pre-commit, build, twine, ...) are declared in the `dev` dependency group in [`pyproject.toml`](pyproject.toml) (installed by `uv sync`) and mirrored in `dev-requirements.txt` (installed by `pip install -r dev-requirements.txt` for conda users).

### Coding style 👮
We use [pre-commit](https://pre-commit.com/) to automatically enforce coding standards before every commit.

Our [.pre-commit-config.yaml](.pre-commit-config.yaml) file defines the tools and checks we want to run:
  - **Formatting**: Black, isort, and autoflake
  - **Linting**: Flake8

**Typing:** Packages can be typed (optional, but strongly recommended). For this, mypy is used. Note that pre-commit currently does not run mypy, so you should run it manually with `mypy <package-dir>`, e.g. `mypy airo-camera-toolkit`.

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
4. **Installation:** add the package to the `[tool.uv.workspace] members` and `[tool.uv.sources]` tables in the root [`pyproject.toml`](pyproject.toml), to the `pip:` list in `environment.yaml`, and to `scripts/build-airo-mono.sh`.

### Command Line Interfaces 💻
For convenient access to specific functions, we provide command-line interfaces (CLIs). This lets you quickly perform tasks like visualizing COCO datasets or starting hand-eye calibration without the need to write Python scripts to change arguments (e.g., changing a data path or robot IP address).


  - **Framework:** [Click](https://click.palletsprojects.com/en/8.1.x/) for composable CLIs.
  - **Exposure:** We use Setuptools [`console_scripts`](https://setuptools.pypa.io/en/latest/userguide/entry_point.html) to make commands available.
  - **Organization:** User-facing CLI commands belong in a top-level `cli.py` file.
  - **Separation:** Keep CLI code distinct from core functionality for maximum flexibility.
  - **Example:** [`airo_dataset_tools/cli.py`](airo-dataset-tools/airo_dataset_tools/cli.py) and [`airo-dataset-tools/setup.py`](airo-dataset-tools/setup.py).
  - **Developer Focus:** Scripts' `__main__()` functions can still house developer-centric CLIs. Consider moving user-friendly ones to the package CLI.

### Versioning & Releasing 🏷️

Read more about it versioning in [docs/versioning.md](docs/versioning.md) and about releasing in [docs/releasing.md](docs/releasing.md).

### Design choices ✏️
- **Minimalism:** Before coding, explore existing libraries. Less code means easier maintenance.
- **Properties:** Employ Python properties ([@property](https://docs.python.org/3/howto/descriptor.html#properties)) for getters/setters. This enhances user interaction and unlocks powerful code patterns.
- **Logging**: Use [loguru](https://loguru.readthedocs.io/en/stable/) for structured debugging instead of print statements.
- **Output Data:** Favor native datatypes or NumPy arrays for easy compatibility. For more complex data, use dataclasses as in [airo-typing](airo-typing).
