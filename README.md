# airo-mono
This repository contains code to facilitate our research in and development of robotic manipulation systems.

## Overview
below is a short overview of the packages in this repo.

| Package | Description| owner |
|-------|-------|--------|
|`airo-blender` |code for (procedural) synthetic data generation | @Victorlouisdg |
| `airo-camera-toolkit`|code for working with RGB(D) cameras, images and pointclouds |@tlpss|
|`airo-dataset-tools`| code for creating, loading and working with datasets| @Victorlouisdg|
| `airo-robots`| minimal interfaces for interacting with the controllers of robot arms and grippers| @tlpss|
| `airo-spatial-math`|code for working with SE3 poses |@tlpss|
|`airo-teleop`| code for teleoperating robot arms |@tlpss|
| `airo-typing`  |common type definitions and conventions (e.g. extrinsics matrix = camera IN world) | @tlpss       |

Each package has a dedicated readme file that contains more information.
Furthermore, each package has a 'code owner'. This is the go-to person if you:
- have questions about what is supported or about the code in general
- want to know more about why something is implemented in a particular way
- want to add new functionality to the package


#TODO: move to camera-toolkit package
For realtime visualisation of robotics data we prefer [rerun.io](https://www.rerun.io/) over  manually hacking something together with opencv/pyqt/... No wrappers are needed here, just pip install the SDK. An example notebook to get to know this tool and its potential can be found [here](airo-camera-toolkit/airo_camera_toolkit/docs/rerun-zed-example.ipynb).

## Installation
There are a number of ways to install packages from this repo.

if you just want to use a package for a downstream application you can install it with pip like this: `python -m pip install ' <pkg-name>[external] @ git+https://github.com/airo-ugent/airo-core@<branch/tag>#subdirectory=<package-dir>'`. Note the [external] specification, this is a quick hack to allow for working with a monorepo while using pip is package manager, read more [here](#developer-guide/). There will now be a `src/` folder in your project where pip has downloaded the repo and from where the package is installed, but you can ignore this as it will be automatically excluded from source control. You can (and should?) lock the pip install to a specific commit in your dependency manager (pip/conda/...)

Alternatively you can add this repo as a submodule and install the relevant packages afterwards with regular pip commands. This might be useful for as long as this repo is making fast/breaking changes without good version management, as you can lock the submodule on a specific commit.

If you want to make changes, you should probably clone this repo first (optionally using git submodules)
and then install all relevant packages in [editable](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) mode, so that any change you make is immediately 'visible' to your python interpreter. If you make make a standalone clone of this repo, you can simply run `conda env create -f environment.yaml`, which does this for you (and also installs some binaries for convenience).

## Developer guide

To set up your development environment after cloning this repo, run:
```
conda env create -f environment.yaml
conda activate airo-mono
pip install -r dev-requirements.txt
pre-commit install
```

### Coding style and testing
Formatting happens with black (code style), isort (sort imports and autoflake (remove unused imports and variables). Flake8 is used as linter. These are bundled with [pre-commit](https://pre-commit.com/) as configured in the `.pre-commit-config.yaml` file.

Packages can be typed (optional, but strongly recommended). For this, mypy is used.

Docstrings should be formatted in the [google docstring format](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

Testing with pytest. Unittests should be grouped per package, as the CI pipeline will run them for each package in isolation. Also note that there should always be at least one test, since pytest will otherwise [throw an error](https://github.com/pytest-dev/pytest/issues/2393).

### CI
we use github actions to do the following checks on each PR, push to master (and push to a branch called CI for doing development on the CI pipeline itself)

- formatting check
- mypy check
- pytest unittests

The tests are executed for each package in isolation using [github actions Matrices](https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs), which means that only that package and its dependencies are installed in an environment to make sure each package correctly declares its dependencies. The downside is that this has some overhead in creating the environments, so we should probably look into caching them once the runtime becomes longer.

### Management of (local) dependencies
[more background on package and dependency management in python](https://ealizadeh.com/blog/guide-to-python-env-pkg-dependency-using-conda-poetry/)

An issue with using a monorepo is that you want to have packages declare their local dependencies as well, while being able to install all packages in editable mode so that all changes that you make are immediately reflected (so that you can in fact edit multiple packages at the same time).

Pip makes this very hard as it by default reinstalls any local package (so it will get reinstalled even if it already was in your environment!) and since you cannot specify editable dependencies in the distutils setup.py.

So if you install a package in editable mode and then install another in editable mode that has the first as a dependency, pip would reinstall that first package in 'normal' mode and your changes to that package would no longer be immediately reflected.

Among others, this is why [many people](https://medium.com/opendoor-labs/our-python-monorepo-d34028f2b6fa) use [Poetry](https://python-poetry.org/docs/basic-usage/) as package manager for python monorepos. Poetry config files can be handled to fix this issue.

However, for now we want to avoid to add this complexity for new contributors. Therefore we use a hacky approach to add the local dependencies through an `extras_require` of the setup.py. Installing the package with its internal dependencies should hence be done with `pip install package[external]`.
### Creating a new package
#TODO
- nested structure
- create setup.py
    - handle internal dependencies with extra_requires {'external'}
- add package name to matrix of CI flows
- add package to top-level readme [here](#functionality)

### Design choices
- attributes that require complex getter/setter behaviour should use python [properties](https://realpython.com/python-property/)
- the easiest code to maintain is no code so thorougly consider if the functionality you want does not already have a good implementation and could be imported with a reasonable dependency cost.
- it is strongly encouraged to create CLI interfaces using [click](https://click.palletsprojects.com/en/8.1.x/)
- it is strongly advised to use logging with [loguru](https://loguru.readthedocs.io/en/stable/), it is prohibited on the other hand to use an overload of print statements.