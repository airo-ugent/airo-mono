# airo-mono
This repository contains python packages for common operations to facilitate our research on and development of robotic manipulation systems.
Instead of reimplementing the same functionalities over and over, this repo provides ready-to-use implementations and aims to leverage experience by updating the implementations with best practices along the way.

## Overview
The repository is structured as a monorepo (hence the name) with multiple python packages.
Below is a short overview of the packages:

| Package | Description| owner |
|-------|-------|--------|
|`airo-blender` |code for (procedural) synthetic data generation | @Victorlouisdg |
| `airo-camera-toolkit`|code for working with RGB(D) cameras, images and pointclouds |@tlpss|
|`airo-dataset-tools`| code for creating, loading and working with datasets| @Victorlouisdg|
| `airo-robots`| minimal interfaces for interacting with the controllers of robot arms and grippers| @tlpss|
| `airo-spatial-algebra`|code for working with SE3 poses |@tlpss|
|`airo-teleop`| code for teleoperating robot arms |@tlpss|
| `airo-typing`  |common type definitions and conventions (e.g. extrinsics matrix = camera IN world) | @tlpss       |

Each package has a dedicated readme file that contains more information.
Furthermore, each package has a 'code owner'. This is the go-to person if you:
- have questions about what is supported or about the code in general
- want to know more about why something is implemented in a particular way
- want to add new functionality to the package

# Installation
There are a number of ways to install packages from this repo.

**directly from github**
if you just want to use a package for a downstream application you can install it with pip like this: `python -m pip install ' <pkg-name>[external] @ git+https://github.com/airo-ugent/airo-core@<branch/tag>#subdirectory=<package-dir>'`. Note the [external] specification, this is a quick hack to allow for working with a monorepo while using pip is package manager, read more [here](#developer-guide/). There will now be a `src/` folder in your project where pip has downloaded the repo and from where the package is installed, but you can ignore this as it will be automatically excluded from source control. You can (and should?) lock the pip install to a specific commit in your dependency manager (pip/conda/...)

**git submodule**
Alternatively you can add this repo as a submodule and install the relevant packages afterwards with regular pip commands. This might be useful for as long as this repo is making fast/breaking changes without good version management, as you can lock the submodule on a specific commit.

**editable install**

If you want to make changes, you should probably clone this repo first (optionally using git submodules)
and then install all relevant packages in [editable](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) mode, so that any change you make is immediately 'visible' to your python interpreter. If you make make a standalone clone of this repo, you can simply run `conda env create -f environment.yaml`, which does this for you (and also installs some binaries for convenience).

# Developer guide
### setting up local environment
To set up your development environment after cloning this repo, run:
```
conda env create -f environment.yaml
conda activate airo-mono
pip install -r dev-requirements.txt
pre-commit install
```

### Coding style
Formatting is done with black (code style), isort (sort imports and autoflake (remove unused imports and variables). Flake8 is used as linter. These are bundled with [pre-commit](https://pre-commit.com/) as configured in the `.pre-commit-config.yaml` file. You can manually run pre-commit with `pre-commit run -a`.

Packages can be typed (optional, but strongly recommended). For this, mypy is used. To run mypy on a package: `mypy <package-outer-dir>`.

Docstrings should be formatted in the [google docstring format](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

### Testing
Testing is done with pytest (as this is more flexible than unittest). Tests should be grouped per package, as the CI pipeline will run them for each package in isolation. Also note that there should always be at least one test, since pytest will otherwise [throw an error](https://github.com/pytest-dev/pytest/issues/2393).

You can manually run the tests for a package with `make pytest <airo-package-dir>`. This will also provide a coverage report.

Testing hardware interfaces etc with unittests is rather hard, but we expect all other code to have tests. Testing not only formalises the ad-hoc playing that you usually do whilst developing, it also enables anyone to later on refactor the code and quickly check if this did not break anything.

For hardware-related code, we expect 'user tests' to be avaiable. By this we mean a script that clearly states what should happen with the hardware, so that someone can connect to the hardware and quickly see if everything is working as expected.
### CI
we use github actions to do the following checks on each PR, push to master (and push to a branch called CI for doing development on the CI pipeline itself)

- formatting check
- mypy static type checking
- pytest unittests.

The tests are executed for each package in isolation using [github actions Matrices](https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs), which means that only that package and its dependencies are installed in an environment to make sure each package correctly declares its dependencies. The downside is that this has some overhead in creating the environments, so we should probably look into caching them once the runtime becomes longer.

We test on python 3.8 (default on ubuntu 20.04), 3.9 and 3.10 (default on Ubuntu 22.04). It is important to test these versions explicitly, e.g. typing with `list` instead of `typing.List` is not allowed in 3.8, but it is in >=3.9.

### Management of (local) dependencies
[more background on package and dependency management in python](https://ealizadeh.com/blog/guide-to-python-env-pkg-dependency-using-conda-poetry/)

An issue with using a monorepo is that you want to have packages declare their local dependencies as well, while being able to install all packages in editable mode so that all changes that you make are immediately reflected (so that you can in fact edit multiple packages at the same time).

Pip makes this very hard as it by default reinstalls any local package (so it will get reinstalled even if it already was in your environment!) and since you cannot specify editable dependencies in the distutils setup.py.

So if you install a package in editable mode and then install another in editable mode that has the first as a dependency, pip would reinstall that first package in 'normal' mode and your changes to that package would no longer be immediately reflected.

Among others, this is why [many people](https://medium.com/opendoor-labs/our-python-monorepo-d34028f2b6fa) use [Poetry](https://python-poetry.org/docs/basic-usage/) as package manager for python monorepos. Poetry config files can be handled to fix this issue.

However, for now we want to avoid adding this complexity for new contributors. Therefore we use an ad-hoc solution by specifying the local dependencies through an `extras_require` of the setup.py. Installing the package with its internal dependencies should hence be done with `pip install package[external]`, whereas an editable installation can still happen with `pip install -e package`. It's now on you to install the local dependencies in editable mode as well.
### Creating a new package
Creating a new package is kind of a hassle atm, in the future we might add a coockiecutter template for it. For now here are the steps you have to take:
- create the nested structure
```
<airo-package>/
    <airo_package>/
        code.py
    test/
        testx.py
    README.md
    setup.py
```
- create the minimal setup.py
    - handle internal dependencies with extra_requires {'external'}
- add package name to matrix of CI flows
- add package to top-level readme [here](#overview)

### Design choices
- class attributes that require getter/setter should use python [properties](https://realpython.com/python-property/). This is not only more pythonic (for end-user engagement with the attribute), but more importantly it enables a whole bunch of better code patterns as you can override a decorator, but not a plain attribute.
- the easiest code to maintain is no code at all. So thorougly consider if the functionality you want does not already have a good implementation and could be imported with a reasonable dependency cost.
- it is strongly encouraged to use [click](https://click.palletsprojects.com/en/8.1.x/) for command line interfaces.
- it is strongly advised to use logging instead of print statements. [loguru](https://loguru.readthedocs.io/en/stable/) is the recommended logging tool.
- Use python dataclasses for configuration, instead of having a ton of argument in the build/init functions of your classes.
- The output of operations should preferably be in a `common` format. E.g. if you have a method that creates a pointcloud, we prefer that method to return a numpy array instead of your own pointcloud class, that internally contains such an array. You can of course use such classes internally, but we prefer that the end user gets 'common formats', because he/she is most likely to immediately extract the numpy arrar out of your custom data class anyways.