# airo-mono
The goals of this repo are
- first and foremost to
    - facilitate our research and development in robotic perception and control
    - facilitate the creation of demos/applications à la Folding competition @ IROS22
by  provide either wrappers to existing libraries or implementions for common functionalities and operations.
- to do this in an opt-in fashion. Users should be able to choose which components to use and which not on the one hand and which frameworks/tools/... to use this repo with, e.g. to allow for interchanging different simulators/ hardware on the one hand and different 'decision making frameworks' on the other hand. This is a.o. achieved by creating base classes / interfaces for abstracting (simulated) hardware.
- to make it easy to use this repo. We want to avoid a steep learning curve as this repo is also meant te be used by people without too much experience in robotic manipulation.
- to avoid reinventing the wheel while doing the above by levering existing libraries if possible. Possible means that existing alternatives have the desired features (obviously) and have an acceptable ease-of-use, level of documentation, robustness and code quality.

Particularly important to note is that , in the spirit of this last item, this repo does not offer advanced robotics features such as optimization-based motion planning, collision checking... If you need such things, you have to use an existing framework such as Moveit, OMPL, Drake,... However these frameworks often come with a steeper learning curve due to their genericity, which is why we offer some basic functionalities in 'barebone python'. Simple things should be simple. You could however still use certain functionalities (such as converting pixels to SE3 poses), then use a framework like Drake to generate a collision-free trajectory to the robot and then interface the robot to drake using our drivers.

## Functionality
below is a short overview of the packages in this repo. Each package has a dedicated readme file with more information.
```
airo-typing             # common typedefs and conventions (e.g. extrinsics matrix = camera IN world)
airo-spatial-algebra    # code for working with SE3 poses
airo-camera-toolkit     # code for working with RGB(D) Images and cameras
airo-robots             # code for working with robot arms and grippers
airo-teleop             # code for teleoperating robot arms
```
For realtime visualisation of robotics data  we prefer [rerun.io](https://www.rerun.io/) over the manually hacking something together with opencv/pyqt/... No wrappers are needed here, just pip install the SDK.

## Installation
There are a number of ways to install packages from this repo, based on whether you want to use them or you also want to make changes to them:

if you just want to use a package for a downstream application you can install it with pip like this: `python -m pip install ' <pkg-name>[external] @ git+https://github.com/airo-ugent/airo-core@<branch/tag>#subdirectory=<package-dir>'`. Note the [external] specification, this is a quick hack to allow for working with a monorepo while using pip is package manager, read more [here](#developer-guide/). There will now be a `src/` folder in your project where pip has downloaded the repo and from where the package is installed, but you can ignore this as it will be automatically excluded from source control.

Alternatively you can add this repo as a submodule and install the relevant packages afterwards with regular pip commands. This might be useful for as long as this repo is making fast/breaking changes without good version management.

If you want to make changes, you should probably clone this repo first
and then install all relevant packages in [editable](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) mode, so that any change you make is immediately 'visible' to your python interpreter. You can simply run `conda env create -f environment.yaml`, which does this for you (and also installs some binaries for convenience).

## Developer guide
To set up your development environment, run:
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

### Design choices
- attributes that require complex getter/setter behaviour should use python [properties](https://realpython.com/python-property/)
- the easiest code to maintain is no code so thorougly consider if the functionality you want does not already have a good implementation and could be imported with a reasonable dependency cost.
- it is strongly encouraged to create CLI interfaces using [click](https://click.palletsprojects.com/en/8.1.x/)
- it is strongly advised to use logging with [loguru](https://loguru.readthedocs.io/en/stable/), it is prohibited on the other hand to use an overload of print statements.

## summary of 'why make this repo?'

The idea is that by having the hardware interface, that we do not have to choose permanently between Moveit or Drake as planning framework and ROS vs. ur_rtde for communication with the robot, or whether to communicate sensor data over ROS or not.

The camera class for example, could easily be wrapped in a ROS node later after all functional code has already been written in a class that inherits from the base RGBD base class. In a simulator you can also easily use the same base class (with partial implementation of the interface), which allows for using the same API and possible for easily swapping out real and simulated hardware.

If we ever were to use Moveit, the IK of the real robot could for example send a request to moveit using the ros_bridge. Or you could directly interact with Moveit w/o using the AIRO core interface, so you are also not limited by it?

In short, it should provide a clear interface that can be implement in any HW/simulation and used to execute plans devised in any motion planning framework. If desired, you could bypass the interface.


inspiration sources:
- Berkeley Autolab: [core](https://github.com/BerkeleyAutomation/autolab_core) [ur python](https://github.com/BerkeleyAutomation/ur5py)
- CMU [frankapy](https://github.com/iamlab-cmu/frankapy) [paper](https://arxiv.org/abs/2011.02398?s=09)


Why not use ROS for everyting?
ROS is a great tool, but hard to create clean code and so generic that it makes easy things hard... Furthermore ROS2 still But this is not a provably right choice, it is driven by (limited) personal experience, the desire for a flat learning curve to serve short-lived projects such as master thesis etc.