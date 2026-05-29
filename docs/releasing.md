# Releasing

To create a new release there are two steps:
1. version bumping
2. create & publish distribution

## Version bumping
see [versioning](./versioning.md)

## Creating a distribution
Use `scripts/build-airo-mono.sh` to create a distribution and publish it on PyPI.

You can find instruction on how to use this script at the top of the file, also repeated here verbatim:

```
This script is used to build and publish the AIRO mono packages.

Usage:
1. Make sure to update version numbers in ALL package pyproject.toml files before running this script.
2. Set up a dev environment with the `build` and `twine` tools available on PATH:
   - uv:    `uv sync`, then `source .venv/bin/activate`
   - conda: `conda activate airo-mono && pip install -r dev-requirements.txt`
3. Make sure you have access to the PyPI projects and have your PyPI API tokens ready.
4. Run this script from airo-mono's root directory as `./scripts/build-airo-mono.sh`.
5. Follow the prompts to build and publish the packages.

For step 3, you can create a ~/.pypirc file with the following content:
[pypi]
username = __token__
password = <your PyPI API Token>

Is it the first time that you're using this script? You should use TestPyPI first.
```
