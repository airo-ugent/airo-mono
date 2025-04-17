#!/usr/bin/env bash

# This script is used to build and publish the AIRO mono packages.
#
# Usage:
# 1. Make sure to update version numbers in ALL setup.py files before running this script.
# 2. Install the dev-requirements.txt file using pip: `pip install -r dev-requirements.txt`.
# 3. Make sure you have access to the PyPI projects and have your PyPI API tokens ready.
# 4. Run this script from airo-mono's root directory as `./scripts/build-airo-mono.sh`.
# 5. Follow the prompts to build and publish the packages.
#
# For step 3, you can create a ~/.pypirc file with the following content:
# [pypi]
# username = __token__
# password = <your PyPI API Token>
#
# Is it the first time that you're using this script? You should use TestPyPI first.

if [[ ! -d "airo-camera-toolkit" || ! -d "airo-dataset-tools" || ! -d "airo-robots" || ! -d "airo-spatial-algebra" || ! -d "airo-teleop" || ! -d "airo-typing" ]]; then
  echo "One or more package directories are missing. Please make sure you are in the root directory of the repository."
  exit 1
fi

# Check if the user has done what is needed before publishing.
read -rp "Did you update the version number in ALL setup.py files? (y/n): " CHOICE
if [[ $CHOICE != "y" ]]; then
  echo "Quitting without building."
  exit 0
fi

# Loop over the packages and run `python -m build` in each package directory to build the package.
for package in airo-camera-toolkit airo-dataset-tools airo-robots airo-spatial-algebra airo-teleop airo-typing; do
  cd "${package}" || exit 1
  echo "Building package: ${package}"
  python -m build
  echo "Done"
  cd ..
done

# Check if the user wishes to continue with publishing.
read -rp "Do you want to publish the packages? (y/n): " CHOICE
if [[ $CHOICE != "y" ]]; then
  echo "Quitting without publishing."
  exit 0
fi

# Loop over the packages and publish them using twine.
for package in airo-camera-toolkit airo-dataset-tools airo-robots airo-spatial-algebra airo-teleop airo-typing; do
  cd "${package}" || exit 1
  echo "Publishing package: ${package}"
  twine upload dist/*
  echo "Done"
  cd ..
done

# Check if the user wants to make a GitHub tag and release.
read -rp "Do you want to make a GitHub tag and release? (y/n): " CHOICE
if [[ $CHOICE == "y" ]]; then
  # Create a new tag and push it to the remote repository.
  read -rp "Enter the tag name (e.g., v1.0.0): " TAG_NAME
  git tag "${TAG_NAME}"
  git push origin "${TAG_NAME}"
  exit 0
fi

