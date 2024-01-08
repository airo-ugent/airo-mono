#!/bin/bash
"""
This is a convenience installation script for the airo mono repo.
It installs all packages in the airo mono repo from a given branch/commit, if nothing is provided, it defaults to main.
This script can take up to a few minutes to complete.

usage:
run this command from the desired python environment:
bash  <path-to-script>/install-airo-mono.sh [branch/commit ID]

e.g. bash install-airo-mono.sh main to install the main.
"""

# optional argument branch name
branch=${1:-main}


echo "Installing airo-mono from branch/commit $branch."

package_names=(
    "airo-typing"
    "airo-spatial-algebra"
    "airo-dataset-tools"
    "airo-robots"
    "airo-teleop"
    "airo-camera-toolkit"

)

# Base URL for the Git repository
base_url="https://github.com/airo-ugent/airo-mono@${branch}#subdirectory="

# Loop through package names and execute pip install command
for package_name in "${package_names[@]}"
do
    cmd="python -m pip install '${package_name} @ git+${base_url}${package_name}'"
    echo $cmd
    eval $cmd
    echo "Installed $package_name."
done

echo "Finished installing airo-mono from branch/commit $branch."
exit 0