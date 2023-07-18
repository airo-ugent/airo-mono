#!/bin/bash

package_names=(
    "airo-camera-toolkit"
    "airo-dataset-tools"
    "airo-robots"
    "airo-spatial-algebra"
    "airo-teleop"
    "airo-typing"
)

# Base URL for the Git repository
base_url="https://github.com/airo-ugent/airo-mono@main#subdirectory="

# Loop through package names and execute pip install command
for package_name in "${package_names[@]}"
do
    cmd="python -m pip install '${package_name}[external] @ git+$base_url$package_name'"
    eval $cmd
    echo "Installed $package_name."
done