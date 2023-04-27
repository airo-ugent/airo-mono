# CVAT for labeling data
This module contains code and documentation for the workflow we use to label image data using CVAT.

why CVAT? There are a number of other tools available. We want the tool to be open-source, free to use, easy to host, and support as many formats as possible. CVAT has all these features, but the same goes for [LabelStudio](https://labelstud.io/) which would be a valid alternative.

## Usage
### connecting to CVAT on local device (AIRO-only)
We run a CVAT server on the `paard` workstation. If you have an account and are connected to the ELIS-network, you can forward CVAT's port to your device using `ssh -L 127.0.0.1:8080:paard:8080 <username>@paard`. You can now browse in chrome  to `localhost:8080` and log in to CVAT.
### adding users
why separate users: makes it easier to assign stuff and to see who did what. See the [CVAT docs]() for how to create users.

### creating annotations
The dataset structure can be nested, but it can only contain image files. If there are other files, you will have to create a separate 'view' that mirrors the structure (so that all paths are still the same) but only contains the image files that need to be labeled.


#### Project Setup
#### Labeling

### Converting cvat annotations to COCO annotations

## CVAT installation
- clone
- add local share
- docker-compose up -d
- create a superuser


