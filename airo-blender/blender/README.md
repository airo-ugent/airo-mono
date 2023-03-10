# Standard directory structure

This directory serves as a unified location to download and extract Blender.
Note however that you are completely free to deviate from this convention.

After extraction, your folder structure should look something like this:

```
airo-mono
└── airo-blender
    └── blender
        └── blender-3.4.1-linux-64
            └── blender                   # The actual Blender executable
            └── 3.4
                └── python
                    └── bin
                    |   └── python3.10    # Blender's Python interpreter
                    └── lib
                        └── python3.10/   # Where Python packages are installed

```