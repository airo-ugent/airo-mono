### Adding Blender to `.bashrc`

I recommend adding the path to the blender directory to the system PATH.

You can do this by adding this line anywhere in your `.bashrc` file in your home directory:
```bash
export PATH="$PATH:/home/$USER/airo-mono/airo-blender/blender/blender-3.4.1-linux-x64"
```
After reopening your terminal, you should be able to open Blender simply by running `blender`.