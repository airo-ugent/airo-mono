
# Cameras
This subpackage contains implementations of the camera interface for the cameras we have at AIRO.

Implementations usually require the installation of SDKs, drivers etc. to communicate with the camera. This information can be found in the class docstring for each camera.

Furthermore, there is code for testing the hardware implementations. But since this requires attaching a physical camera, these are 'user tests' which should be done manually by developers/users. Each camera implementation can be run as a script and will execute the relevant tests, providing instructions on what to look out for.