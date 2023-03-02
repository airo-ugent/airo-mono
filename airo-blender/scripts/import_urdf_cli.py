import bpy
from airo_blender.urdf import import_urdf

bpy.ops.object.delete()  # Delete the default cube
# import_urdf("/home/idlab185/urdf-workshop/universal_robots/ros/ur3/ur3.urdf")
# import_urdf("/home/idlab185/urdf-workshop/universal_robots/ros/ur5/ur5.urdf")
# import_urdf("/home/idlab185/urdf-workshop/universal_robots/ros/ur5e/ur5e.urdf")
import_urdf("/home/idlab185/urdf-workshop/universal_robots/ros/ur16e/ur16e.urdf")
# import_urdf("/home/idlab185/urdfpy/tests/data/ur5/ur5.urdf")
