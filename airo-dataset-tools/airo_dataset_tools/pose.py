from pydantic import BaseModel


class Position(BaseModel):
    """Position in 3D space, all units are in meters."""

    x: float
    y: float
    z: float


class EulerAngles(BaseModel):
    roll: float
    pitch: float
    yaw: float


class Pose(BaseModel):
    position_in_meters: Position
    rotation_euler_xyz_in_radians: EulerAngles
