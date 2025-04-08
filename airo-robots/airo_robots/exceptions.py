class TrajectoryConstraintViolationException(Exception):
    """Exception raised when a trajectory violates constraints."""

    def __init__(self, message):
        super().__init__(message)


class InvalidTrajectoryException(Exception):
    """Exception raised when a trajectory is invalid."""

    def __init__(self, message):
        super().__init__(message)


class RobotSafetyViolationException(Exception):
    """Exception raised when a robot violates safety constraints."""

    def __init__(self, message):
        super().__init__(message)


class RobotConfigurationException(Exception):
    """Exception raised when a robot configuration is invalid."""

    def __init__(self, message):
        super().__init__(message)
