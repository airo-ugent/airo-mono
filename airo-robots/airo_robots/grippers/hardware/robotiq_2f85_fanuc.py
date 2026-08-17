"""Parallel-position-gripper implementation for a Robotiq 2F-85 driven by a FANUC controller.

Only a few discrete openings and force classes are reachable, and nothing at all is readable: see
[fanuc_robotiq.md](fanuc_robotiq.md) for the mechanism, the buckets and what that costs.
"""

import importlib
from types import ModuleType
from typing import Any, Dict, Optional

import numpy as np
from airo_robots.awaitable_action import AwaitableAction
from airo_robots.grippers.parallel_position_gripper import ParallelPositionGripper, ParallelPositionGripperSpecs
from loguru import logger

# The bucket values the GRIPDISP dispatcher on the controller understands, mirrored from
# airo_fanuc.gripper rather than imported so that this module imports without the driver installed.
# _assert_protocol_is_robotiq_2f85 cross-checks them against the driver's own protocol object, so the
# mirror cannot silently drift.
OPEN_FULL = 0
OPEN_MID = 1
OPEN_NARROW = 2

FORCE_LIGHT = 0
FORCE_MEDIUM = 1
FORCE_HARD = 2

# Nominal finger opening [m] of each open bucket, with the default fingertips. Nothing reads these
# back, so they select a bucket rather than measure the fingers.
OPEN_WIDTHS = {OPEN_FULL: 0.085, OPEN_MID: 0.060, OPEN_NARROW: 0.035}

# Nominal finger opening [m] each close force class ends at with nothing between the fingers.
CLOSE_WIDTHS = {FORCE_LIGHT: 0.004, FORCE_MEDIUM: 0.004, FORCE_HARD: 0.0}

# Approximate grasp force [N] of each close force class.
CLOSE_FORCES = {FORCE_LIGHT: 100.0, FORCE_MEDIUM: 140.0, FORCE_HARD: 220.0}

# Datasheet specs of the 2F-85. The speed and force ranges describe the *gripper*; over a FANUC only
# the buckets above are reachable.
ROBOTIQ_2F85_FANUC_SPECS = ParallelPositionGripperSpecs(0.085, 0.0, 220, 25, 0.15, 0.02)

_NO_FEEDBACK_MESSAGE = (
    "A FANUC controller has no gripper API: the driver writes registers that a teach-pendant program "
    "watches, and the only readable state is whether that program has finished — no width, no force, no "
    "speed, no grasp detection. See airo_robots/grippers/hardware/fanuc_robotiq.md."
)

_NO_SPEED_MESSAGE = (
    f"The finger speed of a Robotiq 2F-85 on a FANUC is fixed by the GRIPDISP teach-pendant program on "
    f"the controller and can be neither read nor set from here. {_NO_FEEDBACK_MESSAGE}"
)


def _import_airo_fanuc_gripper() -> ModuleType:
    """Import the optional airo-fanuc gripper module with an actionable error message."""
    try:
        return importlib.import_module("airo_fanuc.gripper")
    except ImportError as exception:
        raise ImportError(
            'Robotiq2F85Fanuc requires the airo-fanuc driver. Install it with `pip install "airo-robots[fanuc]"`.'
        ) from exception


class Robotiq2F85Fanuc(ParallelPositionGripper):
    """A Robotiq 2F-85 wired to a FANUC controller, driven through the airo-fanuc driver.

    Only the discrete buckets in `OPEN_WIDTHS` and `CLOSE_FORCES` are reachable and no gripper state
    is readable, so `move` snaps to the nearest reachable opening and everything needing feedback or a
    continuous setpoint raises ``NotImplementedError``. ``fanuc_robotiq.md`` in this directory has the
    mechanism, the bucket tables and the completion contract.

    Args:
        driver: a connected ``airo_fanuc.FanucDriver`` that was brought up with
            ``DriverPolicy(enable_gripper=True)`` and the Robotiq 2F-85 protocol (the default).
        gripper_specs: Optional specification override.

    Raises:
        ValueError: If the driver has no gripper worker, or its worker is driving a different
            register protocol than the Robotiq 2F-85 preset this class implements.
    """

    def __init__(self, driver: Any, gripper_specs: Optional[ParallelPositionGripperSpecs] = None) -> None:
        if driver.gripper is None:
            raise ValueError(
                "This FanucDriver has no gripper worker. Bring the driver up with "
                "`DriverPolicy(..., enable_gripper=True)` (the default) so that it launches the gripper "
                "dispatcher on the controller."
            )
        self._assert_protocol_is_robotiq_2f85(driver)

        self.driver = driver
        self._worker = driver.gripper
        # The interface lets a caller set a force once and have it apply to every later move.
        self._close_force = FORCE_MEDIUM
        self._last_commanded_width: Optional[float] = None

        super().__init__(gripper_specs if gripper_specs is not None else ROBOTIQ_2F85_FANUC_SPECS)

    @staticmethod
    def _assert_protocol_is_robotiq_2f85(driver: Any) -> None:
        """Check the driver's register protocol against the bucket values this class assumes.

        A driver whose protocol renumbers the buckets, or which is driving a different gripper
        altogether, would otherwise have this class silently commanding the wrong bucket.
        """
        robotiq_protocol = _import_airo_fanuc_gripper().ROBOTIQ_2F85
        configured = driver.gripper.protocol
        if configured is not robotiq_protocol:
            raise ValueError(
                f"The driver is configured with the {configured.name!r} gripper protocol, but this class "
                f"implements {robotiq_protocol.name!r}. Drive your own gripper through `driver.gripper` "
                f"directly, or write a ParallelPositionGripper for it."
            )
        if configured.open_modifiers != tuple(OPEN_WIDTHS) or configured.close_modifiers != tuple(CLOSE_FORCES):
            raise ValueError(
                f"The airo-fanuc Robotiq 2F-85 protocol accepts open buckets {configured.open_modifiers} and "
                f"close classes {configured.close_modifiers}, but this implementation was written against "
                f"{tuple(OPEN_WIDTHS)} and {tuple(CLOSE_FORCES)}. Update this module to match the driver."
            )

    def move(self, width: float, speed: Optional[float] = None, force: Optional[float] = None) -> AwaitableAction:
        """Move the fingers to the reachable opening closest to ``width`` [m].

        A width that snaps to the closed position is sent as a *close* at the current force class, so
        the gripper stops on an object instead of at a position.

        Args:
            width: the desired opening between the fingers [m], snapped to the nearest reachable one.
            speed: not supported, see the `speed` property. Pass ``None``.
            force: the maximum grasp force [N], snapped to the nearest force class and used for this
                and every later close, as in the interface.

        Raises:
            NotImplementedError: if a speed is given.
        """
        if speed is not None:
            raise NotImplementedError(_NO_SPEED_MESSAGE)
        if force is not None:
            self.max_grasp_force = force

        target_width = float(np.clip(width, self.gripper_specs.min_width, self.gripper_specs.max_width))
        closed_width = CLOSE_WIDTHS[self._close_force]
        open_bucket = min(OPEN_WIDTHS, key=lambda bucket: abs(OPEN_WIDTHS[bucket] - target_width))

        if abs(target_width - closed_width) < abs(target_width - OPEN_WIDTHS[open_bucket]):
            reached, description = closed_width, f"close at force class {self._close_force}"
            self._worker.close_gripper(self._close_force)
        else:
            reached, description = OPEN_WIDTHS[open_bucket], f"open bucket {open_bucket}"
            self._worker.open_gripper(open_bucket)

        if abs(reached - width) > 0.001:
            logger.info(
                f"A Robotiq 2F-85 on a FANUC only reaches discrete openings, so the requested {width * 1000:.1f} mm "
                f"was sent as a {description}, which reaches about {reached * 1000:.1f} mm."
            )
        self._last_commanded_width = reached
        return self._gripper_action()

    def get_current_width(self) -> float:
        """Not supported: the gripper reports nothing back over this path."""
        raise NotImplementedError(
            f"The opening of a Robotiq 2F-85 on a FANUC cannot be read. {_NO_FEEDBACK_MESSAGE} Use "
            f"`gripper.last_commanded_width` if the last commanded opening is good enough for your purpose."
        )

    @property
    def last_commanded_width(self) -> Optional[float]:
        """The nominal opening [m] of the last command, or ``None`` before the first one.

        A command, not a measurement: a close that stopped on an object ends up wider than this.
        """
        return self._last_commanded_width

    @property
    def speed(self) -> float:
        """Not supported: the GRIPDISP program on the controller fixes the finger speed."""
        raise NotImplementedError(_NO_SPEED_MESSAGE)

    @speed.setter
    def speed(self, new_speed: float) -> None:
        """Not supported, see the `speed` property."""
        raise NotImplementedError(_NO_SPEED_MESSAGE)

    @property
    def max_grasp_force(self) -> float:
        """The approximate grasp force [N] of the force class the next close will use."""
        return CLOSE_FORCES[self._close_force]

    @max_grasp_force.setter
    def max_grasp_force(self, new_force: float) -> None:
        """Select the force class closest to ``new_force`` [N] for this and every later close."""
        force = float(np.clip(new_force, self.gripper_specs.min_force, self.gripper_specs.max_force))
        self._close_force = min(CLOSE_FORCES, key=lambda cls: abs(CLOSE_FORCES[cls] - force))
        if abs(CLOSE_FORCES[self._close_force] - new_force) > 10.0:
            logger.info(
                f"A Robotiq 2F-85 on a FANUC only has three force classes, so the requested {new_force:.0f} N "
                f"was rounded to class {self._close_force} (about {CLOSE_FORCES[self._close_force]:.0f} N)."
            )

    @property
    def close_force_class(self) -> int:
        """The force class (``FORCE_LIGHT`` / ``FORCE_MEDIUM`` / ``FORCE_HARD``) the next close will use."""
        return self._close_force

    @close_force_class.setter
    def close_force_class(self, force_class: int) -> None:
        if force_class not in CLOSE_FORCES:
            raise ValueError(f"The close force class must be one of {tuple(CLOSE_FORCES)}, got {force_class!r}.")
        self._close_force = force_class

    def is_an_object_grasped(self) -> bool:
        """Not supported: the gripper's object-detection status never reaches the FANUC controller."""
        raise NotImplementedError(f"Grasp detection is not available on a FANUC. {_NO_FEEDBACK_MESSAGE}")

    @property
    def last_result(self) -> Optional[Dict[str, Any]]:
        """The dispatcher's verdict on the last command: ``{"success": bool, "message": str}``.

        ``success`` means the teach-pendant program said it finished, not that a width was reached or
        an object is held. ``None`` before the first command, or while one is still running.
        """
        return self._worker.last_result

    def _gripper_action(self) -> AwaitableAction:
        """An AwaitableAction that completes when the dispatcher has finished the submitted command.

        The driver's worker always reaches a verdict within its own dispatch timeout, so this cannot
        hang on a wedged dispatcher: the action completes and the failure is logged.
        """

        def is_gripper_done() -> bool:
            result = self._worker.last_result
            if result is None:
                return False
            if not result["success"]:
                logger.error(f"The FANUC gripper command did not succeed: {result['message']}")
            return True

        # The condition reads an in-process flag rather than the controller, so it can be polled fast.
        return AwaitableAction(is_gripper_done, default_timeout=10.0, default_sleep_resolution=0.005)


def manually_test_fanuc_gripper(gripper: Robotiq2F85Fanuc) -> None:
    """Manually test a Robotiq 2F-85 on a FANUC.

    The generic ``manually_test_gripper_implementation`` cannot be used: it reads back the finger
    width and sets a speed, neither of which exists on this path.
    """
    input("gripper will now open fully")
    gripper.open().wait()
    print(f"result: {gripper.last_result}")

    for bucket, width in OPEN_WIDTHS.items():
        input(f"gripper will now open to bucket {bucket} (about {width * 1000:.0f} mm)")
        gripper.move(width).wait()
        print(f"result: {gripper.last_result}")

    input("a width in between the buckets (5 cm) will now be requested, it should snap to the nearest bucket")
    gripper.move(0.05).wait()
    print(f"snapped to {gripper.last_commanded_width}, result: {gripper.last_result}")

    input("gripper will now close lightly, you can put an object between the fingers")
    gripper.max_grasp_force = CLOSE_FORCES[FORCE_LIGHT]
    gripper.close().wait()
    print(f"force class {gripper.close_force_class} ({gripper.max_grasp_force} N), result: {gripper.last_result}")

    input("gripper will now close at full force")
    gripper.max_grasp_force = gripper.gripper_specs.max_force
    gripper.close().wait()
    print(f"force class {gripper.close_force_class} ({gripper.max_grasp_force} N), result: {gripper.last_result}")

    input("gripper will now reopen fully")
    gripper.open().wait()
    print(f"result: {gripper.last_result}")

    for unsupported in ("get_current_width", "is_an_object_grasped"):
        try:
            getattr(gripper, unsupported)()
        except NotImplementedError as exception:
            print(f"{unsupported} is not available, as expected: {exception}")


if __name__ == "__main__":
    """test script for a Robotiq 2F-85 on a FANUC.
    e.g. python airo-robots/airo_robots/grippers/hardware/robotiq_2f85_fanuc.py --ip_address 192.168.1.100
    """
    import click
    from airo_robots.manipulators.hardware.fanuc import Fanuc, _import_airo_fanuc, create_crx10ial_profile

    @click.command()
    @click.option("--ip_address", default="192.168.1.100", show_default=True, help="IP address of the FANUC robot.")
    def test_robotiq_fanuc(ip_address: str) -> None:
        """Run the manual gripper tests against a Robotiq 2F-85 mounted on a FANUC."""
        fanuc = _import_airo_fanuc()

        print("The arm will not move, but the GRIPPER WILL: bring-up probes the dispatcher with a benign open,")
        print("and the tests then run through every opening bucket and force class. Keep your hands clear.")
        input(f"about to connect to the FANUC at {ip_address}, press key to start")

        # Bring-up builds the gripper worker and launches the GRIPDISP dispatcher. It can take more
        # than one attempt on a cold controller, so the driver's default of 3 retries is left alone.
        policy = fanuc.DriverPolicy(
            config=fanuc.DriverConfig(profile=create_crx10ial_profile()),
            enable_gripper=True,
        )
        with Fanuc(ip_address, policy) as robot:
            # `Fanuc` wraps the driver's gripper worker itself, but wrapping it here is equivalent and
            # keeps this script independent of the class above having been imported as `__main__`.
            manually_test_fanuc_gripper(Robotiq2F85Fanuc(robot.driver))

    test_robotiq_fanuc()
