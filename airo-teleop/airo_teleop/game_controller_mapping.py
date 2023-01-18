from dataclasses import dataclass


@dataclass
class GameControllerLayout:
    """Pygame mapping for Game controllers

    each game controller is considered to have 2 joysticks, 1 cross, buttons on the back (LB/LT) and 4 buttons
    examples include: Playstation controllers, xbox controllers...
    """

    left_joy_horizontal_index: int
    left_joy_vertical_axis_index: int
    right_joy_horizontal_axis_index: int
    right_joy_vertical_axis_index: int
    lt_axis_index: int
    rt_axis_index: int

    lb_button_index: int
    rb_button_index: int
    a_button_index: int
    b_button_index: int
    y_button_index: int
    x_button_index: int

    horizontal_cross_index: int  # index of the 2d vector for the cross inputs that represents the horizontal direction


XBox360Layout = GameControllerLayout(
    left_joy_horizontal_index=0,
    left_joy_vertical_axis_index=1,
    right_joy_horizontal_axis_index=3,
    right_joy_vertical_axis_index=4,
    lt_axis_index=2,
    rt_axis_index=5,
    lb_button_index=4,
    rb_button_index=5,
    a_button_index=0,
    b_button_index=1,
    y_button_index=3,
    x_button_index=2,
    horizontal_cross_index=0,
)

LogitechF310Layout = GameControllerLayout(
    left_joy_horizontal_index=0,
    left_joy_vertical_axis_index=1,
    right_joy_horizontal_axis_index=3,
    right_joy_vertical_axis_index=4,
    lt_axis_index=2,
    rt_axis_index=5,
    lb_button_index=4,
    rb_button_index=5,
    a_button_index=0,
    b_button_index=1,
    y_button_index=3,
    x_button_index=2,
    horizontal_cross_index=0,
)
