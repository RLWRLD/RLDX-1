import numpy as np

from robosuite.robots import register_robot_class
from robosuite.models.robots import GR1
from robosuite.models.robots.manipulators.gr1_robot import (
    GR1FixedLowerBody,
    GR1ArmsOnly,
)


@register_robot_class("LeggedRobot")
class GR1FixedLowerBodyInspireHands(GR1FixedLowerBody):
    @property
    def default_gripper(self):
        return {"right": "InspireRightHand", "left": "InspireLeftHand"}


@register_robot_class("LeggedRobot")
class GR1FixedLowerBodyFourierHands(GR1FixedLowerBody):
    @property
    def default_gripper(self):
        return {"right": "FourierRightHand", "left": "FourierLeftHand"}


@register_robot_class("LeggedRobot")
class GR1ArmsOnlyInspireHands(GR1ArmsOnly):
    @property
    def default_gripper(self):
        return {"right": "InspireRightHand", "left": "InspireLeftHand"}


@register_robot_class("LeggedRobot")
class GR1ArmsOnlyFourierHands(GR1ArmsOnly):
    @property
    def default_gripper(self):
        return {"right": "FourierRightHand", "left": "FourierLeftHand"}


@register_robot_class("LeggedRobot")
class GR1ArmsAndWaist(GR1):
    def __init__(self, idn=0):
        super().__init__(idn=idn)
        self._remove_joint_actuation("leg")
        self._remove_joint_actuation("head")
        self._remove_free_joint()

    @property
    def init_qpos(self):
        init_qpos = np.array([0.0] * 17)
        right_arm_init = np.array([0.0, -0.1, 0.0, -1.57, 0.0, 0.0, 0.0])
        left_arm_init = np.array([0.0, 0.1, 0.0, -1.57, 0.0, 0.0, 0.0])
        init_qpos[3:10] = right_arm_init
        init_qpos[10:17] = left_arm_init
        return init_qpos


@register_robot_class("LeggedRobot")
class GR1ArmsAndWaistFourierHands(GR1ArmsAndWaist):
    pass
