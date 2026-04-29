import numpy as np

from robocasa.utils.dexmg_utils import DexMGConfigHelper
import robocasa.utils.object_utils as OU

from robocasa import Tabletop
from robocasa.environments.tabletop.tabletop_pnp import DEFAULT_DISTRACTOR_CONFIG
from robocasa.models.fixtures import FixtureType
from robosuite.utils.mjcf_utils import string_to_array

# Default distractor configuration for TabletopLaptopInteraction environment
DEFAULT_DISTRACTOR_CONFIG = {
    "regions": {
        "left_edge": {
            "placement": {
                "pos": (-1, 0),
                "size": (0.4, 1.2),
            },
            "fixtures": [
                {
                    "type": "toaster",
                    "count": (0, 1),
                },
                {
                    "type": "paper_towel",
                    "count": (0, 1),
                },
                {
                    "type": "plant",
                    "count": (0, 1),
                },
            ],
            "objects": [],
        },
        "right_edge": {
            "placement": {
                "pos": (1, 0),
                "size": (0.4, 1.2),
            },
            "fixtures": [
                {
                    "type": "toaster",
                    "count": (0, 1),
                },
                {
                    "type": "paper_towel",
                    "count": (0, 1),
                },
                {
                    "type": "plant",
                    "count": (0, 1),
                },
            ],
            "objects": [],
        },
    }
}


class TabletopLaptopInteraction(Tabletop, DexMGConfigHelper):
    """
    Class encapsulating the atomic laptop interaction tasks.
    """

    VALID_LAYOUTS = [0]

    def __init__(
        self,
        behavior="open",
        distractor_config=None,
        use_distractors=True,
        *args,
        **kwargs
    ):
        assert behavior in ["open", "close"]
        self.behavior = behavior
        distractor_config = distractor_config or DEFAULT_DISTRACTOR_CONFIG
        super().__init__(
            distractor_config=distractor_config,
            use_distractors=use_distractors,
            *args,
            **kwargs
        )

    def _reset_internal(self):
        _, _, self._laptop = self.object_placements["obj"]
        self._lid_joint = self._laptop.joints[0]
        self._lid_joint_range = string_to_array(
            self._laptop.get_joint("lid_hinge").attrib["range"]
        )
        lid_range = np.max(self._lid_joint_range) - np.min(self._lid_joint_range)
        closed_offset = 0.1 * lid_range
        rnd_offset = self.rng.uniform(closed_offset)
        lid_angle = (
            np.min(self._lid_joint_range) + closed_offset + rnd_offset
            if self.behavior == "open"
            else np.max(lid_range) - rnd_offset
        )
        self.sim.data.set_joint_qpos(self._lid_joint, lid_angle)
        super()._reset_internal()

    def _setup_table_references(self):
        super()._setup_table_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, size=(0.45, 0.55))
        )
        self.init_robot_base_pos = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self.behavior == "open":
            ep_meta["lang"] = "open lid of the laptop"
        elif self.behavior == "close":
            ep_meta["lang"] = "close lid of the laptop"
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=["laptop"],
                placement=dict(fixture=self.counter, size=(0.5, 0.5), pos=(0.0, 0.0)),
            )
        )
        return cfgs

    def _check_success(self):
        gripper_obj_far = OU.gripper_obj_far(self, obj_name="obj")
        lid_qpos = self.sim.data.get_joint_qpos(self._lid_joint)
        closed = lid_qpos <= np.deg2rad(10)
        opened = lid_qpos >= np.deg2rad(85)
        if self.behavior == "open":
            return opened and gripper_obj_far
        elif self.behavior == "close":
            return closed and gripper_obj_far

    def get_object(self):
        objects = dict()
        objects["obj"] = dict(
            obj_name=self.objects["obj"].root_body, obj_type="body", obj_joint=None
        )
        return objects

    def get_subtask_term_signals(self):
        signals = dict()
        signals["contact_laptop"] = int(
            self.check_contact(
                self.robots[0].gripper["right"],
                self.objects["obj"],
            )
        )
        return signals

    @staticmethod
    def task_config():
        # task_spec_0 for right hand, task_spec_1 for left hand
        task = DexMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = dict(
            object_ref="obj",
            subtask_term_signal="contact_laptop",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=True,
        )
        task.task_spec_0.subtask_2 = dict(
            object_ref="obj",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=True,
        )
        task.task_spec_1.subtask_1 = dict(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=True,
        )
        return task.to_dict()


class TabletopLaptopOpen(TabletopLaptopInteraction):
    def __init__(self, *args, **kwargs):
        super().__init__(behavior="open", *args, **kwargs)


class TabletopLaptopClose(TabletopLaptopInteraction):
    def __init__(self, *args, **kwargs):
        super().__init__(behavior="close", *args, **kwargs)
