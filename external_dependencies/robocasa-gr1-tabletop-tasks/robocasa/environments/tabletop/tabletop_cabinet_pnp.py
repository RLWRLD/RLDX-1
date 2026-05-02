from robocasa.environments.tabletop.tabletop import *
from robocasa.environments.tabletop.tabletop_pnp import DEFAULT_DISTRACTOR_CONFIG
from robocasa.utils.dexmg_utils import DexMGConfigHelper
from robocasa.utils.object_utils import check_obj_in_receptacle, obj_inside_of

DISTRACTOR_CONFIG = DEFAULT_DISTRACTOR_CONFIG.copy()
DISTRACTOR_CONFIG["regions"]["back_edge"]["objects"] = [
    {
        "type": ["vegetable"],
        "count": (1, 4),
    }
]


class TabletopCabinetPnPClose(Tabletop, DexMGConfigHelper):
    """
    Class encapsulating the cabinet-based pick and place task.

    Args:
        obj_groups (str): Object groups to sample the target object from.
        exclude_obj_groups (str): Object groups to exclude from sampling the target object.
        handedness (Optional[str]): Which hand to optimize object spawning for ("right" or "left").
        obj_scale (float): Scaling factor for the object.
        behavior (str): "open" or "close" for cabinet door manipulation behavior.
    """

    VALID_LAYOUTS = [5]
    NUM_OBJECTS = 1

    def __init__(
        self,
        obj_groups="all",
        exclude_obj_groups=None,
        handedness="left",
        obj_scale=1.0,
        behavior="close",
        distractor_config=DISTRACTOR_CONFIG,
        use_distractors=True,
        *args,
        **kwargs,
    ):
        if handedness not in ("right", "left"):
            raise ValueError("handedness must be 'right' or 'left'")
        assert behavior in ["open", "close"], "Invalid behavior"

        self.obj_groups = obj_groups
        self.exclude_obj_groups = exclude_obj_groups
        self.handedness = handedness
        self.obj_scale = obj_scale
        self.behavior = behavior

        super().__init__(
            *args,
            **kwargs,
            distractor_config=distractor_config,
            use_distractors=use_distractors,
        )

    def _setup_table_references(self):
        """
        Setup references for the cabinet and workspace.
        """
        super()._setup_table_references()
        self.cabinet = self.get_fixture(FixtureType.DOOR_HINGE_SINGLE)
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, size=(0.45, 0.55))
        )
        self.init_robot_base_pos = self.cabinet

    def get_ep_meta(self):
        """
        Get the episode metadata for the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang(obj_name="obj")
        ep_meta[
            "lang"
        ] = f"pick up the {obj_lang}, place it into the cabinet and close the cabinet"
        return ep_meta

    def _reset_internal(self):
        """
        Reset the environment internal state for the cabinet door tasks.
        This includes setting the door state based on the behavior.
        """
        super()._reset_internal()
        if self.behavior == "open":
            self.cabinet.set_door_state(min=0.0, max=0.0, env=self, rng=self.rng)
        elif self.behavior == "close":
            self.cabinet.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        """
        Define object configurations for pick and place.
        """
        cfgs = []
        handedness = (
            self.handedness if self.handedness else self.rng.choice(["left", "right"])
        )
        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                object_scale=self.obj_scale,
                placement=dict(
                    fixture=self.counter,
                    size=(0.2, 0.2),
                    pos=((0.5, -0.8) if handedness == "right" else (-0.5, -0.8)),
                ),
                obj_registries=["objaverse"],
            )
        )
        return cfgs

    def _check_success(self):
        """
        Check if the object is successfully placed inside the cabinet.
        """
        inside_of_cabinet = obj_inside_of(
            env=self,
            obj_name=self.objects["obj"].name,
            fixture_id=self.cabinet,
            partial_check=True,
        )
        door_state_correct = (
            self.is_door_open() if self.behavior == "open" else self.is_door_closed()
        )
        return inside_of_cabinet and door_state_correct

    def is_door_open(self):
        """
        Check if the cabinet door is opened.
        """
        door_state = self.cabinet.get_door_state(env=self)["door"]
        return door_state >= 0.5

    def is_door_closed(self):
        """
        Check if the cabinet door is closed.
        """
        door_state = self.cabinet.get_door_state(env=self)["door"]
        return door_state <= 0.005

    def get_object(self):
        """
        Return object references for the task.
        """
        objects = {
            "obj": dict(
                obj_name=self.objects["obj"].root_body,
                obj_type="body",
                obj_joint=None,
            ),
            "cabinet": dict(
                obj_name=self.cabinet.name + "_door_door",
                obj_type="geom",
                obj_joint=None,
            ),
        }
        return objects

    def get_subtask_term_signals(self):
        """
        Define subtask termination signals.
        """
        signals = {
            "grasp_object": int(
                self._check_grasp(
                    gripper=self.robots[0].gripper["left"],
                    object_geoms=self.objects["obj"],
                )
            ),
            "obj_in_cabinet": int(
                OU.check_obj_fixture_contact(self, "obj", self.cabinet)
            ),
        }
        return signals

    @staticmethod
    def task_config():
        return {
            "task_spec_1": {
                "subtask_1": dict(
                    object_ref="obj",
                    subtask_term_signal="grasp_object",
                    subtask_term_offset_range=(5, 10),
                    selection_strategy="random",
                    selection_strategy_kwargs=None,
                    action_noise=0.0,
                    num_interpolation_steps=5,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=True,
                ),
                "subtask_2": dict(
                    object_ref="cabinet",
                    subtask_term_signal=None,
                    subtask_term_offset_range=None,
                    selection_strategy="random",
                    selection_strategy_kwargs=None,
                    action_noise=0.0,
                    num_interpolation_steps=5,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=True,
                ),
            },
            "task_spec_0": {
                "subtask_1": dict(
                    object_ref="cabinet",
                    subtask_term_signal=None,
                    subtask_term_offset_range=None,
                    selection_strategy="random",
                    selection_strategy_kwargs=None,
                    action_noise=0.0,
                    num_interpolation_steps=5,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=True,
                ),
            },
        }


class PnPCupToCabinetClose(TabletopCabinetPnPClose):
    def __init__(self, *args, **kwargs):
        super().__init__(obj_groups="cup", *args, **kwargs, obj_scale=1.25)


class PnPBottleToCabinetClose(TabletopCabinetPnPClose):
    def __init__(self, *args, **kwargs):
        super().__init__(obj_groups="bottled_water", *args, **kwargs, obj_scale=1.1)


class PnPWineToCabinetClose(TabletopCabinetPnPClose):
    def __init__(self, *args, **kwargs):
        super().__init__(obj_groups="wine", *args, **kwargs, obj_scale=1)


class PnPCanToCabinetClose(TabletopCabinetPnPClose):
    def __init__(self, *args, **kwargs):
        super().__init__(obj_groups="can", *args, **kwargs)


class PnPAppleToCabinetClose(TabletopCabinetPnPClose):
    def __init__(self, *args, **kwargs):
        super().__init__(obj_groups="apple", *args, **kwargs, obj_scale=1.7)
