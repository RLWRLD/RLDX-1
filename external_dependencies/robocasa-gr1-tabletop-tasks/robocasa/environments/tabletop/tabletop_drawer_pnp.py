from robocasa.environments.tabletop.tabletop import *
from robocasa.environments.tabletop.tabletop_pnp import DEFAULT_DISTRACTOR_CONFIG
from robocasa.utils.dexmg_utils import DexMGConfigHelper
from robocasa.utils.object_utils import obj_inside_of

DISTRACTOR_CONFIG = DEFAULT_DISTRACTOR_CONFIG.copy()
DISTRACTOR_CONFIG["regions"]["back_edge"]["objects"] = [
    {
        "type": ["vegetable"],
        "count": (1, 4),
    }
]
DISTRACTOR_CONFIG["regions"]["back_edge"]["fixtures"] = [
    {
        "type": "toaster",
        "count": (0, 1),
    },
    {
        "type": "plant",
        "count": (0, 1),
    },
]


class TabletopDrawerPnPClose(Tabletop, DexMGConfigHelper):
    """
    Class encapsulating the atomic counter to container pick and place task.

    Args:
        container_type (str): Type of container to place the object in.

        obj_groups (str): Object groups to sample the target object from.

        exclude_obj_groups (str): Object groups to exclude from sampling the target object.

        distractor_config (dict): Configuration for distractor objects.

        use_distractors (bool): Whether to use distractor objects.

        handedness (Optional[str]): Which hand to optimize object spawning for ("right" or "left").

        source_container (str): Source container to sample the target object from.

        target_container (str): Target container to place the object in.
    """

    VALID_LAYOUTS = [4]
    NUM_OBJECTS = 1

    def __init__(
        self,
        obj_groups="all",
        exclude_obj_groups=None,
        source_container=None,
        source_container_size=(0.5, 0.5),
        target_container=None,
        target_container_size=(0.5, 0.5),
        handedness="left",
        behavior="close",
        obj_scale=1.0,
        distractor_config=DISTRACTOR_CONFIG,
        use_distractors=True,
        *args,
        **kwargs,
    ):
        if handedness is not None and handedness not in ("right", "left"):
            raise ValueError("handedness must be 'right' or 'left'")
        self.target_container = target_container
        self.source_container = source_container
        self.source_container_size = source_container_size
        self.target_container_size = target_container_size
        self.obj_groups = obj_groups
        self.obj_scale = obj_scale
        self.exclude_obj_groups = exclude_obj_groups
        self.handedness = handedness
        self.behavior = behavior

        super().__init__(
            *args,
            **kwargs,
            distractor_config=distractor_config,
            use_distractors=use_distractors,
        )

    def _setup_table_references(self):
        """
        Setup the table references for the drawer door tasks
        """
        super()._setup_table_references()
        self.drawer = self.get_fixture("drawer_tabletop_main_group")
        self.init_robot_base_pos = self.drawer
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, size=(0.45, 0.55))
        )

    def get_ep_meta(self):
        """
        Get the episode metadata for the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang(obj_name="obj")
        ep_meta[
            "lang"
        ] = f"pick up the {obj_lang}, place it into the drawer and close the drawer"
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        handedness = (
            self.handedness
            if self.handedness is not None
            else self.rng.choice(["left", "right"])
        )
        # Add objects to manipulate
        cfgs.append(
            dict(
                name=f"obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                object_scale=self.obj_scale,
                placement=dict(
                    fixture=self.counter,
                    size=(0.2, 0.2),
                    pos=(
                        (0.5, -0.8) if handedness == "right" else (-0.5, -0.7)
                    ),  # place object in right front corner if right handed
                ),
                obj_registries=["objaverse"],
            )
        )
        return cfgs

    def _check_success(self):
        """
        Check if the object is successfully placed inside the drawer.
        """
        inside_of_drawer = obj_inside_of(
            env=self,
            obj_name=self.objects["obj"].name,
            fixture_id=self.drawer,
            partial_check=True,
        )
        door_state = self.drawer.get_door_state(env=self)["door"]
        door_state_correct = (
            door_state >= 0.5 if self.behavior == "open" else door_state <= 0.005
        )
        return inside_of_drawer and door_state_correct

    def get_object(self):
        """
        Return object references for the task.
        """
        objects = dict()
        objects["obj"] = dict(
            obj_name=self.objects["obj"].root_body,
            obj_type="body",
            obj_joint=None,
        )
        objects["drawer"] = dict(
            obj_name=self.drawer.name + "_bottom",
            obj_type="geom",
            obj_joint=None,
        )
        return objects

    def get_subtask_term_signals(self):
        signals = dict()
        signals["grasp_object"] = int(
            self._check_grasp(
                gripper=self.robots[0].gripper["left"],
                object_geoms=self.objects["obj"],
            )
        )
        signals["obj_in_drawer"] = int(
            OU.check_obj_fixture_contact(self, "obj", self.drawer)
        )

        return signals

    def _reset_internal(self):
        """
        Reset the environment internal state for the drawer door tasks.
        This includes setting the door state based on the behavior.
        """
        super()._reset_internal()
        if self.behavior == "open":
            self.drawer.set_door_state(min=0.0, max=0.0, env=self, rng=self.rng)
        elif self.behavior == "close":
            self.drawer.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)

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
                    object_ref="drawer",
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
                    object_ref="drawer",
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


class PnPCupToDrawerClose(TabletopDrawerPnPClose):
    """
    Class encapsulating the atomic cup to bowl pick and place task.
    A deliberately simple task used for testing.

    cup is chosen as it's easy to grasp and place.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(obj_groups="cup", *args, **kwargs, obj_scale=1.1)


class PnPBottleToDrawerClose(TabletopDrawerPnPClose):
    """
    Class encapsulating the atomic bottle to bowl pick and place task.
    A deliberately simple task used for testing.

    bottle is chosen as it's easy to grasp and place.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(obj_groups="bottled_water", *args, **kwargs, obj_scale=1.1)


class PnPWineToDrawerClose(TabletopDrawerPnPClose):
    """
    Class encapsulating the atomic wine to bowl pick and place task.
    A deliberately simple task used for testing.

    wine is chosen as it's easy to grasp and place.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(obj_groups="wine", *args, **kwargs, obj_scale=0.9)


class PnPCanToDrawerClose(TabletopDrawerPnPClose):
    """
    Class encapsulating the atomic can to bowl pick and place task.
    A deliberately simple task used for testing.

    can is chosen as it's easy to grasp and place.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(obj_groups="can", *args, **kwargs, obj_scale=1.2)


class PnPAppleToDrawerClose(TabletopDrawerPnPClose):
    """
    Class encapsulating the atomic apple to bowl pick and place task.
    A deliberately simple task used for testing.

    apple is chosen as it's easy to grasp and place.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(obj_groups="apple", *args, **kwargs, obj_scale=1.1)
