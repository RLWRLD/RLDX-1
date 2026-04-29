from robocasa.environments.tabletop.tabletop import *
from robocasa.utils.dexmg_utils import DexMGConfigHelper


class TabletopDrawerDoor(Tabletop, DexMGConfigHelper):
    """
    Class encapsulating the atomic drawer door manipulation tasks.

    Args:
        behavior (str): "open" or "close". Used to define the desired
            drawer door manipulation behavior for the task
    """

    VALID_LAYOUTS = [4]

    def __init__(self, behavior=None, *args, **kwargs):
        assert behavior in ["open", "close"]
        self.behavior = behavior
        super().__init__(*args, **kwargs)

    def _setup_table_references(self):
        """
        Setup the table references for the drawer door tasks
        """
        super()._setup_table_references()
        self.drawer = self.get_fixture("drawer_tabletop_main_group")
        self.init_robot_base_pos = self.drawer

    def get_ep_meta(self):
        """
        Get the episode metadata for the drawer door tasks.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"{self.behavior} the drawer door"
        return ep_meta

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

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the drawer door tasks.
        No objects needed for this task.
        """
        return []

    def _check_success(self):
        """
        Check if the drawer door manipulation task is successful.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        door_state = self.drawer.get_door_state(env=self)["door"]
        if self.behavior == "open":
            return door_state >= 0.5
        elif self.behavior == "close":
            return door_state <= 0.005

        return False

    def get_object(self):
        objects = dict()
        objects["door"] = dict(
            obj_name=self.drawer.name + "_door_handle_handle",
            obj_type="geom",
            obj_joint=None,
        )
        return objects

    def get_subtask_term_signals(self):
        signals = dict()
        signals["open_door"] = int(self._check_success())
        return signals

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec.subtask_1 = dict(
            object_ref="door",
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


class TabletopOpenDrawerDoor(TabletopDrawerDoor):
    def __init__(self, *args, **kwargs):
        super().__init__(behavior="open", *args, **kwargs)


class TabletopCloseDrawerDoor(TabletopDrawerDoor):
    def __init__(self, *args, **kwargs):
        super().__init__(behavior="close", *args, **kwargs)
