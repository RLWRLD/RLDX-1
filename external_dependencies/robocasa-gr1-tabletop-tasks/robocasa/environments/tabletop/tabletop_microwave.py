from robocasa.environments.tabletop.tabletop import *
from robocasa.utils.dexmg_utils import DexMGConfigHelper


class TabletopMicrowavePressButton(Tabletop, DexMGConfigHelper):
    """
    Class encapsulating the atomic microwave press button tasks.

    Args:
        behavior (str): "turn_on" or "turn_off". Used to define the desired
            microwave manipulation behavior for the task
    """

    VALID_LAYOUTS = [2]

    def __init__(self, behavior="turn_on", *args, **kwargs):
        assert behavior in ["turn_on", "turn_off"]
        self.behavior = behavior
        super().__init__(*args, **kwargs)

    def _setup_table_references(self):
        """
        Setup the kitchen references for the microwave tasks
        """
        super()._setup_table_references()
        self.microwave = self.get_fixture(FixtureType.MICROWAVE)
        if self.behavior == "turn_off":
            self.microwave._turned_on = True
        self.init_robot_base_pos = self.microwave

    def get_ep_meta(self):
        """
        Get the episode metadata for the microwave tasks.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        if self.behavior == "turn_on":
            ep_meta["lang"] = "press the start button on the microwave"
        elif self.behavior == "turn_off":
            ep_meta["lang"] = "press the stop button on the microwave"
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the microwave tasks. This includes the object placement configurations.
        Place the object inside the microwave and on top of another container object inside the microwave

        Returns:
            list: List of object configurations.
        """
        cfgs = []

        return cfgs

    def _check_success(self):
        """
        Check if the microwave manipulation task is successful.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        turned_on = self.microwave.get_state()["turned_on"]
        gripper_button_far = self.microwave.gripper_button_far(
            self, button="start_button" if self.behavior == "turn_on" else "stop_button"
        )

        if self.behavior == "turn_on":
            return turned_on and gripper_button_far
        elif self.behavior == "turn_off":
            return not turned_on and gripper_button_far

    def get_object(self):
        objects = dict()
        objects["button"] = dict(
            obj_name=self.microwave.name + "_start_button",
            obj_type="geom",
            obj_joint=None,
        )
        return objects

    def get_subtask_term_signals(self):
        signals = dict()
        signals["press_button"] = int(self._check_success())
        return signals

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec.subtask_1 = dict(
            object_ref="button",
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


class TabletopTurnOnMicrowave(TabletopMicrowavePressButton):
    def __init__(self, behavior=None, *args, **kwargs):
        super().__init__(behavior="turn_on", *args, **kwargs)


class TabletopTurnOffMicrowave(TabletopMicrowavePressButton):
    def __init__(self, behavior=None, *args, **kwargs):
        super().__init__(behavior="turn_off", *args, **kwargs)
