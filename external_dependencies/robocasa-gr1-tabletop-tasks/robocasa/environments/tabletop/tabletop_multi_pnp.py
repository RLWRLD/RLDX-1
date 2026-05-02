from robocasa.environments.tabletop.tabletop import *
from robocasa.utils.dexmg_utils import DexMGConfigHelper


class PutAllObjectsInContainer(Tabletop, DexMGConfigHelper):
    """
    Class for tasks where multiple objects need to be placed in a container.
    Objects are randomly spawned on the counter and need to be placed in the container.

    Args:
        container_type (str): Type of container to place objects in.

        num_objects (int): Number of objects to spawn and manipulate.

        obj_groups (list): List of object groups to sample target objects from.

        exclude_obj_groups (list): Object groups to exclude from sampling target objects.

        handedness (str): Which hand to optimize object spawning for ("right" or "left").
    """

    VALID_LAYOUTS = [0]
    NUM_OBJECTS = 3

    def __init__(
        self,
        obj_groups=None,
        exclude_obj_groups=None,
        container_type="container",
        handedness="right",
        *args,
        **kwargs,
    ):
        if handedness is not None and handedness not in ("right", "left"):
            raise ValueError("handedness must be 'right' or 'left'")
        self.container_type = container_type
        self.obj_groups = obj_groups or ["vegetable", "fruit", "can"]
        self.exclude_obj_groups = exclude_obj_groups
        self.handedness = handedness
        super().__init__(*args, **kwargs)

    def _setup_table_references(self):
        super()._setup_table_references()

        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, size=(0.5, 0.5))
        )
        self.init_robot_base_pos = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        container_lang = self.get_obj_lang(obj_name="container")
        obj_langs = [
            self.get_obj_lang(obj_name=f"obj_{i}") for i in range(self.NUM_OBJECTS)
        ]
        if len(obj_langs) == 1:
            ep_meta["lang"] = f"put the {obj_langs[0]} into the {container_lang}"
        else:
            instruction = f"put the {obj_langs[0]}"
            for obj in obj_langs[1:-1]:
                instruction += f", then the {obj}"
            instruction += (
                f", and finally the {obj_langs[-1]} into the {container_lang}"
            )
            ep_meta["lang"] = instruction
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        handedness = (
            self.handedness
            if self.handedness is not None
            else self.rng.choice(["left", "right"])
        )

        # Add container
        cfgs.append(
            dict(
                name="container",
                obj_groups=self.container_type,
                placement=dict(
                    fixture=self.counter,
                    size=(0.5, 0.5),
                    pos=((0, -0.5)),
                ),
            )
        )

        # Add objects to manipulate
        for i in range(self.NUM_OBJECTS):
            cfgs.append(
                dict(
                    name=f"obj_{i}",
                    obj_groups=self.obj_groups[i % len(self.obj_groups)],
                    exclude_obj_groups=self.exclude_obj_groups,
                    graspable=True,
                    placement=dict(
                        fixture=self.counter,
                        size=(0.3, 0.3),
                        pos=(
                            (0.5, -1) if handedness == "right" else (-0.5, -1)
                        ),  # place object in right front corner if right handed
                    ),
                )
            )
        return cfgs

    def _check_success(self):
        # Check if all objects are in the container
        all_objects_in_container = True
        for i in range(self.NUM_OBJECTS):
            if not OU.check_obj_in_receptacle(self, f"obj_{i}", "container"):
                all_objects_in_container = False
                break

        gripper_empty = all(
            OU.gripper_obj_far(self, obj_name=f"obj_{i}")
            for i in range(self.NUM_OBJECTS)
        )

        return all_objects_in_container and gripper_empty

    def get_object(self):
        objects = dict()
        for i in range(self.NUM_OBJECTS):
            objects[f"obj_{i}"] = dict(
                obj_name=self.objects[f"obj_{i}"].root_body,
                obj_type="body",
                obj_joint=None,
            )
        objects["container"] = dict(
            obj_name=self.objects["container"].root_body,
            obj_type="body",
            obj_joint=None,
        )
        return objects

    def get_subtask_term_signals(self):
        # TODO: support both hands
        signals = dict()
        prev_signals = getattr(self, "_prev_signals", {})
        prev_obj_in_container = True  # First object has no prerequisites
        for i in range(self.NUM_OBJECTS):
            # Grasp signal can only be true if previous object is in container
            grasp_check = self._check_grasp(
                gripper=self.robots[0].gripper["right"],
                object_geoms=self.objects[f"obj_{i}"],
            )
            grasp_signal = f"grasp_obj_{i}"
            signals[grasp_signal] = int(grasp_check and prev_obj_in_container)

            # Container signal can only be true if object is grasped and previous object is in container
            in_container = OU.check_obj_in_receptacle(self, f"obj_{i}", "container")
            container_signal = f"obj_{i}_in_container"
            signals[container_signal] = int(in_container and prev_obj_in_container)

            # Update prerequisite for next iteration
            prev_obj_in_container = signals[container_signal]

        # Store signals for next iteration
        self._prev_signals = signals.copy()
        return signals

    @staticmethod
    def task_config():
        task_spec_0 = {}
        for i in range(PutAllObjectsInContainer.NUM_OBJECTS):
            # Subtask to grasp object i
            task_spec_0[f"subtask_{2*i+1}"] = dict(
                object_ref=f"obj_{i}",
                subtask_term_signal=f"grasp_obj_{i}",
                subtask_term_offset_range=(5, 10),
                selection_strategy="random",
                selection_strategy_kwargs=None,
                action_noise=0.05,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=True,
            )
            # Subtask to move object i to container
            task_spec_0[f"subtask_{2*i+2}"] = dict(
                object_ref="container",
                subtask_term_signal=f"obj_{i}_in_container",
                subtask_term_offset_range=None,
                selection_strategy="random",
                selection_strategy_kwargs=None,
                action_noise=0.05,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=True,
            )

        # Set final subtask's termination signal to 0
        task_spec_0[f"subtask_{2*PutAllObjectsInContainer.NUM_OBJECTS}"][
            "subtask_term_signal"
        ] = None

        return {
            "task_spec_0": task_spec_0,
            "task_spec_1": {
                "subtask_1": dict(
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
            },
        }


class PutAllObjectsInBasket(PutAllObjectsInContainer):
    """
    Class for tasks where multiple objects need to be placed in a basket.
    Objects are randomly spawned on the counter and need to be placed in the basket.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(container_type="basket", *args, **kwargs)


class PutAllObjectsOnPlate(PutAllObjectsInContainer):
    """
    Class for tasks where multiple objects need to be placed on a plate.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(container_type="plate", *args, **kwargs)
