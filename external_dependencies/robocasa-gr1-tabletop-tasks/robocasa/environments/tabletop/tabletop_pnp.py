from robocasa.environments.tabletop.tabletop import *
from robocasa.utils.dexmg_utils import DexMGConfigHelper
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import string_to_array


# Default distractor configuration for TabletopPnP environment
DEFAULT_DISTRACTOR_CONFIG = {
    "regions": {
        "back_edge": {
            "placement": {
                "pos": (0, 1),
                "size": (1.2, 0.4),
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
                    "count": (1, 2),
                },
            ],
            "objects": [],
        }
    }
}


class TabletopPnP(Tabletop, DexMGConfigHelper):
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

    VALID_LAYOUTS = [0]

    def __init__(
        self,
        obj_groups="all",
        exclude_obj_groups=None,
        source_container=None,
        source_container_size=(0.5, 0.5),
        target_container=None,
        target_container_size=(0.5, 0.5),
        distractor_config=DEFAULT_DISTRACTOR_CONFIG,
        use_distractors=True,
        handedness="right",
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
        self.exclude_obj_groups = exclude_obj_groups
        self.handedness = handedness

        super().__init__(
            distractor_config=distractor_config,
            use_distractors=use_distractors,
            *args,
            **kwargs,
        )

    def _setup_table_references(self):
        super()._setup_table_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, size=(0.45, 0.55))
        )
        self.init_robot_base_pos = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        source_container_lang = (
            self.get_obj_lang(obj_name="obj_container")
            if self.source_container
            else "counter"
        )
        target_container_lang = (
            self.get_obj_lang(obj_name="container")
            if self.target_container
            else "counter"
        )
        ep_meta[
            "lang"
        ] = f"pick the {obj_lang} from the {source_container_lang} and place it in the {target_container_lang}"
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        if self.target_container:
            cfgs.append(
                dict(
                    name="container",
                    obj_groups=self.target_container,
                    placement=dict(
                        fixture=self.counter,
                        size=self.target_container_size,
                        pos=(0.9, -0.3) if self.handedness == "right" else (-0.9, -0.3),
                    ),
                )
            )

        # Randomize handedness per invoke rather than storing it
        handedness = (
            self.handedness if self.handedness else self.rng.choice(["left", "right"])
        )
        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(
                        self.source_container_size
                        if self.source_container
                        else (0.3, 0.3)
                    ),
                    pos=(0.5, -0.8) if handedness == "right" else (-0.5, -0.8),
                    try_to_place_in=self.source_container,
                ),
            )
        )
        return cfgs

    def _check_success(self):
        if self.target_container:
            gripper_container_far = OU.any_gripper_obj_far(self, obj_name="container")
            gripper_obj_far = OU.any_gripper_obj_far(self, obj_name="obj")
            obj_in_container = OU.check_obj_in_receptacle(self, "obj", "container")
            obj_on_counter = OU.check_obj_fixture_contact(self, "obj", self.counter)
            container_upright = OU.check_obj_upright(self, "container", threshold=0.8)
            return (
                gripper_container_far
                and gripper_obj_far
                and obj_in_container
                and not obj_on_counter
                and container_upright
            )
        else:
            gripper_obj_far = OU.any_gripper_obj_far(self, obj_name="obj")
            obj_on_counter = OU.check_obj_fixture_contact(self, "obj", self.counter)
            return gripper_obj_far and obj_on_counter

    def get_object(self):
        objects = dict()
        objects["obj"] = dict(
            obj_name=self.objects["obj"].root_body, obj_type="body", obj_joint=None
        )

        objects["container"] = dict(
            obj_name=self.objects["container"].root_body,
            obj_type="body",
            obj_joint=None,
        )
        return objects

    def get_subtask_term_signals(self):
        signals = dict()
        signals["grasp_object"] = int(
            self._check_grasp(
                gripper=self.robots[0].gripper["right"],
                object_geoms=self.objects["obj"],
            )
        )
        return signals

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = dict(
            selection_object_ref="container",
            object_ref="obj",
            subtask_term_signal="grasp_object",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=True,
        )
        task.task_spec_0.subtask_2 = dict(
            object_ref="container",
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

    def visualize_spawn_region(self, obj_name="container"):
        """
        Draws a single site representing the full spawn region as a 3D box, cylinder, or sphere.
        """
        import robocasa.utils.object_utils as OU
        import numpy as np
        import mujoco

        highest_spawn_region = OU.get_highest_spawn_region(self, self.objects[obj_name])

        for spawn in self.objects[obj_name].spawns:
            if spawn != highest_spawn_region:
                continue

            # Get spawn region parameters
            region_points = OU.calculate_spawn_region(self, spawn)
            spawn_type = spawn.get("type", "box")

            # Create or update geom
            self.viewer.update()
            current_geom = None
            for geom_idx in range(self.viewer.viewer.user_scn.ngeom):
                if self.viewer.viewer.user_scn.geoms[geom_idx].label == spawn.get(
                    "name"
                ):
                    current_geom = self.viewer.viewer.user_scn.geoms[geom_idx]
                    break
            if current_geom is None:
                self.viewer.viewer.user_scn.ngeom += 1
                current_geom = self.viewer.viewer.user_scn.geoms[
                    self.viewer.viewer.user_scn.ngeom - 1
                ]

            if spawn_type == "box":
                p0, px, py, pz = region_points
                v_x = px - p0
                v_y = py - p0
                v_z = pz - p0
                center = p0 + 0.5 * (v_x + v_y + v_z)
                half_size = np.array(
                    [
                        np.linalg.norm(v_x) / 2.0,
                        np.linalg.norm(v_y) / 2.0,
                        np.linalg.norm(v_z) / 2.0,
                    ]
                )
                R = np.column_stack(
                    (
                        v_x / np.linalg.norm(v_x),
                        v_y / np.linalg.norm(v_y),
                        v_z / np.linalg.norm(v_z),
                    )
                )
                geom_type = mujoco.mjtGeom.mjGEOM_BOX

            elif spawn_type == "cylinder":
                p0, axis_vector, radius = region_points
                height = np.linalg.norm(axis_vector)
                center = p0 + 0.5 * axis_vector
                half_size = np.array([radius, height / 2, radius])
                z_axis = axis_vector / height
                x_axis = (
                    np.array([1, 0, 0]) if abs(z_axis[1]) < 0.9 else np.array([0, 1, 0])
                )
                y_axis = np.cross(z_axis, x_axis)
                x_axis = np.cross(y_axis, z_axis)
                R = np.column_stack((x_axis, y_axis, z_axis))
                geom_type = mujoco.mjtGeom.mjGEOM_CYLINDER

            elif spawn_type == "sphere":
                center, radius = region_points
                half_size = np.array([radius, radius, radius])
                R = np.eye(3)
                geom_type = mujoco.mjtGeom.mjGEOM_SPHERE

            else:
                raise ValueError(f"Invalid spawn type: {spawn_type}")

            mujoco.mjv_initGeom(
                current_geom,
                type=geom_type,
                size=half_size,
                pos=center,
                mat=R.reshape(9, 1),
                rgba=np.array([0, 1, 0, 0.8]),
            )
            current_geom.label = spawn.get("name")

    def visualize_bounding_box(self, obj_name="obj"):
        """
        Draws a single site representing the full container spawn region as a 3D box.
        """
        import numpy as np
        import robocasa.utils.transform_utils as T

        obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])
        obj_quat = T.convert_quat(
            np.array(self.sim.data.body_xquat[self.obj_body_id[obj_name]]), to="xyzw"
        )

        bbox_points = self.objects[obj_name].get_bbox_points(
            trans=obj_pos, rot=obj_quat
        )

        p0, px, py, pz = bbox_points[:4]

        v_x = px - p0
        v_y = py - p0
        v_z = pz - p0
        center = p0 + 0.5 * (v_x + v_y + v_z)
        half_extent_x = np.linalg.norm(v_x) / 2.0
        half_extent_y = np.linalg.norm(v_y) / 2.0
        half_extent_z = np.linalg.norm(v_z) / 2.0
        half_size = np.array([half_extent_x, half_extent_y, half_extent_z])

        x_axis = v_x / np.linalg.norm(v_x)
        y_axis = v_y / np.linalg.norm(v_y)
        z_axis = v_z / np.linalg.norm(v_z)

        R = np.column_stack((x_axis, y_axis, z_axis))

        import mujoco

        self.viewer.update()
        current_geom = None
        for geom_idx in range(self.viewer.viewer.user_scn.ngeom):
            if self.viewer.viewer.user_scn.geoms[geom_idx].label == obj_name:
                current_geom = self.viewer.viewer.user_scn.geoms[geom_idx]
                break
        if current_geom is None:
            self.viewer.viewer.user_scn.ngeom += 1
            current_geom = self.viewer.viewer.user_scn.geoms[
                self.viewer.viewer.user_scn.ngeom - 1
            ]
        mujoco.mjv_initGeom(
            current_geom,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=half_size,
            pos=center,
            mat=R.reshape(9, 1),
            rgba=np.array([0, 0, 1, 0.8]),
        )
        current_geom.label = obj_name


class PnPOnionToBowl(TabletopPnP):
    """
    Class encapsulating the atomic onion to bowl pick and place task.
    A deliberately simple task used for testing.

    Onion is chosen as it's easy to grasp and place.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(obj_groups="onion", target_container="bowl", *args, **kwargs)


class PnPCanToBowl(TabletopPnP):
    """
    Class encapsulating the atomic can to bowl pick and place task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(obj_groups="can", target_container="bowl", *args, **kwargs)


class PnPCupToPlate(TabletopPnP):
    """
    Class encapsulating the atomic cup to plate pick and place task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            obj_groups="cup",
            target_container="plate",
            distractor_config={
                "regions": {
                    "back_edge": {
                        "placement": {
                            "pos": (0, 1),
                            "size": (1.2, 0.5),
                        },
                        "objects": [
                            {
                                "type": "tiered_basket",
                                "count": (0, 1),
                            },
                            {
                                "type": "basket",
                                "count": 1,
                            },
                            {
                                "type": "mug_tree",
                                "count": 1,
                            },
                        ],
                    }
                }
            },
            *args,
            **kwargs,
        )


class PnPAppleToPlate(TabletopPnP):
    """
    Class encapsulating the atomic apple to plate pick and place task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            obj_groups="apple",
            target_container="plate",
            distractor_config={
                "regions": {
                    "back_edge": {
                        "placement": {
                            "pos": (0, 1),
                            "size": (1.2, 0.6),
                        },
                        "objects": [
                            {
                                "type": "bowl",
                                "count": 1,
                            },
                            {
                                "type": "vegetable",
                                "count": (1, 2),
                            },
                            {
                                "type": "mug",
                                "count": (0, 1),
                            },
                            {
                                "type": "clock",
                                "count": 1,
                            },
                            {
                                "type": "coffee_pod",
                                "count": 1,
                            },
                            {
                                "type": "kettle_non_electric",
                                "count": 1,
                            },
                        ],
                    }
                }
            },
            *args,
            **kwargs,
        )


class PnPMilkToBasket(TabletopPnP):
    """
    Class encapsulating the atomic milk to basket pick and place task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            obj_groups="milk",
            target_container="basket",
            distractor_config={
                "regions": {
                    "back_edge": {
                        "placement": {
                            "pos": (0, 1),
                            "size": (1.2, 0.5),
                        },
                        "objects": [
                            {
                                "type": "fruit",
                                "count": (2, 3),
                            },
                            {
                                "type": "clock",
                                "count": (1, 2),
                            },
                            {
                                "type": "cup",
                                "count": (1, 2),
                            },
                            {
                                "type": "tiered_basket",
                                "count": 1,
                            },
                        ],
                    }
                }
            },
            use_distractors=True,
            *args,
            **kwargs,
        )


class PnPKettleToPlate(TabletopPnP):
    """
    Class encapsulating the atomic kettle to plate pick and place task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            obj_groups="kettle_non_electric",
            target_container="plate",
            distractor_config=None,
            use_distractors=False,
            *args,
            **kwargs,
        )


class PnPFruitToPlacemat(TabletopPnP):
    """
    Class encapsulating the atomic fruit to mat pick and place task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            obj_groups="fruit",
            target_container="placemat",
            target_container_size=(0.7, 0.7),
            distractor_config={
                "regions": {
                    "back_edge": {
                        "placement": {
                            "pos": (0, 1),
                            "size": (1.2, 0.4),
                        },
                        "objects": [
                            {
                                "type": "basket",
                                "count": (1, 2),
                            }
                        ],
                    },
                    "center": {
                        "placement": {
                            "pos": (0, 0.3),
                            "size": (0.8, 0.4),
                        },
                        "objects": [
                            {
                                "type": "vegetable",
                                "count": (2, 3),
                            },
                            {
                                "type": "milk",
                                "count": 1,
                            },
                        ],
                    },
                }
            },
            *args,
            **kwargs,
        )


class PnPCounterToPlate(TabletopPnP):
    """
    Class encapsulating the atomic counter to plate pick and place task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(target_container="plate", *args, **kwargs)


class PnPCounterToBowl(TabletopPnP):
    """
    Class encapsulating the atomic counter to bowl pick and place task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(target_container="bowl", *args, **kwargs)


class PnPCounterToCuttingBoard(TabletopPnP):
    """
    Class encapsulating the atomic counter to cutting board pick and place task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(target_container="cutting_board", *args, **kwargs)


class PnPCounterToPot(TabletopPnP):
    """
    Class encapsulating the atomic counter to pot pick and place task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            target_container="pot", target_container_size=(0.7, 0.7), *args, **kwargs
        )


class PnPCounterToPan(TabletopPnP):
    """
    Class encapsulating the atomic counter to pan pick and place task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(target_container="pan", *args, **kwargs)


class PnPPlateToPlate(TabletopPnP):
    """
    Class encapsulating the atomic plate to plate pick and place task.
    """

    def __init__(
        self,
        obj_groups=["vegetable", "fruit", "dairy"],
        *args,
        **kwargs,
    ):
        super().__init__(
            source_container="plate",
            target_container="plate",
            distractor_config={
                "regions": {
                    "center": {
                        "placement": {
                            "pos": (0, 0.5),
                            "size": (0.8, 0.8),
                        },
                        "objects": [
                            {"type": "bowl", "count": (1, 2)},
                            {"type": "vegetable", "count": (1, 3)},
                            {"type": "fruit", "count": (1, 3)},
                        ],
                    },
                    "back_edge": {
                        "placement": {
                            "pos": (0, 1),
                            "size": (1.2, 0.6),
                        },
                        "objects": [
                            {"type": "mug_tree", "count": 1},
                            {"type": "basket", "count": (0, 2)},
                        ],
                    },
                }
            },
            use_distractors=True,
            obj_groups=obj_groups,
            *args,
            **kwargs,
        )


class PnPMilkPlateToPlate(TabletopPnP):
    """
    Class encapsulating the atomic milk plate to plate pick and place task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            obj_groups="milk",
            distractor_config={
                "regions": {
                    "center": {
                        "placement": {
                            "pos": (0, 0),
                            "size": (0.6, 0.6),
                        },
                        "objects": [
                            {"type": "bowl", "count": (1, 2)},
                            {"type": "vegetable", "count": (1, 2)},
                            {"type": "fruit", "count": (1, 2)},
                        ],
                    },
                    "back_edge": {
                        "placement": {
                            "pos": (0, 1),
                            "size": (1.2, 0.5),
                        },
                        "objects": [
                            {"type": "mug_tree", "count": 1},
                            {"type": "basket", "count": (0, 2)},
                        ],
                    },
                }
            },
            use_distractors=True,
            source_container="plate",
            target_container="plate",
            *args,
            **kwargs,
        )


class PnPVegetableBowlToPlate(TabletopPnP):
    """
    Class encapsulating the atomic pick and place task where a vegetable is placed in a bowl and then transferred to an empty plate.
    Other plates with
    """

    VALID_LAYOUTS = [0]

    def __init__(self, *args, **kwargs):
        super().__init__(
            target_container="plate",
            source_container="bowl",
            obj_groups="vegetable",
            distractor_config={
                "regions": {
                    "center": {
                        "placement": {
                            "pos": (0, 0),
                            "size": (0.8, 0.8),
                        },
                        "objects": [
                            {"type": "fruit", "count": (1, 3)},
                            {"type": "vegetable", "count": (1, 2)},
                            {"type": "basket", "count": (0, 1)},
                            {"type": "rubix_cube", "count": 1},
                            {"type": "milk", "count": (0, 1)},
                            {"type": "placemat", "count": (0, 1)},
                            {"type": "bowl", "count": (1, 2)},
                        ],
                    }
                }
            },
            use_distractors=True,
            *args,
            **kwargs,
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        veg_lang = self.get_obj_lang(obj_name="obj")
        ep_meta[
            "lang"
        ] = f"pick the {veg_lang} from the bowl and place it on the empty plate"
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        cfgs.append(
            dict(
                name="distractor_milk_1",
                obj_groups="milk",
                placement=dict(
                    fixture=self.counter,
                    size=(0.5, 0.5),
                    pos=(1, 0.5),
                    try_to_place_in="plate",
                ),
            )
        )
        cfgs.append(
            dict(
                name="distractor_rubix_cube_1",
                obj_groups="rubix_cube",
                placement=dict(
                    fixture=self.counter,
                    size=(0.5, 0.5),
                    pos=(-1, 0.5),
                    try_to_place_in="plate",
                ),
            )
        )
        cfgs.append(
            dict(
                name="distractor_bowl_1",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    size=(0.5, 0.5),
                    pos=(0.5, 1),
                    optional=True,
                ),
            )
        )
        cfgs.append(
            dict(
                name="distractor_cup_1",
                obj_groups="cup",
                placement=dict(
                    fixture=self.counter,
                    size=(0.5, 0.5),
                    pos=(-0.5, 1),
                    optional=True,
                ),
            )
        )
        return cfgs


class PnPObjectsToShelf(TabletopPnP):
    """
    Class encapsulating the atomic objects to shelf pick and place task.
    """

    def __init__(
        self,
        obj_groups=["vegetable", "fruit", "dairy"],
        *args,
        **kwargs,
    ):
        super().__init__(
            target_container="tiered_shelf",
            target_container_size=(0.7, 0.4),
            obj_groups=obj_groups,
            distractor_config=None,
            use_distractors=False,
            *args,
            **kwargs,
        )


class PnPObjectsToShelfLevel(TabletopPnP):
    """
    Class encapsulating the atomic objects to specific shelf level pick and place task.
    """

    def __init__(
        self,
        obj_groups=["vegetable", "fruit", "dairy"],
        *args,
        **kwargs,
    ):
        super().__init__(
            target_container="tiered_shelf",
            target_container_size=(0.7, 0.4),
            obj_groups=obj_groups,
            *args,
            **kwargs,
        )
        self.target_site_id = None

    def _get_obj_cfgs(self):
        # place close to back of table instead of front of table
        cfgs = super()._get_obj_cfgs()
        cfgs[0]["placement"]["pos"] = (
            (0.9, 0.4) if self.handedness == "right" else (-0.9, 0.4)
        )
        return cfgs

    def _load_model(self):
        super()._load_model()
        _, _, ref_obj = self.object_placements["container"]
        self.target_site_id, _ = ref_obj.get_random_spawn(
            self.rng, exclude_disabled=True
        )

    def _check_success(self):
        obj_on_shelf = bool(
            self.objects["container"].closest_spawn_id(self, self.objects["obj"])
            == self.target_site_id
        )
        return obj_on_shelf and super()._check_success()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        source_container_lang = "counter"
        target_container_lang = self.get_obj_lang(obj_name="container")
        ep_meta["lang"] = (
            f"pick the {obj_lang} from the {source_container_lang} and place it "
            f"on the level #{self.target_site_id + 1} of the the {target_container_lang}"
        )
        return ep_meta


class PnPObjectsShelfToCounter(TabletopPnP):
    """
    Class encapsulating the atomic objects from shelf to counter task.
    """

    def __init__(
        self,
        obj_groups=["vegetable", "fruit"],
        *args,
        **kwargs,
    ):
        super().__init__(
            source_container="tiered_shelf",
            source_container_size=(0.7, 0.4),
            obj_groups=obj_groups,
            *args,
            **kwargs,
        )

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        cfgs[0]["placement"]["pos"] = (
            (0.9, 0.4) if self.handedness == "right" else (-0.9, 0.4)
        )
        for c in cfgs:
            if c["name"] == "obj":
                c["placement"]["site_id"] = -1
        return cfgs


class PnPObjectsShelfLevelToLevel(TabletopPnP):
    """
    Class encapsulating the atomic objects from one shelf level to another level task.
    """

    def __init__(
        self,
        obj_groups=["vegetable", "fruit"],
        *args,
        **kwargs,
    ):
        super().__init__(
            source_container="tiered_shelf",
            source_container_size=(0.7, 0.4),
            obj_groups=obj_groups,
            *args,
            **kwargs,
        )
        self.target_site_id = None

    def _get_obj_cfgs(self):
        # put shelf on back of table instead of front of table
        cfgs = super()._get_obj_cfgs()
        cfgs[0]["placement"]["pos"] = (
            (0.9, 0.4) if self.handedness == "right" else (-0.9, 0.4)
        )
        return cfgs

    def _load_model(self):
        super()._load_model()
        _, _, ref_obj = self.object_placements["obj_container"]
        self.target_site_id, _ = ref_obj.get_random_spawn(
            self.rng, exclude_disabled=True
        )

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        for c in cfgs:
            if c["name"] == "obj":
                c["placement"]["site_id"] = -1
        return cfgs

    def _check_success(self):
        gripper_container_far = OU.gripper_obj_far(self, obj_name="obj_container")
        gripper_obj_far = OU.gripper_obj_far(self, obj_name="obj")
        obj_in_container = OU.check_obj_in_receptacle(self, "obj", "obj_container")
        obj_on_shelf = bool(
            self.objects["obj_container"].closest_spawn_id(self, self.objects["obj"])
            == self.target_site_id
        )
        obj_on_counter = OU.check_obj_fixture_contact(self, "obj", self.counter)
        return (
            gripper_container_far
            and gripper_obj_far
            and obj_in_container
            and obj_on_shelf
            and not obj_on_counter
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        source_container_lang = self.get_obj_lang(obj_name="obj_container")
        ep_meta["lang"] = (
            f"pick the {obj_lang} from the {source_container_lang} and place it "
            f"on the level #{self.target_site_id + 1} of the the {source_container_lang}"
        )
        return ep_meta


class PnPObjectsToTieredBasket(TabletopPnP):
    """
    Class encapsulating the atomic objects to tiered basket pick and place task.
    """

    def __init__(
        self,
        obj_groups=["vegetable", "fruit", "dairy"],
        *args,
        **kwargs,
    ):
        super().__init__(
            target_container="tiered_basket",
            obj_groups=obj_groups,
            *args,
            **kwargs,
        )

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        shelf_pos, shelf_rot = _PnpTieredBasketUtils.get_shelf_pose(self)
        for c in cfgs:
            if c["name"] == "container":
                c["placement"]["pos"] = shelf_pos
                c["placement"]["rotation"] = shelf_rot
        return cfgs


class PnPObjectsToTieredBasketLevel(TabletopPnP):
    """
    Class encapsulating the atomic objects to specific tiered basket level pick and place task.
    """

    def __init__(
        self,
        obj_groups=["vegetable", "fruit", "dairy"],
        *args,
        **kwargs,
    ):
        super().__init__(
            target_container="tiered_basket",
            target_container_size=(0.7, 0.5),
            obj_groups=obj_groups,
            *args,
            **kwargs,
        )
        self.target_site_id = None

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        shelf_pos, shelf_rot = _PnpTieredBasketUtils.get_shelf_pose(self)
        for c in cfgs:
            if c["name"] == "container":
                c["placement"]["pos"] = shelf_pos
                c["placement"]["rotation"] = shelf_rot
        return cfgs

    def _load_model(self):
        super()._load_model()
        _, _, ref_obj = self.object_placements["container"]
        self.target_site_id, _ = ref_obj.get_random_spawn(
            self.rng, exclude_disabled=True
        )

    def _check_success(self):
        obj_on_shelf = bool(
            self.objects["container"].closest_spawn_id(self, self.objects["obj"])
            == self.target_site_id
        )
        return obj_on_shelf and super()._check_success()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        source_container_lang = "counter"
        target_container_lang = self.get_obj_lang(obj_name="container")
        ep_meta["lang"] = (
            f"pick the {obj_lang} from the {source_container_lang} and place it "
            f"on the level #{self.target_site_id + 1} of the the {target_container_lang}"
        )
        return ep_meta


class PnPObjectsTieredBasketToCounter(TabletopPnP):
    """
    Class encapsulating the atomic objects from tiered basket to counter task.
    """

    def __init__(
        self,
        obj_groups=["vegetable", "fruit"],
        *args,
        **kwargs,
    ):
        super().__init__(
            source_container="tiered_basket",
            source_container_size=(0.7, 0.5),
            obj_groups=obj_groups,
            *args,
            **kwargs,
        )

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        shelf_pos, shelf_rot = _PnpTieredBasketUtils.get_shelf_pose(self)
        for c in cfgs:
            if c["name"] == "obj":
                c["placement"]["site_id"] = -1
                c["placement"]["pos"] = shelf_pos
                c["placement"]["rotation"] = shelf_rot
        return cfgs


class PnPObjectsTieredBasketLevelToLevel(TabletopPnP):
    """
    Class encapsulating the atomic objects from one tiered basket level to another level task.
    """

    def __init__(
        self,
        obj_groups=["vegetable", "fruit"],
        *args,
        **kwargs,
    ):
        super().__init__(
            source_container="tiered_basket",
            source_container_size=(0.7, 0.5),
            obj_groups=obj_groups,
            *args,
            **kwargs,
        )
        self.target_site_id = None

    def _load_model(self):
        super()._load_model()
        _, _, ref_obj = self.object_placements["obj_container"]
        self.target_site_id, _ = ref_obj.get_random_spawn(
            self.rng, exclude_disabled=True
        )

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        shelf_pos, shelf_rot = _PnpTieredBasketUtils.get_shelf_pose(self)
        for c in cfgs:
            if c["name"] == "obj":
                c["placement"]["site_id"] = -1
                c["placement"]["pos"] = shelf_pos
                c["placement"]["rotation"] = shelf_rot
        return cfgs

    def _check_success(self):
        gripper_container_far = OU.gripper_obj_far(self, obj_name="obj_container")
        gripper_obj_far = OU.gripper_obj_far(self, obj_name="obj")
        obj_in_container = OU.check_obj_in_receptacle(self, "obj", "obj_container")
        obj_on_shelf = bool(
            self.objects["obj_container"].closest_spawn_id(self, self.objects["obj"])
            == self.target_site_id
        )
        obj_on_counter = OU.check_obj_fixture_contact(self, "obj", self.counter)
        return (
            gripper_container_far
            and gripper_obj_far
            and obj_in_container
            and obj_on_shelf
            and not obj_on_counter
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        source_container_lang = self.get_obj_lang(obj_name="obj_container")
        ep_meta["lang"] = (
            f"pick the {obj_lang} from the {source_container_lang} and place it "
            f"on the level #{self.target_site_id + 1} of the the {source_container_lang}"
        )
        return ep_meta


class _PnpTieredBasketUtils:
    """Utils for tired basket tasks."""

    @staticmethod
    def get_shelf_pose(task: TabletopPnP):
        handedness = (
            task.handedness if task.handedness else task.rng.choice(["left", "right"])
        )
        shelf_pos = (0.75, 0.3) if handedness == "right" else (-0.75, 0.3)
        shelf_rotation = (
            (-5 * np.pi / 8, -7 * np.pi / 8)
            if handedness == "right"
            else (-3 * np.pi / 8, -1 * np.pi / 8)
        )
        return shelf_pos, shelf_rotation


class PnPRubixCubeBasketToCounter(TabletopPnP):
    """
    Class encapsulating the atomic basket to counter pick and place task.
    AKA taking objects out of the basket
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            source_container="basket",
            obj_groups="rubix_cube",
            distractor_config={
                "regions": {
                    "back_edge": {
                        "placement": {
                            "pos": (0, 1),
                            "size": (1.2, 0.5),
                        },
                        "objects": [
                            {
                                "type": "plate",
                                "count": 1,
                            },
                            {
                                "type": "tiered_basket",
                                "count": (0, 1),
                            },
                            {
                                "type": "tiered_shelf",
                                "count": (0, 1),
                            },
                        ],
                    }
                }
            },
            use_distractors=True,
            *args,
            **kwargs,
        )

    def get_object(self):
        objects = dict()
        objects["obj"] = dict(
            obj_name=self.objects["obj"].root_body, obj_type="body", obj_joint=None
        )

        objects["obj_container"] = dict(
            obj_name=self.objects["obj_container"].root_body,
            obj_type="body",
            obj_joint=None,
        )
        return objects

    def get_subtask_term_signals(self):
        signals = dict()
        signals["grasp_object"] = int(
            self._check_grasp(
                gripper=self.robots[0].gripper["right"],
                object_geoms=self.objects["obj"],
            )
        )
        return signals

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = dict(
            object_ref="obj",
            subtask_term_signal="grasp_object",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=True,
        )
        task.task_spec_0.subtask_2 = dict(
            object_ref="obj_container",
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


class PnPCupToPlateNoDistractors(TabletopPnP):
    """
    Class encapsulating the atomic cup to plate pick and place task without distractors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            obj_groups="cup",
            target_container="plate",
            distractor_config=None,
            use_distractors=False,
            source_container_size=(0.5, 0.3),
            target_container_size=(0.5, 0.3),
            *args,
            **kwargs,
        )


class PnPCupToDishRackUpperLevel(TabletopPnP):
    """
    Class encapsulating the atomic cup to dish rack pick and place task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            obj_groups=["cup", "milk"],
            target_container=[
                "objects/sketchfab/dish_rack/dish_rack_0/model.xml",
                "objects/sketchfab/dish_rack/dish_rack_4/model.xml",
            ],
            distractor_config=None,
            use_distractors=False,
            source_container_size=(0.5, 0.3),
            target_container_size=(0.7, 0.5),
            *args,
            **kwargs,
        )
        self.target_site_id = 1

    def _get_obj_cfgs(self):
        # place close to back of table instead of front of table
        cfgs = super()._get_obj_cfgs()
        # container
        cfgs[0]["placement"] = dict(
            fixture=self.counter,
            size=(0.9, 0.4),
            pos=(1.0, -1.0),
            rotation=(-np.pi, np.pi),
        )
        # object
        cfgs[1]["placement"] = dict(
            fixture=self.counter,
            size=(0.6, 0.2),
            pos=(1.0, -1.0),
        )
        return cfgs

    def _load_model(self):
        super()._load_model()
        _, _, ref_obj = self.object_placements["container"]

        # robot0_cotrainview=dict(
        #     pos=[0.22, 0.05, 0.0425],
        #     quat=[0.67397475,0.21391128, -0.21391128, -0.67397475],
        #     camera_attribs=dict(fovy="90"),
        #     parent_body="robot0_head_pitch",
        # ),
        camera = find_elements(
            root=self.model.worldbody,
            tags="camera",
            attribs={"name": "robot0_cotrainview"},
        )
        if camera is None:
            camera = ET.Element("camera")
            camera.set("mode", "fixed")
            camera.set("name", "robot0_cotrainview")
            camera.set("pos", "0.22 0.05 0.0425")
            camera.set("xyaxes", "-0.01 -1. 0. 0.85 0. 0.5")
            camera.set("fovy", "90")
            cam_root = find_elements(
                root=self.model.worldbody,
                tags="body",
                attribs={"name": "robot0_head_pitch"},
            )
            cam_root.append(camera)

    def compute_robot_base_placement_pose(self, offset=None):
        robot_base_pos, robot_base_ori = super().compute_robot_base_placement_pose(
            offset
        )
        # Randomization for the table
        robot_base_pos += np.array([0.05, 0.05, 0])
        return robot_base_pos, robot_base_ori

    def _check_success(self):
        obj_on_shelf = OU.check_obj_in_site(
            self, "obj", "container", self.target_site_id
        )
        return obj_on_shelf and super()._check_success()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        target_container_lang = self.get_obj_lang(obj_name="container")
        ep_meta[
            "lang"
        ] = f"pick the {obj_lang} from the {target_container_lang} and place it on the level #{self.target_site_id + 1} of the the {target_container_lang}"
        return ep_meta


class PnPBreadBasketToBowl(TabletopPnP):
    """
    Class encapsulating the atomic pick and place task where a piece of bread in a basket is then moved to a bowl.
    """

    def __init__(self, *args, **kwargs):
        kwargs["style_ids"] = [4]
        super().__init__(
            obj_groups=[
                "objects/objaverse/bread/bread_1/model.xml",
                "objects/objaverse/bread/bread_17/model.xml",
                "objects/objaverse/bread/bread_20/model.xml",
                "objects/objaverse/bread/bread_21/model.xml",
            ],
            source_container="basket",
            target_container="bowl",
            distractor_config=None,
            use_distractors=False,
            source_container_size=(0.5, 0.1),
            target_container_size=(0.5, 0.1),
            *args,
            **kwargs,
        )

    def _get_obj_cfgs(self):
        cfgs = []

        if self.target_container:
            cfgs.append(
                dict(
                    name="container",
                    obj_groups=self.target_container,
                    placement=dict(
                        fixture=self.counter,
                        size=self.target_container_size,
                        inner_margin=0.1,
                        ensure_object_boundary_in_range=False,
                        pos=(0.5, -0.9) if self.handedness == "right" else (-0.5, -0.9),
                    ),
                )
            )

        # Randomize handedness per invoke rather than storing it
        handedness = (
            self.handedness if self.handedness else self.rng.choice(["left", "right"])
        )
        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(
                        self.source_container_size
                        if self.source_container
                        else (0.3, 0.3)
                    ),
                    ensure_object_boundary_in_range=False,
                    pos=(0.6, -0.9) if handedness == "right" else (-0.6, -0.9),
                    try_to_place_in=self.source_container,
                ),
            )
        )
        return cfgs


class PnPPouring(TabletopPnP):
    def __init__(self, *args, **kwargs):
        super().__init__(
            obj_groups=["primitive"],
            source_container="cup",
            target_container="bowl",
            distractor_config=None,
            use_distractors=False,
            *args,
            **kwargs,
        )

    def _get_obj_cfgs(self):
        cfgs = []

        if self.target_container:
            cfgs.append(
                dict(
                    name="container",
                    obj_groups=self.target_container,
                    placement=dict(
                        fixture=self.counter,
                        size=(0.8, 0.4),
                        pos=(0.2, -0.8),
                    ),
                )
            )
        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(0.5, 0.2),
                    pos=(0.4, -0.9),
                    try_to_place_in=self.source_container,
                ),
            )
        )
        return cfgs

    def _load_model(self):
        super()._load_model()
        camera = find_elements(
            root=self.model.worldbody,
            tags="camera",
            attribs={"name": "robot0_cotrainview"},
        )
        if camera is None:
            camera = ET.Element("camera")
            camera.set("mode", "fixed")
            camera.set("name", "robot0_cotrainview")
            camera.set("pos", "0.22 0.05 0.0425")
            camera.set("xyaxes", "-0.01 -1. 0. 0.85 0. 0.5")
            camera.set("fovy", "90")
            cam_root = find_elements(
                root=self.model.worldbody,
                tags="body",
                attribs={"name": "robot0_head_pitch"},
            )
            cam_root.append(camera)

    def compute_robot_base_placement_pose(self, offset=None):
        robot_base_pos, robot_base_ori = super().compute_robot_base_placement_pose(
            offset
        )
        # Randomization for the table
        robot_base_pos -= self.rng.uniform(0, 0.05, 3)
        return robot_base_pos, robot_base_ori

    def _create_objects(self):
        """
        Creates and places objects in the tabletop environment.
        Helper function called by _load_model()
        """
        # add objects
        self.objects = {}
        self.object_cfgs = self._get_obj_cfgs()
        num_objects = len(self.object_cfgs)
        exclude_cat = []
        self.object_cfgs.extend(self._get_distractor_obj_cfgs())
        addl_obj_cfgs = []
        for obj_num, cfg in enumerate(self.object_cfgs):
            cfg["type"] = "object"
            if "name" not in cfg:
                cfg["name"] = "obj_{}".format(obj_num + 1)
            if obj_num >= num_objects:
                # objects part of distractors
                cfg["exclude_cat"] = exclude_cat

            if cfg["name"] == "obj":
                model = BallObject(
                    name="ball_obj",
                    size=[0.02],
                    density=50.0,
                    friction=None,
                    rgba=self.rng.uniform(0, 1, size=3).tolist() + [1.0],
                )
                cfg["info"] = {
                    "groups_containing_sampled_obj": [
                        "all",
                        "peach",
                        "fruit",
                        "food",
                        "in_container",
                    ],
                    "groups": ["ball"],
                    "cat": "primitive",
                }
                self.objects["obj"] = model
                self.model.merge_objects([model])
            else:
                model, info = self._create_obj(cfg)
                if obj_num < num_objects:
                    # objects part of the task
                    exclude_cat.append(info["cat"])
                cfg["info"] = info
                self.objects[model.name] = model
                self.model.merge_objects([model])

            try_to_place_in = cfg["placement"].get("try_to_place_in", None)

            # place object in a container and add container as an object to the scene
            if try_to_place_in and (
                "in_container" in cfg["info"]["groups_containing_sampled_obj"]
            ):
                container_cfg = {
                    "name": cfg["name"] + "_container",
                    "obj_groups": cfg["placement"].get("try_to_place_in"),
                    "placement": deepcopy(cfg["placement"]),
                    "type": "object",
                }

                container_kwargs = cfg["placement"].get("container_kwargs", None)
                if container_kwargs is not None:
                    for k, v in container_kwargs.items():
                        container_cfg[k] = v

                # add in the new object to the model
                addl_obj_cfgs.append(container_cfg)
                model, info = self._create_obj(container_cfg)
                container_cfg["info"] = info
                self.objects[model.name] = model
                self.model.merge_objects([model])

                reference = container_cfg["name"]
                site_id = cfg["placement"].get("site_id", None)
                if site_id is not None:
                    reference = (reference, site_id)

                # modify object config to lie inside of container
                cfg["placement"] = dict(
                    size=(0.01, 0.01),
                    ensure_object_boundary_in_range=False,
                    sample_args=dict(
                        reference=reference,
                    ),
                )

        # place the additional objects (usually containers) in first
        self.object_cfgs = addl_obj_cfgs + self.object_cfgs

        # # remove objects that didn't get created
        # self.object_cfgs = [cfg for cfg in self.object_cfgs if "model" in cfg]

    def get_object(self):
        objects = dict()
        objects["obj"] = dict(
            obj_name=self.objects["obj"].root_body, obj_type="BODY", obj_joint=None
        )

        objects["obj_container"] = dict(
            obj_name=self.objects["obj_container"].root_body,
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
        signals = dict()
        signals["grasp_obj_container"] = int(
            self._check_grasp(
                gripper=self.robots[0].gripper["right"],
                object_geoms=self.objects["obj_container"],
            )
        )
        return signals

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = dict(
            object_ref="obj_container",
            subtask_term_signal="grasp_obj_container",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=True,
        )
        task.task_spec_0.subtask_2 = dict(
            object_ref="container",
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


class PnPFruitToPlate(TabletopPnP):
    def __init__(self, *args, **kwargs):
        super().__init__(
            obj_groups=["vegetable", "fruit"],
            target_container="plate",
            source_container_size=(0.5, 0.3),
            target_container_size=(0.5, 0.3),
            distractor_config={
                "regions": {
                    "back_edge": {
                        "placement": {
                            "pos": (0, 1),
                            "size": (1.2, 0.4),
                        },
                        "objects": [
                            {
                                "type": "basket",
                                "count": (1, 2),
                            }
                        ],
                    },
                    "center": {
                        "placement": {
                            "pos": (0.5, -0.9),
                            "size": (0.3, 0.3),
                        },
                        "objects": [
                            {
                                "type": ["vegetable", "fruit"],
                                "count": (2, 4),
                            },
                            {
                                "type": "milk",
                                "count": 1,
                            },
                        ],
                    },
                }
            },
            *args,
            **kwargs,
        )

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        cfgs[0]["placement"]["size"] = (0.7, 0.7)
        return cfgs

    def _create_objects(self):
        """
        Creates and places objects in the tabletop environment.
        Helper function called by _load_model()
        """
        # add objects
        self.objects = {}
        exclude_cat = []
        if "object_cfgs" in self._ep_meta:
            self.object_cfgs = self._ep_meta["object_cfgs"]
            for obj_num, cfg in enumerate(self.object_cfgs):
                if "name" not in cfg:
                    cfg["name"] = "obj_{}".format(obj_num + 1)
                model, info = self._create_obj(cfg)
                cfg["info"] = info
                self.objects[model.name] = model
                self.model.merge_objects([model])
        else:
            self.object_cfgs = self._get_obj_cfgs()
            unique_obj_num = len(self.object_cfgs)
            self.object_cfgs.extend(self._get_distractor_obj_cfgs())
            addl_obj_cfgs = []
            for obj_num, cfg in enumerate(self.object_cfgs):
                cfg["type"] = "object"
                if "name" not in cfg:
                    cfg["name"] = "obj_{}".format(obj_num + 1)
                cfg["exclude_cat"] = exclude_cat
                model, info = self._create_obj(cfg)
                cfg["info"] = info
                if obj_num < unique_obj_num:
                    exclude_cat.append(info["cat"])
                self.objects[model.name] = model
                self.model.merge_objects([model])

                try_to_place_in = cfg["placement"].get("try_to_place_in", None)

                # place object in a container and add container as an object to the scene
                if try_to_place_in and (
                    "in_container" in cfg["info"]["groups_containing_sampled_obj"]
                ):
                    container_cfg = {
                        "name": cfg["name"] + "_container",
                        "obj_groups": cfg["placement"].get("try_to_place_in"),
                        "placement": deepcopy(cfg["placement"]),
                        "type": "object",
                    }

                    container_kwargs = cfg["placement"].get("container_kwargs", None)
                    if container_kwargs is not None:
                        for k, v in container_kwargs.items():
                            container_cfg[k] = v

                    # add in the new object to the model
                    addl_obj_cfgs.append(container_cfg)
                    model, info = self._create_obj(container_cfg)
                    container_cfg["info"] = info
                    self.objects[model.name] = model
                    self.model.merge_objects([model])

                    # modify object config to lie inside of container
                    cfg["placement"] = dict(
                        size=(0.01, 0.01),
                        ensure_object_boundary_in_range=False,
                        sample_args=dict(
                            reference=container_cfg["name"],
                        ),
                    )

            # place the additional objects (usually containers) in first
            self.object_cfgs = addl_obj_cfgs + self.object_cfgs

            # # remove objects that didn't get created
            # self.object_cfgs = [cfg for cfg in self.object_cfgs if "model" in cfg]


class PnPFruitToPlateSplitA(PnPFruitToPlate):
    def __init__(self, *args, **kwargs):
        kwargs["obj_instance_split"] = "A"
        super().__init__(*args, **kwargs)


class PnPFruitToPlateSplitB(PnPFruitToPlate):
    def __init__(self, *args, **kwargs):
        kwargs["obj_instance_split"] = "B"
        super().__init__(*args, **kwargs)


class PnPCylindricalToPlate(TabletopPnP):
    """
    Class for picking and placing cylindrical objects (bottled_drink, bottled_water, boxed_drink, can, milk) onto a plate.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            target_container="plate",
            obj_groups=["bottled_drink", "bottled_water", "boxed_drink", "can", "milk"],
            distractor_config=None,
            use_distractors=False,
            source_container_size=(0.5, 0.3),
            target_container_size=(0.5, 0.3),
            *args,
            **kwargs,
        )


COTRAIN_REAL_MATCHED_ROBOT_INITIAL_POSE = [
    -0.22963779,
    -0.38363408,
    0.14360377,
    -1.5289252,
    -0.2897802,
    -0.07134621,
    -0.04550289,
    -0.10933163,
    0.43292055,
    -0.15983289,
    -1.48233023,
    0.2359135,
    0.26184522,
    0.00830735,
]


class PnPMilkPlateToPlateCotrain(PnPMilkPlateToPlate):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            {
                "layout_ids": [3],
                "source_container_size": (0.5, 0.3),
                "target_container_size": (0.5, 0.3),
            }
        )
        super().__init__(*args, **kwargs)

    def _load_model(self):
        super()._load_model()
        table = find_elements(
            root=self.model.worldbody,
            tags="body",
            attribs={"name": "table_main_group_main"},
        )
        if "euler" in table.attrib:
            del table.attrib["euler"]
            table.set("xyaxes", "1 0.12 0 0 1 0")

        # robot0_cotrainview=dict(
        #     pos=[0.22, 0.05, 0.0425],
        #     quat=[0.67397475,0.21391128, -0.21391128, -0.67397475],
        #     camera_attribs=dict(fovy="90"),
        #     parent_body="robot0_head_pitch",
        # ),
        camera = find_elements(
            root=self.model.worldbody,
            tags="camera",
            attribs={"name": "robot0_cotrainview"},
        )
        if camera is None:
            camera = ET.Element("camera")
            camera.set("mode", "fixed")
            camera.set("name", "robot0_cotrainview")
            camera.set("pos", "0.22 0.05 0.0425")
            camera.set("xyaxes", "-0.01 -1. 0. 0.85 0. 0.5")
            camera.set("fovy", "90")
            cam_root = find_elements(
                root=self.model.worldbody,
                tags="body",
                attribs={"name": "robot0_head_pitch"},
            )
            cam_root.append(camera)

        # Randomization for the camera
        randomize_range = {
            "xyaxes": (-0.02, 0.02),
            "fov": (-5, 5),
            "pos": (-0.02, 0.02),
        }
        angle_type = "xyaxes"
        camera_orientation = string_to_array(camera.get(angle_type))
        camera_fov = string_to_array(camera.get("fovy"))
        camera_pos = string_to_array(camera.get("pos"))
        camera_orientation += self.rng.uniform(
            randomize_range["xyaxes"][0], randomize_range["xyaxes"][1], 6
        )
        camera_fov += self.rng.uniform(
            randomize_range["fov"][0], randomize_range["fov"][1], 1
        )
        camera_pos += self.rng.uniform(
            randomize_range["pos"][0], randomize_range["pos"][1], 3
        )

        camera.set(angle_type, array_to_string(camera_orientation))
        camera.set("fovy", array_to_string(camera_fov))
        camera.set("pos", array_to_string(camera_pos))

    def compute_robot_base_placement_pose(self, offset=None):
        robot_base_pos, robot_base_ori = super().compute_robot_base_placement_pose(
            offset
        )
        # Randomization for the table
        robot_base_pos += np.array([0.05, 0.05, 0])
        robot_base_pos -= self.rng.uniform(0, 0.05, 3)
        return robot_base_pos, robot_base_ori

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        # Randomization for joints
        joint_rand_strength = 0.2
        rand_joint_names = []
        for name in self.sim.model.joint_names:
            if "robot0_" in name:
                rand_joint_names.append(name)
        joint_ids = [self.sim.model.joint_name2id(name) for name in rand_joint_names]
        joint_pos = np.array(COTRAIN_REAL_MATCHED_ROBOT_INITIAL_POSE)
        new_joint_pos = np.array(joint_pos) + self.rng.uniform(
            -joint_rand_strength, joint_rand_strength, len(joint_pos)
        )
        self.sim.data.qpos[joint_ids] = new_joint_pos
        self.sim.forward()


class PnPAppleToPlateCotrain(PnPAppleToPlate):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            {
                "layout_ids": [3],
                "source_container_size": (0.5, 0.3),
                "target_container_size": (0.5, 0.3),
            }
        )
        super().__init__(*args, **kwargs)

    def _load_model(self):
        super()._load_model()
        table = find_elements(
            root=self.model.worldbody,
            tags="body",
            attribs={"name": "table_main_group_main"},
        )
        if "euler" in table.attrib:
            del table.attrib["euler"]
            table.set("xyaxes", "1 0.12 0 0 1 0")

        # robot0_cotrainview=dict(
        #     pos=[0.22, 0.05, 0.0425],
        #     quat=[0.67397475,0.21391128, -0.21391128, -0.67397475],
        #     camera_attribs=dict(fovy="90"),
        #     parent_body="robot0_head_pitch",
        # ),
        camera = find_elements(
            root=self.model.worldbody,
            tags="camera",
            attribs={"name": "robot0_cotrainview"},
        )
        if camera is None:
            camera = ET.Element("camera")
            camera.set("mode", "fixed")
            camera.set("name", "robot0_cotrainview")
            camera.set("pos", "0.22 0.05 0.0425")
            camera.set("xyaxes", "-0.01 -1. 0. 0.85 0. 0.5")
            camera.set("fovy", "90")
            cam_root = find_elements(
                root=self.model.worldbody,
                tags="body",
                attribs={"name": "robot0_head_pitch"},
            )
            cam_root.append(camera)

        # Randomization for the camera
        randomize_range = {
            "xyaxes": (-0.02, 0.02),
            "fov": (-5, 5),
            "pos": (-0.02, 0.02),
        }
        angle_type = "xyaxes"
        camera_orientation = string_to_array(camera.get(angle_type))
        camera_fov = string_to_array(camera.get("fovy"))
        camera_pos = string_to_array(camera.get("pos"))
        camera_orientation += self.rng.uniform(
            randomize_range["xyaxes"][0], randomize_range["xyaxes"][1], 6
        )
        camera_fov += self.rng.uniform(
            randomize_range["fov"][0], randomize_range["fov"][1], 1
        )
        camera_pos += self.rng.uniform(
            randomize_range["pos"][0], randomize_range["pos"][1], 3
        )

        camera.set(angle_type, array_to_string(camera_orientation))
        camera.set("fovy", array_to_string(camera_fov))
        camera.set("pos", array_to_string(camera_pos))

    def compute_robot_base_placement_pose(self, offset=None):
        robot_base_pos, robot_base_ori = super().compute_robot_base_placement_pose(
            offset
        )
        # Randomization for the table
        robot_base_pos += np.array([0.05, 0.05, 0])
        robot_base_pos -= self.rng.uniform(0, 0.05, 3)
        return robot_base_pos, robot_base_ori

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        # Randomization for joints
        joint_rand_strength = 0.2
        rand_joint_names = []
        for name in self.sim.model.joint_names:
            if "robot0_" in name:
                rand_joint_names.append(name)
        joint_ids = [self.sim.model.joint_name2id(name) for name in rand_joint_names]
        joint_pos = np.array(COTRAIN_REAL_MATCHED_ROBOT_INITIAL_POSE)
        new_joint_pos = np.array(joint_pos) + self.rng.uniform(
            -joint_rand_strength, joint_rand_strength, len(joint_pos)
        )
        self.sim.data.qpos[joint_ids] = new_joint_pos
        self.sim.forward()
