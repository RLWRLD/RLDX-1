from typing import List, Tuple
import numpy as np
from copy import deepcopy
from itertools import combinations, chain
import zlib

import robocasa.utils.object_utils as OU
from robocasa.environments.tabletop.tabletop_pnp import TabletopPnP
from robocasa.models.objects.kitchen_objects import get_all_obj_cats
from robocasa.utils.cotrain_utils import (
    COTRAIN_REAL_MATCHED_ROBOT_INITIAL_POSE,
)
from robocasa.models.fixtures.fixture import FixtureType

# ===============================================================================
#                                 Configurations
# ===============================================================================

SEED = 42
rng = np.random.default_rng(seed=SEED)

# Task Generation Configuration
TASK_CONFIG = {
    "registries": ["objaverse", "sketchfab", "lightwheel"],
    "obj_groups": [
        "vegetable",
        "bread_food",
        "pastry",
        "sweets",
        "fruit",
        "meat",
        "drink",
        "cooked_food",
        "toy",
    ],
    "source_containers": ["cutting_board", "tray", "plate", "placemat"],
    "target_containers": [
        "basket",
        "pan",
        "pot",
        "bowl",
        "plate",
        "tiered_shelf",
        "tiered_basket",
        "cardboard_box",
    ],
    "exclude_combos": [],
    "similar_containers": {
        "tray": ["cutting_board"],
        "cutting_board": ["tray"],
        "basket": ["tiered_basket"],
        "tiered_basket": ["basket", "tiered_shelf"],
        "tiered_shelf": ["tiered_basket"],
        "pot": ["pan"],
        "pan": ["pot"],
    },
}

# Distractor Configurations
RANDOM_DISTRACTOR_CONFIG = {
    "distractor_obj": {  # placing object on the source container
        "placement": {
            "try_to_place_in": "obj_container",
            "use_existing_container": True,
            "ensure_object_in_ref_region": True,
            "ensure_object_boundary_in_range": False,
        },
        "objects": [
            {
                "type": ("obj", "distractor"),
                "count": (1, 1),
                "exclude_groups": ["drink"],
            },
        ],
    },
    "distractor_source_container": {  # placing object on a new container, different from the source container
        "placement": {
            "try_to_place_in": ("source_container", "distractor"),
            "use_existing_container": False,
            "ensure_object_in_ref_region": True,
            "ensure_object_boundary_in_range": False,
            "size": (1.1, 0.45),
            "pos": (0, -0.6),
            "container_kwargs": {
                "exclude_obj_cat": True,
            },
        },
        "objects": [
            {
                "type": ("obj", "any"),
                "count": (1, 1),
                "exclude_groups": ["drink"],
                "exclude_obj_cat": False,
            },
        ],
    },
    "distractor_target_container": {  # a distractor target container
        "placement": {
            "size": (1.1, 0.45),
            "pos": (0, -0.6),
            "ensure_object_boundary_in_range": False,
        },
        "objects": [
            {
                "type": ("target_container", "distractor"),
                "count": (1, 1),
            },
        ],
    },
}

FIXED_DISTRACTOR_CONFIG = {
    "back_edge": {
        "placement": {
            "pos": (0, 0.9),
            "rotation": (np.pi / 2, np.pi / 2),
            "size": (1.0, 0.1),
            "try_not_to_place_in": ["container", "obj_container"],
            "ensure_object_out_of_ref_region": True,
            "ensure_object_boundary_in_range": False,
        },
        "fixtures": [
            {
                "type": ["toaster"],
                "count": (0, 1),
            },
            {
                "type": ["plant"],
                "count": (0, 1),
            },
        ],
        "objects": [
            {
                "type": ["chips", "cereal", "book", "candle"],
                "count": (0, 1),
            },
        ],
    },
}


class PositionSampler:
    """Samples positions for containers and objects based on handedness."""

    _POSITIONS = {
        "right": {
            "container": [(0.75, -0.15)],
            "obj": [(0.85, -0.9)],
        },
        "left": {
            "container": [(-0.7, -0.45), (-0.2, -0.45)],
            "obj": [(-0.2, -0.85), (-0.7, -0.85)],
        },
    }
    _SIZE = {
        "obj": [(0.7, 0.3)],
        "container": [(0.6, 0.2)],
    }

    @staticmethod
    def sample(
        handedness: str, rng: np.random.Generator
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Sample a container and object position pair."""
        if handedness not in ["left", "right"]:
            raise ValueError(f"Handedness must be 'left' or 'right', got {handedness}")

        positions = PositionSampler._POSITIONS[handedness]
        container_positions = positions["container"]
        obj_positions = positions["obj"]

        idx = rng.choice(len(container_positions))
        return (
            container_positions[idx],
            obj_positions[idx],
            PositionSampler._SIZE["container"][idx],
            PositionSampler._SIZE["obj"][idx],
        )

    @staticmethod
    def get_size(key) -> Tuple[float, float]:
        return PositionSampler._SIZE[key]


OBJ_SCALES = {
    "tiered_shelf": 0.7,
    "tiered_basket": 0.7,
    "plate": 1.2,
    "cardboard_box": 0.9,
    "placemat": 0.8,
    "rubix_cube": 0.8,
    "tray": 0.8,
    "cutting_board": 0.8,
}

FIXTURE_SCALES = {
    "paper_towel": (0.3),
    "plant": 0.3,
    "toaster": 1.0,
}

# ===============================================================================
#                             Helper Functions
# ===============================================================================


def construct_distractor_obj_cfgs(
    included_configs: List[str] = None,
    randomize_configs: bool = False,
    task_seed: int = None,
):
    """Construct distractor object configurations."""

    distractor_config = {"regions": {}}
    distractor_config["regions"].update(deepcopy(FIXED_DISTRACTOR_CONFIG))

    task_rng = np.random.default_rng(seed=task_seed if task_seed else SEED)

    for region in distractor_config["regions"]:
        for fixture in distractor_config["regions"][region]["fixtures"]:
            if isinstance(fixture["type"], list):
                fixture["type"] = task_rng.choice(fixture["type"])
                fixture["scale"] = FIXTURE_SCALES.get(fixture["type"], 1.0)
        for obj in distractor_config["regions"][region]["objects"]:
            if isinstance(obj["type"], list):
                obj["type"] = task_rng.choice(obj["type"])
                if "object_scale" in obj:
                    obj["object_scale"] = OBJ_SCALES[obj["type"]]
    if randomize_configs:
        p = 0.4
        # Randomly include items with 0.4 probability
        for item in RANDOM_DISTRACTOR_CONFIG.items():
            if task_rng.random() < p:
                distractor_config["regions"].update({item[0]: item[1]})
    else:
        if included_configs is not None:
            # If not randomizing, include distractor items in included_configs
            for item in RANDOM_DISTRACTOR_CONFIG.items():
                if item[0] in included_configs:
                    distractor_config["regions"].update({item[0]: item[1]})
        else:
            distractor_config["regions"] = {}

    return distractor_config


def is_excluded(
    obj_group: str, source: str, target: str, exclude_list: List[Tuple[str, str, str]]
) -> bool:
    """Check if a combination is excluded."""
    for excluded_obj, excluded_source, excluded_target in exclude_list:
        matches_obj = excluded_obj == "*" or excluded_obj == obj_group
        matches_source = excluded_source == "*" or excluded_source == source
        matches_target = excluded_target == "*" or excluded_target == target
        if matches_obj and matches_source and matches_target:
            return True
    return False


def get_excluded_obj_cats(obj_cats: List[str]) -> List[str]:
    """Get all object categories that are not in val_obj_cats."""
    all_obj_cats = []
    # Get all object categories from all groups
    for group in TASK_CONFIG["obj_groups"]:
        cats = get_all_obj_cats(
            groups=[group], registries=TASK_CONFIG["registries"], attrs=["graspable"]
        )
        if cats:
            all_obj_cats.extend(cats)

    # Remove duplicates (deterministic order for reproducibility across processes)
    all_obj_cats = sorted(set(all_obj_cats))
    return [cat for cat in all_obj_cats if cat not in obj_cats]


def get_excluded_container_combos(
    container_combos: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """Get all valid container combinations that are not in val_container_combos."""
    all_combos = []
    for source in TASK_CONFIG["source_containers"]:
        for target in TASK_CONFIG["target_containers"]:
            if not is_excluded("*", source, target, TASK_CONFIG["exclude_combos"]):
                combo = (source, target)
                if combo not in container_combos:
                    all_combos.append(combo)
    return all_combos


# ===============================================================================
#                             Class Generation
# ===============================================================================

def create_pnp_class(
    class_name: str,
    obj_cat: str,
    source: str,
    target: str,
    all_obj_cats: List[str],
    all_source_containers: List[str],
    all_target_containers: List[str],
    distractor_cfg: dict = None,
    obj_instance_split: str = None,
    distractor_obj_cats: List[str] = None,
) -> type:
    """Creates a pick-and-place class dynamically."""

    if not distractor_obj_cats:
        distractor_obj_cats = all_obj_cats

    def __init__(self, *args, **kwargs):
        if "obj_instance_split" not in kwargs:
            kwargs["obj_instance_split"] = obj_instance_split

        TabletopPnP.__init__(
            self,
            obj_groups=obj_cat,
            source_container=source,
            source_container_size=PositionSampler.get_size("obj"),
            target_container=target,
            target_container_size=PositionSampler.get_size("container"),
            distractor_config=distractor_cfg,
            obj_registries=TASK_CONFIG["registries"],
            *args,
            **kwargs,
        )

    def _create_objects(self):
        """
        Creates and places objects in the tabletop environment.
        Helper function called by _load_model()
        """
        # add objects
        self.objects = {}
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
            num_objects = len(self.object_cfgs)
            exclude_cat = []
            self.object_cfgs.extend(self._get_distractor_obj_cfgs())
            addl_obj_cfgs = []
            invalid_obj_cfgs = []
            # if self.source_container in [
            #     "plate",
            #     "bowl",
            # ]:
            #     cfg["exclude_cat"] += get_all_obj_cats(["drink"])
            for obj_num, cfg in enumerate(self.object_cfgs):
                cfg["type"] = "object"
                if "name" not in cfg:
                    cfg["name"] = "obj_{}".format(obj_num + 1)
                if obj_num >= num_objects and cfg.get("exclude_obj_cat", True):
                    # objects part of distractors
                    cfg["exclude_cat"] = cfg.get("exclude_cat", []) + exclude_cat
                if cfg["obj_groups"] == "obj":
                    # Find the obj config and use its info
                    for obj_cfg in self.object_cfgs:
                        if obj_cfg["name"] == "obj":
                            cfg["obj_groups"] = obj_cfg["info"]["cat"]
                            break
                try:
                    model, info = self._create_obj(cfg)
                except ValueError:
                    # no valid object found
                    # this is due to the object group selected only has 1 object category which is the same as the target object
                    # so we skip this object config
                    print(
                        f"No valid object found for {cfg['name']}, skip this object config"
                    )
                    invalid_obj_cfgs.append(cfg)
                    continue
                if obj_num < num_objects:
                    # objects part of the task
                    exclude_cat.append(info["cat"])
                cfg["info"] = info
                self.objects[model.name] = model
                self.model.merge_objects([model])

                try_to_place_in = cfg["placement"].get("try_to_place_in", None)

                # place object in a container and add container as an object to the scene
                if try_to_place_in:
                    assert (
                        "in_container" in cfg["info"]["groups_containing_sampled_obj"]
                    ), f"Object {cfg['name']} is not part of in_container group"

                    if cfg["placement"].get("use_existing_container", False):
                        # use existing container, no need to create a new one
                        reference = cfg["placement"]["try_to_place_in"]
                    else:
                        container_cfg = {
                            "name": cfg["name"] + "_container",
                            "obj_groups": cfg["placement"].get("try_to_place_in"),
                            "placement": deepcopy(cfg["placement"]),
                            "type": "object",
                        }

                        container_kwargs = cfg["placement"].get(
                            "container_kwargs", None
                        )
                        if container_kwargs is not None:
                            for k, v in container_kwargs.items():
                                container_cfg[k] = v

                        # containers that are part of distractors
                        if obj_num >= num_objects and container_cfg.get(
                            "exclude_obj_cat", True
                        ):
                            container_cfg["exclude_cat"] = (
                                container_cfg.get("exclude_cat", []) + exclude_cat
                            )

                        # add in the new object to the model
                        addl_obj_cfgs.append(container_cfg)
                        model, info = self._create_obj(container_cfg)
                        container_cfg["info"] = info
                        self.objects[model.name] = model
                        self.model.merge_objects([model])

                        if obj_num < num_objects:
                            # container part of the task
                            exclude_cat.append(info["cat"])

                        reference = container_cfg["name"]
                        site_id = cfg["placement"].get("site_id", None)
                        if site_id is not None:
                            reference = (reference, site_id)

                    # modify object config to lie inside of container
                    cfg["placement"] = dict(
                        # pos=(0, 0),
                        size=(0.1, 0.1),
                        ensure_object_boundary_in_range=False,
                        ensure_object_in_ref_region=cfg["placement"].get(
                            "ensure_object_in_ref_region", False
                        ),
                        sample_args=dict(
                            reference=reference,
                        ),
                    )

                if cfg["placement"].get("try_not_to_place_in", None):
                    reference = cfg["placement"]["try_not_to_place_in"]
                    cfg["placement"].update(
                        ensure_object_boundary_in_range=False,
                        ensure_object_out_of_ref_region=cfg["placement"].get(
                            "ensure_object_out_of_ref_region", False
                        ),
                        sample_args=dict(
                            neg_reference=reference,
                        ),
                    )

            # place the additional objects (usually containers) in first

            for cfg in invalid_obj_cfgs:
                self.object_cfgs.remove(cfg)
 
            self.object_cfgs = addl_obj_cfgs + self.object_cfgs

            # # remove objects that didn't get created
            # self.object_cfgs = [cfg for cfg in self.object_cfgs if "model" in cfg]

    def _get_obj_cfgs(self):
        """Get object configurations for the scene."""

        pos_container, pos_obj, size_container, size_obj = PositionSampler.sample(
            self.handedness,
            self.rng,
        )

        return [
            {
                "name": "container",
                "obj_groups": self.target_container,
                "placement": {
                    "fixture": self.counter,
                    "size": size_container,
                    "pos": pos_container,
                    "rotation": (-np.pi / 2, np.pi / 2),
                    "ensure_object_boundary_in_range": False,
                },
                "object_scale": OBJ_SCALES.get(self.target_container, None),
            },
            {
                "name": "obj",
                "obj_groups": self.obj_groups,
                "exclude_obj_groups": self.exclude_obj_groups,
                "graspable": True,
                "placement": {
                    "fixture": self.counter,
                    "size": size_obj if size_obj else (0.3, 0.3),
                    "pos": pos_obj,
                    "rotation": (-np.pi / 2, np.pi / 2),
                    "try_to_place_in": self.source_container,
                    "ensure_object_in_ref_region": True,
                    "ensure_object_boundary_in_range": True,
                },
                "object_scale": OBJ_SCALES,
            },
        ]

    def _get_distractor_obj_cfgs(self):
        """Returns configurations for distractor objects"""

        def _resolve_type_tuple(type_tuple):
            """Helper function to resolve (A, B) type tuples
            A: 'obj', 'source_container', or 'target_container'
            B: 'focus', 'distractor', or 'any'
            """
            category, mode = type_tuple

            if category == "obj":
                pool = distractor_obj_cats.copy()
                focus_item = self.obj_groups
            elif category == "source_container":
                pool = all_source_containers.copy()
                focus_item = self.source_container
            elif category == "target_container":
                pool = all_target_containers.copy()
                focus_item = self.target_container
            else:
                raise ValueError(f"Unknown category type: {category}")

            if mode == "focus":
                return focus_item
            elif mode == "distractor":
                if focus_item in pool:
                    pool.remove(focus_item)
                    for similar_item in TASK_CONFIG["similar_containers"].get(
                        focus_item, []
                    ):
                        if similar_item in pool:
                            pool.remove(similar_item)
                            # print(f"Removed similar item: {similar_item}")
                return pool
            elif mode == "any":
                return pool
            else:
                raise ValueError(f"Unknown mode: {mode}")

        for config in self.distractor_config.get("regions", {}).values():
            # Handle placement configurations
            if config.get("placement", None):
                try_to_place_in = config["placement"].get("try_to_place_in", None)
                if isinstance(try_to_place_in, tuple):
                    config["placement"]["try_to_place_in"] = _resolve_type_tuple(
                        try_to_place_in,
                    )

            # Handle object configurations
            for obj_config in config.get("objects", []):
                obj_type = obj_config.get("type", None)
                if isinstance(obj_type, tuple):
                    resolved_type = _resolve_type_tuple(
                        obj_type,
                    )
                    obj_config["type"] = resolved_type
                    # Set exclude_obj_cat to True if we're dealing with objects
                    if obj_type[0] == "obj":
                        obj_config["exclude_obj_cat"] = True

        # Start processing distractor object configurations
        if self.distractor_config is None:
            return []

        cfgs = []

        counter = self.fixture_refs.get(
            "counter", self.get_fixture(id=FixtureType.COUNTER)
        )

        # Process each region's objects
        for region_name, region_config in self.distractor_config.get(
            "regions", {}
        ).items():
            placement = region_config.get("placement", {})

            for i, obj_config in enumerate(region_config.get("objects", [])):
                count_spec = obj_config.get("count", 1)
                if isinstance(count_spec, (list, tuple)):
                    min_count = count_spec[0]
                    max_count = self.rng.integers(min_count, count_spec[1] + 1)
                else:
                    min_count = max_count = count_spec

                for j in range(max_count):
                    cfgs.append(
                        dict(
                            name=f"distractor_obj_{region_name}_type_{i}_num_{j}",
                            type="object",
                            obj_groups=obj_config["type"],
                            exclude_cat=obj_config.get("exclude_cat", []),
                            graspable=False,
                            exclude_obj_cat=obj_config.get("exclude_obj_cat", True),
                            placement={
                                **placement,
                                "optional": i >= min_count,
                                "fixture": counter,
                            },
                        )
                    )

        return cfgs

    def _check_grasp_distractor_obj(self):
        check_grasp_distractor_obj = False
        if "distractor_obj" in self.distractor_config.get("regions", {}).keys():
            for cfg in self.object_cfgs:
                if "distractor_obj_distractor_obj" in cfg["name"]:
                    distractor = self.objects[cfg["name"]]
                    if self._check_grasp(
                        gripper=self.robots[0].gripper["right"],
                        object_geoms=distractor,
                    ):
                        check_grasp_distractor_obj = True
        return check_grasp_distractor_obj

    def _check_success(self):

        gripper_container_far = OU.any_gripper_obj_far(self, obj_name="container")
        gripper_obj_far = OU.any_gripper_obj_far(self, obj_name="obj")
        if self.target_container in [
            "tiered_basket",
            "tiered_shelf",
        ]:  # currently only these containers have spawn regions
            highest_spawn_region = OU.get_highest_spawn_region(
                self, self.objects["container"]
            )
        else:
            highest_spawn_region = None
        obj_in_container = OU.check_obj_in_receptacle(
            self, "obj", "container", spawn_regions=[highest_spawn_region]
        )
        obj_on_counter = OU.check_obj_fixture_contact(self, "obj", self.counter)
        container_upright = OU.check_obj_upright(self, "container", threshold=0.8)

        # check if the distractor object is in the target container if it has the distractor_obj key
        if "distractor_obj" in self.distractor_config.get("regions", {}).keys():
            distractor_obj_in_receptacle = False
            for cfg in self.object_cfgs:
                if "distractor_obj_distractor_obj" in cfg["name"]:
                    distractor_obj_in_receptacle |= OU.check_obj_in_receptacle(
                        self,
                        cfg["name"],
                        "container",
                        spawn_regions=[highest_spawn_region],
                    )
        else:
            distractor_obj_in_receptacle = False
        return (
            gripper_container_far
            and gripper_obj_far
            and obj_in_container
            and not obj_on_counter
            and container_upright
            and not distractor_obj_in_receptacle
        )

    def _reset_internal(self):
        TabletopPnP._reset_internal(self)

        # Randomization for joints
        joint_rand_strength = 0.2

        cotrain_qpos = dict()
        for robot_type, qpos in COTRAIN_REAL_MATCHED_ROBOT_INITIAL_POSE.items():
            if isinstance(self.robots[0].robot_model, robot_type):
                cotrain_qpos = deepcopy(qpos)
                break
        if not cotrain_qpos:
            print(f"No cotrain qpos for {self.robot_names}")

        current_qpos = self.sim.data.qpos.copy()

        for name in self.sim.model.joint_names:
            if "robot0_" in name:
                joint_id = self.sim.model.joint_name2id(name)
                if name in cotrain_qpos:
                    # For joints in COTRAIN_REAL_MATCHED_ROBOT_INITIAL_POSE, apply pos + randomization
                    new_pos = cotrain_qpos[name] + self.rng.uniform(
                        -joint_rand_strength, joint_rand_strength
                    )
                else:
                    # For other joints, use current pos
                    new_pos = current_qpos[joint_id]
                self.sim.data.qpos[joint_id] = new_pos

        self.sim.forward()

    def compute_robot_base_placement_pose(self, offset=None):
        robot_base_pos, robot_base_ori = TabletopPnP.compute_robot_base_placement_pose(
            self, offset
        )
        # Randomization for the table
        robot_base_pos += np.array([0.05, 0.05, -0.05])
        robot_base_pos -= self.rng.uniform(0, 0.05, 3)
        return robot_base_pos, robot_base_ori

    return type(
        class_name,
        (TabletopPnP,),
        {
            "__init__": __init__,
            "_create_objects": _create_objects,
            "_get_obj_cfgs": _get_obj_cfgs,
            "_get_distractor_obj_cfgs": _get_distractor_obj_cfgs,
            "_check_success": _check_success,
            "_check_grasp_distractor_obj": _check_grasp_distractor_obj,
            "_reset_internal": _reset_internal,
            "compute_robot_base_placement_pose": compute_robot_base_placement_pose,
        },
    )


def generate_task_classes(
    obj_cats: List[str],
    container_combos: List[Tuple[str, str]],
    prefix: str = "PnP",
    distractor_configs: dict = None,
    randomize_distractor_configs: bool = False,
    obj_instance_split: str = None,
    distractor_obj_cats: List[str] = None,
    postfix: str = None,
):
    """Generate all test task classes based on configuration."""
    task_infos = []
    source_containers_set = set([c for c, _ in container_combos])
    target_containers_set = set([t for _, t in container_combos])
    for source_container, target_container in container_combos:
        class_name = f"{prefix}From{source_container.replace('_','').title()}To{target_container.replace('_','').title()}"
        if postfix:
            class_name += postfix

        # NOTE: Python built-in hash() is randomized per-process by default (PYTHONHASHSEED),
        # which breaks determinism when envs are spawned in multiple processes.
        # Use a stable hash to generate a deterministic per-task seed.
        task_seed = zlib.crc32(class_name.encode("utf-8")) & 0xFFFFFFFF

        distractor_cfg = construct_distractor_obj_cfgs(
            included_configs=distractor_configs[(source_container, target_container)]
            if distractor_configs
            else None,
            randomize_configs=randomize_distractor_configs,
            task_seed=task_seed,
        )

        globals()[class_name] = create_pnp_class(
            class_name,
            obj_cats,
            source_container,
            target_container,
            all_obj_cats=obj_cats,
            # Deterministic order (avoid set-order dependence across processes)
            all_source_containers=sorted(source_containers_set),
            all_target_containers=sorted(target_containers_set),
            distractor_cfg=distractor_cfg,
            obj_instance_split=obj_instance_split,
            distractor_obj_cats=distractor_obj_cats,
        )
        task_infos.append(
            {
                "class_name": class_name,
                "source_container": source_container,
                "target_container": target_container,
                "obj_cats": obj_cats,
                "distractor_keys": distractor_cfg["regions"].keys(),
                "randomize_distractor_configs": randomize_distractor_configs,
            }
        )
    return task_infos


# ===============================================================================
#                             Main Execution
# ===============================================================================

novel_obj_cats = [
    "sweet_potato",
    "bell_pepper",
    "lemon",
    "croissant",
    "pear",
    "squash",
    "cupcake",
    "can",
    "tomato",
    "eggplant",
]
novel_container_combos = [
    # cutting_board
    ("cutting_board", "pot"),
    ("cutting_board", "basket"),
    ("cutting_board", "tiered_basket"),
    ("cutting_board", "pan"),
    ("cutting_board", "cardboard_box"),
    # placemat
    ("placemat", "bowl"),
    ("placemat", "plate"),
    ("placemat", "basket"),
    ("placemat", "tiered_shelf"),
    # plate
    ("plate", "pan"),
    ("plate", "cardboard_box"),
    ("plate", "bowl"),
    ("plate", "plate"),
    # tray
    ("tray", "tiered_shelf"),
    ("tray", "plate"),
    ("tray", "tiered_basket"),
    ("tray", "cardboard_box"),
    ("tray", "pot"),
]

base_obj_cats = get_excluded_obj_cats(novel_obj_cats)
base_container_combos = get_excluded_container_combos(novel_container_combos)


distractor_config_for_pretrain_base = {
    ("cutting_board", "bowl"): [],
    ("cutting_board", "plate"): ["distractor_obj", "distractor_target_container"],
    ("cutting_board", "tiered_shelf"): ["distractor_source_container"],
    ("tray", "basket"): ["distractor_obj"],
    ("tray", "pan"): ["distractor_target_container"],
    ("tray", "bowl"): ["distractor_target_container"],
    ("plate", "basket"): ["distractor_obj"],
    ("plate", "pot"): [],
    ("plate", "tiered_shelf"): ["distractor_obj", "distractor_target_container"],
    ("plate", "tiered_basket"): ["distractor_obj", "distractor_source_container"],
    ("placemat", "pan"): ["distractor_obj", "distractor_source_container"],
    ("placemat", "pot"): ["distractor_obj", "distractor_target_container"],
    ("placemat", "tiered_basket"): ["distractor_obj"],
    ("placemat", "cardboard_box"): [],
    ("cutting_board", "pot"): ["distractor_source_container"],
    ("cutting_board", "basket"): ["distractor_source_container"],
    ("cutting_board", "tiered_basket"): [
        "distractor_obj",
        "distractor_target_container",
    ],
    ("cutting_board", "pan"): [],
    ("cutting_board", "cardboard_box"): [],
    ("placemat", "bowl"): ["distractor_obj", "distractor_target_container"],
    ("placemat", "plate"): ["distractor_obj"],
    ("placemat", "basket"): ["distractor_obj", "distractor_source_container"],
    ("placemat", "tiered_shelf"): ["distractor_target_container"],
    ("plate", "pan"): ["distractor_source_container", "distractor_target_container"],
    ("plate", "cardboard_box"): ["distractor_source_container"],
    ("plate", "bowl"): [
        "distractor_obj",
        "distractor_source_container",
        "distractor_target_container",
    ],
    ("plate", "plate"): [],
    ("tray", "tiered_shelf"): ["distractor_target_container"],
    ("tray", "plate"): ["distractor_source_container", "distractor_target_container"],
    ("tray", "tiered_basket"): ["distractor_source_container"],
    ("tray", "cardboard_box"): [
        "distractor_obj",
        "distractor_source_container",
        "distractor_target_container",
    ],
    ("tray", "pot"): ["distractor_source_container"],
}

distractor_config_for_pretrain_novel = {
    ("cutting_board", "bowl"): [],
    ("cutting_board", "plate"): ["distractor_obj"],
    ("cutting_board", "tiered_shelf"): [
        "distractor_obj",
        "distractor_target_container",
    ],
    ("tray", "basket"): ["distractor_source_container"],
    ("tray", "pan"): [],
    ("tray", "bowl"): ["distractor_obj"],
    ("plate", "basket"): ["distractor_obj", "distractor_target_container"],
    ("plate", "pot"): ["distractor_source_container", "distractor_target_container"],
    ("plate", "tiered_shelf"): [
        "distractor_source_container",
        "distractor_target_container",
    ],
    ("plate", "tiered_basket"): ["distractor_obj", "distractor_source_container"],
    ("placemat", "pan"): ["distractor_target_container"],
    ("placemat", "pot"): ["distractor_obj"],
    ("placemat", "tiered_basket"): ["distractor_source_container"],
    ("placemat", "cardboard_box"): ["distractor_target_container"],
}

distractor_config_for_posttrain = {
    ("tray", "cardboard_box"): [],
    ("tray", "pot"): [],
    ("plate", "plate"): [],
    ("placemat", "plate"): ["distractor_obj"],
    ("plate", "cardboard_box"): ["distractor_obj"],
    ("plate", "pan"): ["distractor_obj"],
    ("placemat", "basket"): ["distractor_source_container"],
    ("cutting_board", "pan"): ["distractor_source_container"],
    ("cutting_board", "pot"): ["distractor_target_container"],
    ("tray", "plate"): ["distractor_target_container"],
    ("placemat", "bowl"): ["distractor_obj", "distractor_source_container"],
    ("plate", "bowl"): ["distractor_obj", "distractor_source_container"],
    ("cutting_board", "tiered_basket"): [
        "distractor_obj",
        "distractor_target_container",
    ],
    ("cutting_board", "cardboard_box"): [
        "distractor_obj",
        "distractor_target_container",
    ],
    ("tray", "tiered_basket"): [
        "distractor_source_container",
        "distractor_target_container",
    ],
    ("cutting_board", "basket"): [
        "distractor_source_container",
        "distractor_target_container",
    ],
    ("tray", "tiered_shelf"): [
        "distractor_obj",
        "distractor_source_container",
        "distractor_target_container",
    ],
    ("placemat", "tiered_shelf"): [
        "distractor_obj",
        "distractor_source_container",
        "distractor_target_container",
    ],
}

# Generate all task classes

# Pre-train
pretrain_task_infos = generate_task_classes(
    base_obj_cats,
    base_container_combos,
    prefix="PretrainPnPBase",
    randomize_distractor_configs=False,
    distractor_configs=distractor_config_for_pretrain_base,
    obj_instance_split="A",
    postfix="SplitA",
)
pretrain_task_infos.extend(
    generate_task_classes(
        base_obj_cats,
        novel_container_combos,
        prefix="PretrainPnPBase",
        randomize_distractor_configs=False,
        distractor_configs=distractor_config_for_pretrain_base,
        obj_instance_split="A",
        postfix="SplitA",
    )
)
pretrain_task_infos.extend(
    generate_task_classes(
        novel_obj_cats,
        base_container_combos,
        prefix="PretrainPnPNovel",
        randomize_distractor_configs=False,
        distractor_configs=distractor_config_for_pretrain_novel,
        obj_instance_split="A",
        postfix="SplitA",
    )
)

# Post-train
posttrain_task_infos = generate_task_classes(
    novel_obj_cats,
    novel_container_combos,
    prefix="PosttrainPnPNovel",
    randomize_distractor_configs=False,
    distractor_configs=distractor_config_for_posttrain,
    obj_instance_split="A",
    postfix="SplitA",
)

eval_task_infos = generate_task_classes(
    novel_obj_cats,
    novel_container_combos,
    prefix="EvalPnPNovel",
    randomize_distractor_configs=False,
    distractor_configs=distractor_config_for_posttrain,
    obj_instance_split="B",
    postfix="SplitB",
)


distractor_config_always_distractor_obj = deepcopy(distractor_config_for_posttrain)
for key, value in distractor_config_always_distractor_obj.items():
    if "distractor_obj" not in value:
        if rng.random() < 0.5 and len(distractor_config_always_distractor_obj[key]) > 1:
            # random drop one of the distractor keys
            distractor_config_always_distractor_obj[key].remove(
                rng.choice(distractor_config_always_distractor_obj[key])
            )
        distractor_config_always_distractor_obj[key].append("distractor_obj")


# generate posttrain tasks with a specific object group as target object
posttrain_specific_obj_task_infos = []
for obj_group in ["fruit"]:
    posttrain_specific_obj_task_infos.extend(
        generate_task_classes(
            [obj_group],
            novel_container_combos,
            prefix=f"PosttrainPnP{obj_group.title()}",
            randomize_distractor_configs=False,
            distractor_configs=distractor_config_always_distractor_obj,
            # obj_instance_split="A",
            # postfix="SplitA",
        )
    )


if __name__ == "__main__":

    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'  # Resets color to terminal default

    # print pretrain task names
    print(f"{GREEN}DC24 Pretrain task names: {len(pretrain_task_infos)} {RESET}")
    for task_info in pretrain_task_infos:
        print(f"{task_info['class_name']}")

    print()
    # print posttrain task names
    print(f"{GREEN}DC24 Posttrain task names: {len(posttrain_task_infos)} {RESET}")
    for task_info in posttrain_task_infos:
        print(f"{task_info['class_name']}")
