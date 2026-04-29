from rldx.configs.data.embodiment_configs import register_modality_config
from rldx.data.embodiment_tags import get_embodimenttag_by_name
from rldx.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


PRETRAIN_MODALITY_CONFIGS = {
    "fractal20220817_data": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # end_effector_rotation
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_close
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "kuka": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # end_effector_rotation
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_close
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "bridge_orig": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary", "secondary"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # end_effector_rotation
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_close
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "taco_play": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary", "wrist"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # end_effector_rotation
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_close
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "jaco_play": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary", "wrist"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # end_effector_rotation
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_close
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "berkeley_cable_routing": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary", "secondary", "wrist"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["joint_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # end_effector_rotation
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_close
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "roboturk": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["none"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # end_effector_rotation
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_close
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "viola": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary", "wrist"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["joint_position", "gripper_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # end_effector_rotation
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_close
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "berkeley_autolab_ur5": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary", "wrist"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # end_effector_rotation
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_close
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "toto": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["joint_position", "gripper_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # end_effector_rotation
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_close
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "language_table": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["end_effector_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "stanford_hydra_dataset_converted_externally_to_rlds": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary", "wrist"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # end_effector_rotation
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_close
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "austin_buds_dataset_converted_externally_to_rlds": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary", "wrist"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["joint_position", "gripper_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # end_effector_rotation
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_close
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "nyu_franka_play_dataset_converted_externally_to_rlds": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary", "secondary"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # end_effector_rotation
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_close
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "furniture_bench_dataset_converted_externally_to_rlds": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary", "wrist"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # end_effector_rotation
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_close
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "ucsd_kitchen_dataset_converted_externally_to_rlds": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["joint_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # end_effector_rotation
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_close
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "austin_sailor_dataset_converted_externally_to_rlds": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary", "wrist"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # end_effector_rotation
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_close
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "austin_sirius_dataset_converted_externally_to_rlds": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary", "wrist"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # end_effector_rotation
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_close
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "dlr_edan_shared_control_converted_externally_to_rlds": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # end_effector_rotation
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_close
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary", "wrist"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # end_effector_rotation
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_close
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "utaustin_mutex": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary", "wrist"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["joint_position", "gripper_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # end_effector_rotation
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_close
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "berkeley_fanuc_manipulation": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary", "wrist"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["joint_position", "gripper_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # end_effector_rotation
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_close
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "cmu_stretch": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # end_effector_rotation
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_close
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "bc_z": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "primary",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # end_effector_rotation
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_close
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "fmb_dataset": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary", "secondary", "wrist"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # end_effector_rotation
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_close
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "dobbe": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["wrist"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # end_effector_rotation
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_close
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "droid": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary", "secondary", "wrist"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
            action_configs=[
                # end_effector_position
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # end_effector_rotation
                ActionConfig(
                    rep=ActionRepresentation.DELTA,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_close
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "agibot_dexhand": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["state"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["action"],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "agibot_gripper": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary", "wrist_left", "wrist_right"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["state"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["action"],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "galaxea": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary", "wrist_left", "wrist_right"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["state"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["action"],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "humanoid_everyday_g1": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["egocentric_resized"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm",
                "left_hand",
                "right_arm",
                "right_hand",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=[
                "left_arm",
                "left_hand",
                "right_arm",
                "right_hand",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "humanoid_everyday_h1": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["egocentric_resized"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm",
                "left_hand",
                "right_arm",
                "right_hand",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=[
                "left_arm",
                "left_hand",
                "right_arm",
                "right_hand",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "action_net": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["primary"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["state"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["action"],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "neural_gr1": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm",
                "left_hand",
                "left_leg",
                "neck",
                "right_arm",
                "right_hand",
                "right_leg",
                "waist",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=[
                "left_arm",
                "left_hand",
                "left_leg",
                "neck",
                "right_arm",
                "right_hand",
                "right_leg",
                "waist",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "new_embodiment": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm",
                "left_hand",
                "left_leg",
                "neck",
                "right_arm",
                "right_hand",
                "right_leg",
                "waist",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=[
                "left_arm",
                "left_hand",
                "left_leg",
                "neck",
                "right_arm",
                "right_hand",
                "right_leg",
                "waist",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
}


for name, modality_config in PRETRAIN_MODALITY_CONFIGS.items():
    register_modality_config(modality_config, get_embodimenttag_by_name(name))
