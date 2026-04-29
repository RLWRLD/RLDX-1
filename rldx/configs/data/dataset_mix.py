from rldx.data.embodiment_tags import EmbodimentTag


dataset_mix = {
    "debug": [
        {
            "dataset_name": "berkeley_autolab_ur5",
            "mix_ratio": 0.003,
            "embodiment_tag": EmbodimentTag.OXE_BERKELEY_AUTOLAB_UR5,
        },
    ],
    "oxe_magic_soup_plus": [
        {
            "dataset_name": "fractal20220817_data",
            "mix_ratio": 0.04,
            "embodiment_tag": EmbodimentTag.OXE_FRACTAL,
        },
        {"dataset_name": "kuka", "mix_ratio": 0.04, "embodiment_tag": EmbodimentTag.OXE_KUKA},
        {
            "dataset_name": "bridge_orig",
            "mix_ratio": 0.04,
            "embodiment_tag": EmbodimentTag.OXE_BRIDGE_ORIG,
        },
        {
            "dataset_name": "taco_play",
            "mix_ratio": 0.0084,
            "embodiment_tag": EmbodimentTag.OXE_TACO,
        },
        {
            "dataset_name": "jaco_play",
            "mix_ratio": 0.0015,
            "embodiment_tag": EmbodimentTag.OXE_JACO,
        },
        {
            "dataset_name": "berkeley_cable_routing",
            "mix_ratio": 0.0008,
            "embodiment_tag": EmbodimentTag.OXE_BERKELEY_CABLE_ROUTING,
        },
        {
            "dataset_name": "roboturk",
            "mix_ratio": 0.0065,
            "embodiment_tag": EmbodimentTag.OXE_ROBOTURK,
        },
        {"dataset_name": "viola", "mix_ratio": 0.003, "embodiment_tag": EmbodimentTag.OXE_VIOLA},
        {
            "dataset_name": "berkeley_autolab_ur5",
            "mix_ratio": 0.003,
            "embodiment_tag": EmbodimentTag.OXE_BERKELEY_AUTOLAB_UR5,
        },
        {"dataset_name": "toto", "mix_ratio": 0.005, "embodiment_tag": EmbodimentTag.OXE_TOTO},
        {
            "dataset_name": "language_table",
            "mix_ratio": 0.0125,
            "embodiment_tag": EmbodimentTag.OXE_LANGUAGE_TABLE,
        },
        {
            "dataset_name": "stanford_hydra_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.0125,
            "embodiment_tag": EmbodimentTag.OXE_STANFORD_HYDRA,
        },
        {
            "dataset_name": "austin_buds_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.0006,
            "embodiment_tag": EmbodimentTag.OXE_AUSTIN_BUDS,
        },
        {
            "dataset_name": "nyu_franka_play_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.002,
            "embodiment_tag": EmbodimentTag.OXE_NYU_FRANKA_PLAY,
        },
        {
            "dataset_name": "furniture_bench_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.007,
            "embodiment_tag": EmbodimentTag.OXE_FURNITURE_BENCH,
        },
        {
            "dataset_name": "ucsd_kitchen_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.0001,
            "embodiment_tag": EmbodimentTag.OXE_UCSD_KITCHEN,
        },
        {
            "dataset_name": "austin_sailor_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.006,
            "embodiment_tag": EmbodimentTag.OXE_AUSTIN_SAILOR,
        },
        {
            "dataset_name": "austin_sirius_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.005,
            "embodiment_tag": EmbodimentTag.OXE_AUSTIN_SIRIUS,
        },
        {
            "dataset_name": "dlr_edan_shared_control_converted_externally_to_rlds",
            "mix_ratio": 0.0002,
            "embodiment_tag": EmbodimentTag.OXE_DLR_EDAN_SHARED_CONTROL,
        },
        {
            "dataset_name": "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
            "mix_ratio": 0.0025,
            "embodiment_tag": EmbodimentTag.OXE_IAMLAB_CMU_PICKUP_INSERT,
        },
        {
            "dataset_name": "utaustin_mutex",
            "mix_ratio": 0.006,
            "embodiment_tag": EmbodimentTag.OXE_UTAUSTIN_MUTEX,
        },
        {
            "dataset_name": "berkeley_fanuc_manipulation",
            "mix_ratio": 0.002,
            "embodiment_tag": EmbodimentTag.OXE_BERKELEY_FANUC_MANIPULATION,
        },
        {
            "dataset_name": "cmu_stretch",
            "mix_ratio": 0.0004,
            "embodiment_tag": EmbodimentTag.OXE_CMU_STRETCH,
        },
        {"dataset_name": "bc_z", "mix_ratio": 0.02, "embodiment_tag": EmbodimentTag.OXE_BC_Z},
        {
            "dataset_name": "fmb_dataset",
            "mix_ratio": 0.02,
            "embodiment_tag": EmbodimentTag.OXE_FMB_DATASET,
        },
        {"dataset_name": "dobbe", "mix_ratio": 0.005, "embodiment_tag": EmbodimentTag.OXE_DOBBE},
        {"dataset_name": "droid", "mix_ratio": 0.25, "embodiment_tag": EmbodimentTag.OXE_DROID},
    ],
    "rldx_mix_v3": [
        {
            "dataset_name": "fractal20220817_data",
            "mix_ratio": 0.04,
            "embodiment_tag": EmbodimentTag.OXE_FRACTAL,
        },
        {"dataset_name": "kuka", "mix_ratio": 0.04, "embodiment_tag": EmbodimentTag.OXE_KUKA},
        {
            "dataset_name": "bridge_orig",
            "mix_ratio": 0.04,
            "embodiment_tag": EmbodimentTag.OXE_BRIDGE_ORIG,
        },
        {
            "dataset_name": "taco_play",
            "mix_ratio": 0.0084,
            "embodiment_tag": EmbodimentTag.OXE_TACO,
        },
        {
            "dataset_name": "jaco_play",
            "mix_ratio": 0.0015,
            "embodiment_tag": EmbodimentTag.OXE_JACO,
        },
        {
            "dataset_name": "berkeley_cable_routing",
            "mix_ratio": 0.0008,
            "embodiment_tag": EmbodimentTag.OXE_BERKELEY_CABLE_ROUTING,
        },
        {
            "dataset_name": "roboturk",
            "mix_ratio": 0.0065,
            "embodiment_tag": EmbodimentTag.OXE_ROBOTURK,
        },
        {"dataset_name": "viola", "mix_ratio": 0.003, "embodiment_tag": EmbodimentTag.OXE_VIOLA},
        {
            "dataset_name": "berkeley_autolab_ur5",
            "mix_ratio": 0.003,
            "embodiment_tag": EmbodimentTag.OXE_BERKELEY_AUTOLAB_UR5,
        },
        {"dataset_name": "toto", "mix_ratio": 0.005, "embodiment_tag": EmbodimentTag.OXE_TOTO},
        {
            "dataset_name": "language_table",
            "mix_ratio": 0.0125,
            "embodiment_tag": EmbodimentTag.OXE_LANGUAGE_TABLE,
        },
        {
            "dataset_name": "stanford_hydra_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.0125,
            "embodiment_tag": EmbodimentTag.OXE_STANFORD_HYDRA,
        },
        {
            "dataset_name": "austin_buds_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.0006,
            "embodiment_tag": EmbodimentTag.OXE_AUSTIN_BUDS,
        },
        {
            "dataset_name": "nyu_franka_play_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.002,
            "embodiment_tag": EmbodimentTag.OXE_NYU_FRANKA_PLAY,
        },
        {
            "dataset_name": "furniture_bench_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.007,
            "embodiment_tag": EmbodimentTag.OXE_FURNITURE_BENCH,
        },
        {
            "dataset_name": "ucsd_kitchen_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.0001,
            "embodiment_tag": EmbodimentTag.OXE_UCSD_KITCHEN,
        },
        {
            "dataset_name": "austin_sailor_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.006,
            "embodiment_tag": EmbodimentTag.OXE_AUSTIN_SAILOR,
        },
        {
            "dataset_name": "austin_sirius_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.005,
            "embodiment_tag": EmbodimentTag.OXE_AUSTIN_SIRIUS,
        },
        {
            "dataset_name": "dlr_edan_shared_control_converted_externally_to_rlds",
            "mix_ratio": 0.0002,
            "embodiment_tag": EmbodimentTag.OXE_DLR_EDAN_SHARED_CONTROL,
        },
        {
            "dataset_name": "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
            "mix_ratio": 0.0025,
            "embodiment_tag": EmbodimentTag.OXE_IAMLAB_CMU_PICKUP_INSERT,
        },
        {
            "dataset_name": "utaustin_mutex",
            "mix_ratio": 0.006,
            "embodiment_tag": EmbodimentTag.OXE_UTAUSTIN_MUTEX,
        },
        {
            "dataset_name": "berkeley_fanuc_manipulation",
            "mix_ratio": 0.002,
            "embodiment_tag": EmbodimentTag.OXE_BERKELEY_FANUC_MANIPULATION,
        },
        {
            "dataset_name": "cmu_stretch",
            "mix_ratio": 0.0004,
            "embodiment_tag": EmbodimentTag.OXE_CMU_STRETCH,
        },
        {"dataset_name": "bc_z", "mix_ratio": 0.02, "embodiment_tag": EmbodimentTag.OXE_BC_Z},
        {
            "dataset_name": "fmb_dataset",
            "mix_ratio": 0.02,
            "embodiment_tag": EmbodimentTag.OXE_FMB_DATASET,
        },
        {"dataset_name": "dobbe", "mix_ratio": 0.005, "embodiment_tag": EmbodimentTag.OXE_DOBBE},
        {"dataset_name": "droid", "mix_ratio": 0.25, "embodiment_tag": EmbodimentTag.OXE_DROID},
        {
            "dataset_name": "galaxea_part1",
            "mix_ratio": 0.02,
            "embodiment_tag": EmbodimentTag.GALAXEA,
        },
        {
            "dataset_name": "galaxea_part2",
            "mix_ratio": 0.02,
            "embodiment_tag": EmbodimentTag.GALAXEA,
        },
        {
            "dataset_name": "galaxea_part3",
            "mix_ratio": 0.02,
            "embodiment_tag": EmbodimentTag.GALAXEA,
        },
        {
            "dataset_name": "galaxea_part4",
            "mix_ratio": 0.02,
            "embodiment_tag": EmbodimentTag.GALAXEA,
        },
        {
            "dataset_name": "galaxea_part5",
            "mix_ratio": 0.02,
            "embodiment_tag": EmbodimentTag.GALAXEA,
        },
        {
            "dataset_name": "agibot_gripper_part1",
            "mix_ratio": 0.025,
            "embodiment_tag": EmbodimentTag.AGIBOT_GRIPPER,
        },
        {
            "dataset_name": "agibot_gripper_part2",
            "mix_ratio": 0.025,
            "embodiment_tag": EmbodimentTag.AGIBOT_GRIPPER,
        },
        {
            "dataset_name": "agibot_gripper_part3",
            "mix_ratio": 0.025,
            "embodiment_tag": EmbodimentTag.AGIBOT_GRIPPER,
        },
        {
            "dataset_name": "agibot_gripper_part4",
            "mix_ratio": 0.025,
            "embodiment_tag": EmbodimentTag.AGIBOT_GRIPPER,
        },
        {
            "dataset_name": "agibot_dexhand",
            "mix_ratio": 0.05,
            "embodiment_tag": EmbodimentTag.AGIBOT_DEXHAND,
        },
        {
            "dataset_name": "action_net",
            "mix_ratio": 0.05,
            "embodiment_tag": EmbodimentTag.ACTION_NET,
        },
        {
            "dataset_name": "neural_robocurate_v1",
            "mix_ratio": 0.04,
            "embodiment_tag": EmbodimentTag.NEURAL_GR1,
        },
        {
            "dataset_name": "neural_robocurate_v2",
            "mix_ratio": 0.03,
            "embodiment_tag": EmbodimentTag.NEURAL_GR1,
        },
        {
            "dataset_name": "neural_robocurate_v3",
            "mix_ratio": 0.03,
            "embodiment_tag": EmbodimentTag.NEURAL_GR1,
        },
        {
            "dataset_name": "humanoid_everyday_g1",
            "mix_ratio": 0.025,
            "embodiment_tag": EmbodimentTag.HUMANOID_EVERYDAY_G1,
        },
        {
            "dataset_name": "humanoid_everyday_h1",
            "mix_ratio": 0.025,
            "embodiment_tag": EmbodimentTag.HUMANOID_EVERYDAY_H1,
        },
        {
            "dataset_name": "humanoid_everyday_h1",
            "mix_ratio": 0.83,
            "embodiment_tag": EmbodimentTag.HUMANOID_EVERYDAY_H1,
        },
    ],
    "rldx_mix_v4": [
        {
            "dataset_name": "fractal20220817_data",
            "mix_ratio": 0.04,
            "embodiment_tag": EmbodimentTag.OXE_FRACTAL,
        },
        {"dataset_name": "kuka", "mix_ratio": 0.04, "embodiment_tag": EmbodimentTag.OXE_KUKA},
        {
            "dataset_name": "bridge_orig",
            "mix_ratio": 0.04,
            "embodiment_tag": EmbodimentTag.OXE_BRIDGE_ORIG,
        },
        {
            "dataset_name": "taco_play",
            "mix_ratio": 0.0084,
            "embodiment_tag": EmbodimentTag.OXE_TACO,
        },
        {
            "dataset_name": "jaco_play",
            "mix_ratio": 0.0015,
            "embodiment_tag": EmbodimentTag.OXE_JACO,
        },
        {
            "dataset_name": "berkeley_cable_routing",
            "mix_ratio": 0.0008,
            "embodiment_tag": EmbodimentTag.OXE_BERKELEY_CABLE_ROUTING,
        },
        {
            "dataset_name": "roboturk",
            "mix_ratio": 0.0065,
            "embodiment_tag": EmbodimentTag.OXE_ROBOTURK,
        },
        {"dataset_name": "viola", "mix_ratio": 0.003, "embodiment_tag": EmbodimentTag.OXE_VIOLA},
        {
            "dataset_name": "berkeley_autolab_ur5",
            "mix_ratio": 0.003,
            "embodiment_tag": EmbodimentTag.OXE_BERKELEY_AUTOLAB_UR5,
        },
        {"dataset_name": "toto", "mix_ratio": 0.005, "embodiment_tag": EmbodimentTag.OXE_TOTO},
        {
            "dataset_name": "language_table",
            "mix_ratio": 0.0125,
            "embodiment_tag": EmbodimentTag.OXE_LANGUAGE_TABLE,
        },
        {
            "dataset_name": "stanford_hydra_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.0125,
            "embodiment_tag": EmbodimentTag.OXE_STANFORD_HYDRA,
        },
        {
            "dataset_name": "austin_buds_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.0006,
            "embodiment_tag": EmbodimentTag.OXE_AUSTIN_BUDS,
        },
        {
            "dataset_name": "nyu_franka_play_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.002,
            "embodiment_tag": EmbodimentTag.OXE_NYU_FRANKA_PLAY,
        },
        {
            "dataset_name": "furniture_bench_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.007,
            "embodiment_tag": EmbodimentTag.OXE_FURNITURE_BENCH,
        },
        {
            "dataset_name": "ucsd_kitchen_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.0001,
            "embodiment_tag": EmbodimentTag.OXE_UCSD_KITCHEN,
        },
        {
            "dataset_name": "austin_sailor_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.006,
            "embodiment_tag": EmbodimentTag.OXE_AUSTIN_SAILOR,
        },
        {
            "dataset_name": "austin_sirius_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.005,
            "embodiment_tag": EmbodimentTag.OXE_AUSTIN_SIRIUS,
        },
        {
            "dataset_name": "dlr_edan_shared_control_converted_externally_to_rlds",
            "mix_ratio": 0.0002,
            "embodiment_tag": EmbodimentTag.OXE_DLR_EDAN_SHARED_CONTROL,
        },
        {
            "dataset_name": "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
            "mix_ratio": 0.0025,
            "embodiment_tag": EmbodimentTag.OXE_IAMLAB_CMU_PICKUP_INSERT,
        },
        {
            "dataset_name": "utaustin_mutex",
            "mix_ratio": 0.006,
            "embodiment_tag": EmbodimentTag.OXE_UTAUSTIN_MUTEX,
        },
        {
            "dataset_name": "berkeley_fanuc_manipulation",
            "mix_ratio": 0.002,
            "embodiment_tag": EmbodimentTag.OXE_BERKELEY_FANUC_MANIPULATION,
        },
        {
            "dataset_name": "cmu_stretch",
            "mix_ratio": 0.0004,
            "embodiment_tag": EmbodimentTag.OXE_CMU_STRETCH,
        },
        {"dataset_name": "bc_z", "mix_ratio": 0.02, "embodiment_tag": EmbodimentTag.OXE_BC_Z},
        {
            "dataset_name": "fmb_dataset",
            "mix_ratio": 0.02,
            "embodiment_tag": EmbodimentTag.OXE_FMB_DATASET,
        },
        {"dataset_name": "dobbe", "mix_ratio": 0.005, "embodiment_tag": EmbodimentTag.OXE_DOBBE},
        {"dataset_name": "droid", "mix_ratio": 0.25, "embodiment_tag": EmbodimentTag.OXE_DROID},
        {
            "dataset_name": "galaxea_part1",
            "mix_ratio": 0.02,
            "embodiment_tag": EmbodimentTag.GALAXEA,
        },
        {
            "dataset_name": "galaxea_part2",
            "mix_ratio": 0.02,
            "embodiment_tag": EmbodimentTag.GALAXEA,
        },
        {
            "dataset_name": "galaxea_part3",
            "mix_ratio": 0.02,
            "embodiment_tag": EmbodimentTag.GALAXEA,
        },
        {
            "dataset_name": "galaxea_part4",
            "mix_ratio": 0.02,
            "embodiment_tag": EmbodimentTag.GALAXEA,
        },
        {
            "dataset_name": "galaxea_part5",
            "mix_ratio": 0.02,
            "embodiment_tag": EmbodimentTag.GALAXEA,
        },
        {
            "dataset_name": "agibot_gripper_part1",
            "mix_ratio": 0.025,
            "embodiment_tag": EmbodimentTag.AGIBOT_GRIPPER,
        },
        {
            "dataset_name": "agibot_gripper_part2",
            "mix_ratio": 0.025,
            "embodiment_tag": EmbodimentTag.AGIBOT_GRIPPER,
        },
        {
            "dataset_name": "agibot_gripper_part3",
            "mix_ratio": 0.025,
            "embodiment_tag": EmbodimentTag.AGIBOT_GRIPPER,
        },
        {
            "dataset_name": "agibot_gripper_part4",
            "mix_ratio": 0.025,
            "embodiment_tag": EmbodimentTag.AGIBOT_GRIPPER,
        },
        {
            "dataset_name": "agibot_dexhand",
            "mix_ratio": 0.05,
            "embodiment_tag": EmbodimentTag.AGIBOT_DEXHAND,
        },
        {
            "dataset_name": "action_net",
            "mix_ratio": 0.05,
            "embodiment_tag": EmbodimentTag.ACTION_NET,
        },
        {
            "dataset_name": "neural_robocurate_v1",
            "mix_ratio": 0.04,
            "embodiment_tag": EmbodimentTag.NEURAL_GR1,
        },
        {
            "dataset_name": "neural_robocurate_v2",
            "mix_ratio": 0.03,
            "embodiment_tag": EmbodimentTag.NEURAL_GR1,
        },
        {
            "dataset_name": "neural_robocurate_v3",
            "mix_ratio": 0.03,
            "embodiment_tag": EmbodimentTag.NEURAL_GR1,
        },
        {
            "dataset_name": "humanoid_everyday_g1",
            "mix_ratio": 0.025,
            "embodiment_tag": EmbodimentTag.HUMANOID_EVERYDAY_G1,
        },
        {
            "dataset_name": "humanoid_everyday_h1",
            "mix_ratio": 0.025,
            "embodiment_tag": EmbodimentTag.HUMANOID_EVERYDAY_H1,
        },
        {
            "dataset_name": "new_embodiment",
            "mix_ratio": 0.05,
            "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        },
    ],
    "rldx_mix_v5": [
        {
            "dataset_name": "fractal20220817_data",
            "mix_ratio": 0.04,
            "embodiment_tag": EmbodimentTag.OXE_FRACTAL,
        },
        {"dataset_name": "kuka", "mix_ratio": 0.04, "embodiment_tag": EmbodimentTag.OXE_KUKA},
        {
            "dataset_name": "bridge_orig",
            "mix_ratio": 0.04,
            "embodiment_tag": EmbodimentTag.OXE_BRIDGE_ORIG,
        },
        {
            "dataset_name": "taco_play",
            "mix_ratio": 0.0084,
            "embodiment_tag": EmbodimentTag.OXE_TACO,
        },
        {
            "dataset_name": "jaco_play",
            "mix_ratio": 0.0015,
            "embodiment_tag": EmbodimentTag.OXE_JACO,
        },
        {
            "dataset_name": "berkeley_cable_routing",
            "mix_ratio": 0.0008,
            "embodiment_tag": EmbodimentTag.OXE_BERKELEY_CABLE_ROUTING,
        },
        {
            "dataset_name": "roboturk",
            "mix_ratio": 0.0065,
            "embodiment_tag": EmbodimentTag.OXE_ROBOTURK,
        },
        {"dataset_name": "viola", "mix_ratio": 0.003, "embodiment_tag": EmbodimentTag.OXE_VIOLA},
        {
            "dataset_name": "berkeley_autolab_ur5",
            "mix_ratio": 0.003,
            "embodiment_tag": EmbodimentTag.OXE_BERKELEY_AUTOLAB_UR5,
        },
        {"dataset_name": "toto", "mix_ratio": 0.005, "embodiment_tag": EmbodimentTag.OXE_TOTO},
        {
            "dataset_name": "language_table",
            "mix_ratio": 0.0125,
            "embodiment_tag": EmbodimentTag.OXE_LANGUAGE_TABLE,
        },
        {
            "dataset_name": "stanford_hydra_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.0125,
            "embodiment_tag": EmbodimentTag.OXE_STANFORD_HYDRA,
        },
        {
            "dataset_name": "austin_buds_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.0006,
            "embodiment_tag": EmbodimentTag.OXE_AUSTIN_BUDS,
        },
        {
            "dataset_name": "nyu_franka_play_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.002,
            "embodiment_tag": EmbodimentTag.OXE_NYU_FRANKA_PLAY,
        },
        {
            "dataset_name": "furniture_bench_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.007,
            "embodiment_tag": EmbodimentTag.OXE_FURNITURE_BENCH,
        },
        {
            "dataset_name": "ucsd_kitchen_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.0001,
            "embodiment_tag": EmbodimentTag.OXE_UCSD_KITCHEN,
        },
        {
            "dataset_name": "austin_sailor_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.006,
            "embodiment_tag": EmbodimentTag.OXE_AUSTIN_SAILOR,
        },
        {
            "dataset_name": "austin_sirius_dataset_converted_externally_to_rlds",
            "mix_ratio": 0.005,
            "embodiment_tag": EmbodimentTag.OXE_AUSTIN_SIRIUS,
        },
        {
            "dataset_name": "dlr_edan_shared_control_converted_externally_to_rlds",
            "mix_ratio": 0.0002,
            "embodiment_tag": EmbodimentTag.OXE_DLR_EDAN_SHARED_CONTROL,
        },
        {
            "dataset_name": "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
            "mix_ratio": 0.0025,
            "embodiment_tag": EmbodimentTag.OXE_IAMLAB_CMU_PICKUP_INSERT,
        },
        {
            "dataset_name": "utaustin_mutex",
            "mix_ratio": 0.006,
            "embodiment_tag": EmbodimentTag.OXE_UTAUSTIN_MUTEX,
        },
        {
            "dataset_name": "berkeley_fanuc_manipulation",
            "mix_ratio": 0.002,
            "embodiment_tag": EmbodimentTag.OXE_BERKELEY_FANUC_MANIPULATION,
        },
        {
            "dataset_name": "cmu_stretch",
            "mix_ratio": 0.0004,
            "embodiment_tag": EmbodimentTag.OXE_CMU_STRETCH,
        },
        {"dataset_name": "bc_z", "mix_ratio": 0.02, "embodiment_tag": EmbodimentTag.OXE_BC_Z},
        {
            "dataset_name": "fmb_dataset",
            "mix_ratio": 0.02,
            "embodiment_tag": EmbodimentTag.OXE_FMB_DATASET,
        },
        {"dataset_name": "dobbe", "mix_ratio": 0.005, "embodiment_tag": EmbodimentTag.OXE_DOBBE},
        {"dataset_name": "droid", "mix_ratio": 0.25, "embodiment_tag": EmbodimentTag.OXE_DROID},
        {
            "dataset_name": "galaxea_part1",
            "mix_ratio": 0.02,
            "embodiment_tag": EmbodimentTag.GALAXEA,
        },
        {
            "dataset_name": "galaxea_part2",
            "mix_ratio": 0.02,
            "embodiment_tag": EmbodimentTag.GALAXEA,
        },
        {
            "dataset_name": "galaxea_part3",
            "mix_ratio": 0.02,
            "embodiment_tag": EmbodimentTag.GALAXEA,
        },
        {
            "dataset_name": "galaxea_part4",
            "mix_ratio": 0.02,
            "embodiment_tag": EmbodimentTag.GALAXEA,
        },
        {
            "dataset_name": "galaxea_part5",
            "mix_ratio": 0.02,
            "embodiment_tag": EmbodimentTag.GALAXEA,
        },
        {
            "dataset_name": "agibot_gripper_part1",
            "mix_ratio": 0.025,
            "embodiment_tag": EmbodimentTag.AGIBOT_GRIPPER,
        },
        {
            "dataset_name": "agibot_gripper_part2",
            "mix_ratio": 0.025,
            "embodiment_tag": EmbodimentTag.AGIBOT_GRIPPER,
        },
        {
            "dataset_name": "agibot_gripper_part3",
            "mix_ratio": 0.025,
            "embodiment_tag": EmbodimentTag.AGIBOT_GRIPPER,
        },
        {
            "dataset_name": "agibot_gripper_part4",
            "mix_ratio": 0.025,
            "embodiment_tag": EmbodimentTag.AGIBOT_GRIPPER,
        },
        {
            "dataset_name": "agibot_dexhand",
            "mix_ratio": 0.05,
            "embodiment_tag": EmbodimentTag.AGIBOT_DEXHAND,
        },
        {
            "dataset_name": "action_net",
            "mix_ratio": 0.1,
            "embodiment_tag": EmbodimentTag.ACTION_NET,
        },
        {
            "dataset_name": "march_robocurate_v1",
            "mix_ratio": 0.03,
            "embodiment_tag": EmbodimentTag.NEURAL_GR1,
        },
        {
            "dataset_name": "march_robocurate_v2",
            "mix_ratio": 0.025,
            "embodiment_tag": EmbodimentTag.NEURAL_GR1,
        },
        {
            "dataset_name": "march_robocurate_v3",
            "mix_ratio": 0.035,
            "embodiment_tag": EmbodimentTag.NEURAL_GR1,
        },
        {
            "dataset_name": "march_robocurate_v4",
            "mix_ratio": 0.01,
            "embodiment_tag": EmbodimentTag.NEURAL_GR1,
        },
        {
            "dataset_name": "humanoid_everyday_g1",
            "mix_ratio": 0.025,
            "embodiment_tag": EmbodimentTag.HUMANOID_EVERYDAY_G1,
        },
        {
            "dataset_name": "humanoid_everyday_h1",
            "mix_ratio": 0.025,
            "embodiment_tag": EmbodimentTag.HUMANOID_EVERYDAY_H1,
        },
    ],
    "robocasa_mimicgen_3000": [
        {
            "dataset_name": "single_panda_gripper.CloseDoubleDoor",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "single_panda_gripper.CloseDrawer",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "single_panda_gripper.CloseSingleDoor",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "single_panda_gripper.CoffeePressButton",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "single_panda_gripper.CoffeeServeMug",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "single_panda_gripper.CoffeeSetupMug",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "single_panda_gripper.OpenDoubleDoor",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "single_panda_gripper.OpenDrawer",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "single_panda_gripper.OpenSingleDoor",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "single_panda_gripper.PnPCabToCounter",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "single_panda_gripper.PnPCounterToCab",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "single_panda_gripper.PnPCounterToMicrowave",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "single_panda_gripper.PnPCounterToSink",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "single_panda_gripper.PnPCounterToStove",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "single_panda_gripper.PnPMicrowaveToCounter",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "single_panda_gripper.PnPSinkToCounter",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "single_panda_gripper.PnPStoveToCounter",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "single_panda_gripper.TurnOffMicrowave",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "single_panda_gripper.TurnOffSinkFaucet",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "single_panda_gripper.TurnOffStove",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "single_panda_gripper.TurnOnMicrowave",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "single_panda_gripper.TurnOnSinkFaucet",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "single_panda_gripper.TurnOnStove",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "single_panda_gripper.TurnSinkSpout",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
    ],
    "calvin": [
        {
            "dataset_name": "calvin_task_ABC_D_lerobot_0_4",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "calvin_task_ABC_D_lerobot_1_4",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "calvin_task_ABC_D_lerobot_2_4",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "calvin_task_ABC_D_lerobot_3_4",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
    ],
    "openarm_inspire_pnp_flat": [
        {
            "dataset_name": "task_1_pnp_flat_right_hand",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "task_2_pnp_flat_left_to_right_hand",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
    ],
    "openarm_inspire_pnp_flat_v2": [
        {
            "dataset_name": "task_1_pnp_flat_right_hand",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "task_2_pnp_flat_left_to_right_hand",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "task_3_pnp_flat_pick_only",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
    ],
    "rldx1_midtrain_allex": [
        {
            "dataset_name": "real_allex",
            "mix_ratio": 0.5,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "robocurate_contiguous_seen_img_seen_instruction",
            "mix_ratio": 0.15,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "robocurate_i2i_img_novel_instruction",
            "mix_ratio": 0.25,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "robocurate_seen_img_novel_instruction",
            "mix_ratio": 0.1,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
    ],
    "rldx1_midtrain_droid": [
        {
            "dataset_name": "droid_formidtrain_0328",
            "mix_ratio": 0.8,
            "embodiment_tag": EmbodimentTag.OXE_DROID,
        },
        {
            "dataset_name": "inhouse_myungkyu_droid_format_0328_fix",
            "mix_ratio": 0.2,
            "embodiment_tag": EmbodimentTag.OXE_DROID,
        },
    ],
    "rldx1_midtrain_droid_debug": [
        {
            "dataset_name": "droid_formidtrain_0328",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.OXE_DROID,
        },
    ],
    "gr1_tabletop_1000demo": [
        {
            "dataset_name": "gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
    ],
    "gr1_tabletop_1000demo_clean": [
        {
            "dataset_name": "gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        },
    ],
}
