# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified from the original NVIDIA Isaac GR00T v1.7.
# Original source: https://github.com/NVIDIA/Isaac-GR00T

"""
Embodiment tags are used to identify the robot embodiment in the data.

Naming convention:
<dataset>_<robot_name>

If using multiple datasets, e.g. sim GR1 and real GR1, we can drop the dataset name and use only the robot name.
"""

from enum import Enum


class EmbodimentTag(Enum):
    # New embodiment during post-training
    GENERAL_EMBODIMENT = "general_embodiment"
    ##### Pretrain embodiment tags #####
    OXE_FRACTAL = "fractal20220817_data"
    OXE_KUKA = "kuka"
    OXE_BRIDGE_ORIG = "bridge_orig"
    OXE_TACO = "taco_play"
    OXE_JACO = "jaco_play"
    OXE_BERKELEY_CABLE_ROUTING = "berkeley_cable_routing"
    OXE_ROBOTURK = "roboturk"
    OXE_VIOLA = "viola"
    OXE_BERKELEY_AUTOLAB_UR5 = "berkeley_autolab_ur5"
    OXE_TOTO = "toto"
    OXE_LANGUAGE_TABLE = "language_table"
    OXE_STANFORD_HYDRA = "stanford_hydra_dataset_converted_externally_to_rlds"
    OXE_AUSTIN_BUDS = "austin_buds_dataset_converted_externally_to_rlds"
    OXE_NYU_FRANKA_PLAY = "nyu_franka_play_dataset_converted_externally_to_rlds"
    OXE_FURNITURE_BENCH = "furniture_bench_dataset_converted_externally_to_rlds"
    OXE_UCSD_KITCHEN = "ucsd_kitchen_dataset_converted_externally_to_rlds"
    OXE_AUSTIN_SAILOR = "austin_sailor_dataset_converted_externally_to_rlds"
    OXE_AUSTIN_SIRIUS = "austin_sirius_dataset_converted_externally_to_rlds"
    OXE_DLR_EDAN_SHARED_CONTROL = "dlr_edan_shared_control_converted_externally_to_rlds"
    OXE_IAMLAB_CMU_PICKUP_INSERT = "iamlab_cmu_pickup_insert_converted_externally_to_rlds"
    OXE_UTAUSTIN_MUTEX = "utaustin_mutex"
    OXE_BERKELEY_FANUC_MANIPULATION = "berkeley_fanuc_manipulation"
    OXE_CMU_STRETCH = "cmu_stretch"
    OXE_BC_Z = "bc_z"
    OXE_FMB_DATASET = "fmb_dataset"
    OXE_DOBBE = "dobbe"
    OXE_DROID = "droid"

    AGIBOT_DEXHAND = "agibot_dexhand"
    AGIBOT_GRIPPER = "agibot_gripper"
    GALAXEA = "galaxea"

    HUMANOID_EVERYDAY_G1 = "humanoid_everyday_g1"
    HUMANOID_EVERYDAY_H1 = "humanoid_everyday_h1"
    ACTION_NET = "action_net"
    NEURAL_GR1 = "neural_gr1"
    NEW_EMBODIMENT = "new_embodiment"

    # Sim / benchmark embodiment tags (includes robocasa_panda_omron, oxe_google, ...)
    ROBOCASA_PANDA_OMRON = "robocasa_panda_omron"
    OXE_GOOGLE = "oxe_google"
    OXE_WIDOWX = "oxe_widowx"
    LIBERO_PANDA = "libero_panda"
    BEHAVIOR_R1_PRO = "behavior_r1_pro"
    UNITREE_G1 = "unitree_g1"
    GR1 = "gr1"


def get_embodimenttag_by_name(name: str):
    """
    Returns the EmbodimentTag enum member that matches the given value string.
    Returns None if no match is found.
    """
    for tag in EmbodimentTag:
        if tag.value == name:
            return tag
    return None
