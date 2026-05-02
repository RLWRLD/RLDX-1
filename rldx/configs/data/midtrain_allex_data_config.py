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

"""ALLEX mid-training modality config — chunk=40, physics hist=1 (mono video).

  - action_horizon = 40 (predict 40 future action steps per chunk)
  - physics hist = 1 (only the current timestep as history), fut = 40
    (PhysicsHead flow-matching assertion: physics_fut_len == action_horizon)

Use with `--action-horizon 40` so the assembly-time validator does not flag
a mismatch.
"""

from rldx.configs.data.embodiment_configs import register_modality_config
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


MIDTRAIN_ALLEX_MODALITY_CONFIGS = {
    "allex": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["camera_ego_left"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm_joints",
                "left_hand_joints",
                "neck_joints",
                "right_arm_joints",
                "right_hand_joints",
                "waist_joints",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(40)),
            modality_keys=[
                "left_arm_joints",
                "left_hand_joints",
                "neck_joints",
                "right_arm_joints",
                "right_hand_joints",
                "waist_joints",
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
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
        "torque": ModalityConfig(  # 48 dim
            # hist=1 (current timestep only), fut=40 (== action_horizon)
            delta_indices=list(range(0, 41)),
            modality_keys=[
                "left_arm_effort",
                "left_hand_effort",
                "neck_effort",
                "right_arm_effort",
                "right_hand_effort",
                "waist_effort",
            ],
        ),
    },
}


for name, modality_config in MIDTRAIN_ALLEX_MODALITY_CONFIGS.items():
    register_modality_config(modality_config, EmbodimentTag.GENERAL_EMBODIMENT)
