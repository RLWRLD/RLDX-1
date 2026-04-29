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

import os

from rldx.data.embodiment_tags import EmbodimentTag


def is_groot_locomanip_env(env_name: str) -> bool:
    return env_name.startswith("gr00tlocomanip")


def is_behavior_env(env_name: str) -> bool:
    return env_name.startswith("sim_behavior_r1_pro")


def is_gr1_env(env_name: str) -> bool:
    """ensures gr1 and gr1_unified are the same embodiment tag"""
    return env_name.startswith("gr1") or env_name.startswith("gr1_unified")


def is_robocasa365_env(env_name: str) -> bool:
    return env_name.startswith("robocasa/")


def is_robocasa_kitchen_env(env_name: str) -> bool:
    """Match RoboCasa-Kitchen tasks (``robocasa_panda_omron/<TASK>_PandaOmron_Env``)."""
    return env_name.startswith("robocasa_panda_omron/")


def is_libero_env(env_name: str) -> bool:
    return env_name.startswith("libero_sim/")


def is_libero_plus_env(env_name: str) -> bool:
    return env_name.startswith("libero_plus_sim/")


def is_simpler_google_env(env_name: str) -> bool:
    return env_name.startswith("simpler_env_google/")


def is_simpler_widowx_env(env_name: str) -> bool:
    return env_name.startswith("simpler_env_widowx/")


def get_embodiment_tag_from_env_name(env_name: str) -> EmbodimentTag:
    if is_robocasa365_env(env_name) or is_robocasa_kitchen_env(env_name):
        # BJ: ALINVLA 는 GENERAL_EMBODIMENT 태그를 사용하는 것이 기본값, 만약에 GR00T N1.6 를 RoboCasa365 에서 평가할 경우에는 IS_RLDX_EVAL=True 로 설정하면
        # BJ: ROBOCASA_PANDA_OMRON 로 조정합니다.
        is_rldx_eval = os.getenv("IS_RLDX_EVAL", False)
        if is_rldx_eval:
            return EmbodimentTag.ROBOCASA_PANDA_OMRON
        else:
            return EmbodimentTag.GENERAL_EMBODIMENT

    if is_groot_locomanip_env(env_name):
        groot_locomanip_mappings = {
            "gr00tlocomanip_g1": EmbodimentTag.UNITREE_G1,
            "gr00tlocomanip_g1_sim": EmbodimentTag.UNITREE_G1,
            "gr00tlocomanip_g1_new": EmbodimentTag.UNITREE_G1,
        }
        return groot_locomanip_mappings[env_name.split("/")[0]]

    if is_behavior_env(env_name):
        return EmbodimentTag.BEHAVIOR_R1_PRO

    if is_gr1_env(env_name):
        # RLDX-1-FT-GR1 was trained with GENERAL_EMBODIMENT (post v2 swap),
        # not the legacy GR00T-style GR1 tag. Default to GE so the
        # production RLWRLD ckpt routes correctly; users evaluating a
        # legacy GR00T checkpoint that still carries the GR1 modality_config
        # can opt in via IS_RLDX_EVAL=1 to restore the original behavior.
        is_rldx_eval = os.getenv("IS_RLDX_EVAL", False)
        return EmbodimentTag.GR1 if is_rldx_eval else EmbodimentTag.GENERAL_EMBODIMENT

    if is_libero_env(env_name) or is_libero_plus_env(env_name):
        return EmbodimentTag.GENERAL_EMBODIMENT

    if is_simpler_google_env(env_name) or is_simpler_widowx_env(env_name):
        return EmbodimentTag.OXE_BRIDGE_ORIG

    return EmbodimentTag(env_name.split("/")[0])
