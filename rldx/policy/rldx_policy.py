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

"""RLDX Policy implementation for inference.

This module provides the core policy classes for running RLDX models:
- RLDXPolicy: Base policy class for model inference
- RLDXSimPolicyWrapper: Wrapper for compatibility with existing RLDX simulation environments
"""

from typing import Any

import numpy as np
import torch
import transformers.modeling_utils as mu


# Register a string alias for torch.complex64 on the transformers side
# (the RoPE `freqs_cis_*` buffers are complex64, and without this the
# from_pretrained path fails to match the dtype). This monkeypatch must
# run before the remaining imports that end up resolving the dtype map,
# so the subsequent imports are intentionally past the top of the file.
mu.str_to_torch_dtype.setdefault("C64", torch.complex64)

from rldx.data.embodiment_tags import EmbodimentTag  # noqa: E402
from rldx.data.types import ModalityConfig  # noqa: E402

from .observation_validator import ObservationValidator  # noqa: E402
from .policy import BasePolicy, PolicyWrapper  # noqa: E402
from .policy_loader import PolicyLoader, RTCOverrides  # noqa: E402
from .policy_runtime import PolicyRuntime  # noqa: E402
from .session_registry import ResetScope, SessionRegistry  # noqa: E402
from .step_request import decode_options_to_step_request  # noqa: E402


class RLDXPolicy(BasePolicy):
    """Core policy class for RLDX model inference.

    This policy handles the end-to-end inference pipeline:
    1. Validates input observations
    2. Processes observations with pretrained VLA processor
    3. Runs model inference
    4. Decodes and returns actions

    The policy expects observations with specific modalities (video, state, language)
    and returns actions in the format defined by the model's modality configuration.
    """

    def __init__(
        self,
        embodiment_tag: EmbodimentTag,
        model_path: str,
        *,
        device: int | str,
        strict: bool = True,
        sample_timestep_from_beta_dist: bool = False,
        denoising_timesteps: list[float] | None = None,
        verbose: bool = False,
        deactivate_memory: bool = False,
        require_physics: bool = False,
        rtc_inference_mode: str | None = None,
        rtc_inference_delay: int | None = None,
        rtc_inference_exec_horizon: int | None = None,
        rtc_jacobian_beta: float | None = None,
        rtc_jacobian_steps_only: int | None = None,
    ):
        """Initialize the RLDX Policy.

        Args:
            embodiment_tag: The embodiment tag defining the robot/environment type
            model_path: Path to the pretrained model checkpoint directory
            device: Device to run the model on (e.g., 'cuda:0', 0, 'cpu')
            strict: Whether to enforce strict input validation (default: True)
            rtc_inference_mode / rtc_inference_delay / rtc_inference_exec_horizon /
                rtc_jacobian_beta / rtc_jacobian_steps_only:
                Optional overrides for the model's RTC configuration. If provided,
                these are written onto ``model.config`` AFTER load so you can
                enable Real-Time Chunking on a checkpoint that was trained
                without it.
        """
        # Import this to register all models.
        import rldx.model  # noqa: F401

        super().__init__(strict=strict)
        self.verbose = verbose
        self._infer_lock = __import__("threading").Lock()
        # SessionRegistry: single owner of per-session mutable state.
        # Unbounded — lifetime is caller-managed via reset()/drop()/clear().
        self.registry: SessionRegistry = SessionRegistry()

        # Delegate ckpt / processor / config resolution to PolicyLoader.
        # Loader handles: model load (with fallback), processor load, physics
        # delta filter, RTC overrides, memory config.
        loader_result = PolicyLoader.load(
            embodiment_tag=embodiment_tag,
            model_path=model_path,
            device=device,
            deactivate_memory=deactivate_memory,
            sample_timestep_from_beta_dist=sample_timestep_from_beta_dist,
            denoising_timesteps=denoising_timesteps,
            rtc_overrides=RTCOverrides(
                mode=rtc_inference_mode,
                delay=rtc_inference_delay,
                exec_horizon=rtc_inference_exec_horizon,
                jacobian_beta=rtc_jacobian_beta,
                jacobian_steps_only=rtc_jacobian_steps_only,
            ),
        )
        self.model = loader_result.model
        self.processor = loader_result.processor
        self.embodiment_tag = loader_result.embodiment_tag
        self.modality_configs = loader_result.modality_configs
        self.collate_fn = loader_result.collate_fn
        self.physics_keys = loader_result.physics_keys
        self.require_physics = require_physics
        self.language_key = loader_result.language_key
        self.rtc_inference_mode = loader_result.rtc_inference_mode
        self.rtc_inference_delay = loader_result.rtc_inference_delay
        self.rtc_exec_horizon = loader_result.rtc_exec_horizon
        self._rtc_enabled = loader_result.rtc_enabled
        self.use_memory = loader_result.use_memory

        # Observation / action validator — owns the wire-boundary shape checks.
        # BasePolicy.check_observation / .check_action are delegated here.
        # use_memory is forwarded so the video shape-mismatch hint can
        # explain the buffering contract only when it actually applies.
        # Wire-boundary validation: unpadded per-key dims (not model.config.max_*_dim).
        norm_params = self.processor.state_action_processor.norm_params[self.embodiment_tag.value]
        expected_state_dims = {
            k: norm_params["state"][k]["dim"].item()
            for k in self.modality_configs["state"].modality_keys
        }
        expected_action_dims = {
            k: norm_params["action"][k]["dim"].item()
            for k in self.modality_configs["action"].modality_keys
        }
        self.validator: ObservationValidator = ObservationValidator(
            modality_configs=self.modality_configs,
            require_physics=self.require_physics,
            physics_keys=self.physics_keys,
            use_memory=self.use_memory,
            expected_state_dims=expected_state_dims,
            expected_action_dims=expected_action_dims,
        )

        # PolicyRuntime — orchestrates the inference pipeline. Dependencies
        # injected explicitly so Runtime can be unit-tested with stubs or
        # swapped for alternate pipelines without subclassing RLDXPolicy.
        self.runtime: PolicyRuntime = PolicyRuntime(
            model=self.model,
            processor=self.processor,
            modality_configs=self.modality_configs,
            embodiment_tag=self.embodiment_tag,
            collate_fn=self.collate_fn,
            language_key=self.language_key,
            registry=self.registry,
            infer_lock=self._infer_lock,
            use_memory=self.use_memory,
            rtc_inference_mode=self.rtc_inference_mode,
            rtc_inference_delay=self.rtc_inference_delay,
            rtc_exec_horizon=self.rtc_exec_horizon,
            rtc_enabled=self._rtc_enabled,
            verbose=self.verbose,
        )

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Thin facade — delegates to PolicyRuntime.step.

        The full pipeline (unbatch → process → RTC → inference → decode)
        lives in ``PolicyRuntime``. Here we only decode the options dict
        into a typed StepRequest (wire boundary) and hand off.
        """
        if self.verbose and options is not None:
            print("\n[SERVER-LOG] === Options received ===")
            print(f"[SERVER-LOG] {options}")
        request = decode_options_to_step_request(observation, options)
        return self.runtime.step(request)

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate observation structure + dtypes + shapes.

        Delegates to :class:`ObservationValidator`. Required to satisfy the
        BasePolicy abstract interface; the implementation lives in
        ``rldx/policy/observation_validator.py``.
        """
        self.validator.check_observation(observation)

    # Backward compat for external tests that referenced the old location.
    # Implementation lives on PolicyRuntime; exposed here as a staticmethod
    # alias so `RLDXPolicy._select_inference_context(...)` still works.
    _select_inference_context = staticmethod(PolicyRuntime._select_inference_context)

    def check_action(self, action: dict[str, Any]) -> None:
        """Validate action structure + dtypes + shapes.

        Delegates to :class:`ObservationValidator`. See ``check_observation``
        for rationale.
        """
        self.validator.check_action(action)

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.modality_configs

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset session state held in the registry.

        Two modes:
          - ``options["session_ids"]`` present → EPISODE-reset only those sids
            (keeps their registry entries but clears memory_tokens and rtc_chunk).
          - Otherwise → clear the entire registry.

        Per-call reset during inference is the primary path — see
        ``_get_action`` where ``options["reset_memory"]`` mask triggers
        per-sample EPISODE reset inline. This method is the explicit
        out-of-band reset entry point for callers that prefer a dedicated
        call over the options-dict channel.

        Args:
            options: Optional dict. Recognised keys:
                - ``session_ids``: list[str] — sids to reset. If absent, all
                  sessions in the registry are cleared.

        Returns:
            ``{"cleared_sessions": [sid, ...]}`` listing sids that were touched.
        """
        if options is None or "session_ids" not in options:
            cleared = self.registry.clear()
            return {"cleared_sessions": cleared}
        sids = options["session_ids"]
        if not isinstance(sids, list):
            sids = list(sids)
        touched = self.registry.reset(sids, ResetScope.EPISODE)
        return {"cleared_sessions": touched}


class RLDXSimPolicyWrapper(PolicyWrapper):
    """Wrapper for RLDXPolicy to enable compatibility with existing RLDX simulation environments.

    This wrapper is specifically designed for retro-fitting the RLDX policy with the current
    RLDX simulation environment interface. It handles the transformation between the flat
    observation format used by RLDX sim environments (with keys like 'video.camera_name',
    'state.joint_positions') and the nested format expected by RLDXPolicy.

    **Important**: If you are using other environments, custom robots, or building new environments,
    you should use `RLDXPolicy` directly and format your observations according to its interface.
    This wrapper is only needed for compatibility with the existing RLDX sim infrastructure.

    Key transformations performed by this wrapper:
    - Observation keys: 'video.cam' -> observation['video']['cam']
    - Observation keys: 'state.joints' -> observation['state']['joints']
    - Language keys: 'task' or 'annotation.human.coarse_action' -> observation['language']['task']
    - Action keys: action['joints'] -> 'action.joints'
    """

    def __init__(self, policy: RLDXPolicy, *, strict: bool = True):
        """Initialize the wrapper around a RLDXPolicy instance.

        Args:
            policy: The RLDXPolicy instance to wrap
            strict: Whether to enforce strict validation (default: True)
        """
        super().__init__(policy, strict=strict)
        self.policy: RLDXPolicy = policy
        assert len(self.policy.modality_configs["language"].delta_indices) == 1, (
            "Only one language delta index is supported"
        )

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate observation from RLDX sim environment format.

        This validation is specific to the flat observation format used by RLDX sim environments.
        Unlike RLDXPolicy.check_observation which expects nested dicts, this expects flat keys.

        Expected observation structure (RLDX sim format):
            - Flat keys like 'video.camera_name': np.ndarray[np.uint8, (B, T, H, W, C)]
            - Flat keys like 'state.state_name': np.ndarray[np.float32, (B, T, D)]
            - Language keys: tuple[str] or list[str] with shape (B,)
                - Key can be 'task' or 'annotation.human.coarse_action' (for DC envs)

        Args:
            observation: Flat observation dictionary from RLDX sim environment

        Raises:
            AssertionError: If any validation check fails
        """
        modality_configs = self.get_modality_config()

        # ===== LIBERO KEY MAPPING =====
        if "state.x" in observation:
            if all(key in observation for key in ["state.x", "state.y", "state.z"]):
                x, y, z = observation["state.x"], observation["state.y"], observation["state.z"]
                observation["state.eef_pos_absolute"] = np.concatenate([x, y, z], axis=-1)
            if all(key in observation for key in ["state.roll", "state.pitch", "state.yaw"]):
                roll, pitch, yaw = (
                    observation["state.roll"],
                    observation["state.pitch"],
                    observation["state.yaw"],
                )
                observation["state.eef_rot_absolute"] = np.concatenate([roll, pitch, yaw], axis=-1)
            if "state.gripper" in observation:
                observation["state.gripper_close"] = observation["state.gripper"]
            if "video.image" in observation:
                observation["video.front_view"] = observation["video.image"]
            if "video.wrist_image" in observation:
                observation["video.left_wrist_view"] = observation["video.wrist_image"]

        # ===== VIDEO VALIDATION =====
        # Check video modalities with flat key format: 'video.camera_name'
        for video_key in modality_configs["video"].modality_keys:
            # Construct flat key expected in RLDX sim environment
            parsed_key = f"video.{video_key}"

            if parsed_key not in observation:
                if (
                    video_key == "ego_view"
                    and "video.ego_view_bg_crop_pad_res256_freq20" in observation
                ):
                    observation[parsed_key] = observation[
                        "video.ego_view_bg_crop_pad_res256_freq20"
                    ]
                elif video_key == "ego_view" and "video.ego_view_pad_res256_freq20" in observation:
                    observation[parsed_key] = observation["video.ego_view_pad_res256_freq20"]
                elif video_key == "left_view" and "video.res256_image_side_0" in observation:
                    observation[parsed_key] = observation["video.res256_image_side_0"]
                elif video_key == "right_view" and "video.res256_image_side_1" in observation:
                    observation[parsed_key] = observation["video.res256_image_side_1"]
                elif video_key == "wrist_view" and "video.res256_image_wrist_0" in observation:
                    observation[parsed_key] = observation["video.res256_image_wrist_0"]
                else:
                    raise AssertionError(f"Video key '{parsed_key}' must be in observation")

            batched_video = observation[parsed_key]

            # Verify data type is numpy array
            assert isinstance(batched_video, np.ndarray), (
                f"Video key '{video_key}' must be a numpy array. Got {type(batched_video)}"
            )

            # Verify dtype is uint8 (standard for image data, range 0-255)
            assert batched_video.dtype == np.uint8, (
                f"Video key '{video_key}' must be a numpy array of type np.uint8. Got {batched_video.dtype}"
            )

            # Verify shape has 5 dimensions: (B, T, H, W, C)
            assert batched_video.ndim == 5, (
                f"Video key '{video_key}' must be a numpy array of shape (B, T, H, W, C), got {batched_video.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_video.shape[1] == len(modality_configs["video"].delta_indices), (
                f"Video key '{video_key}'s horizon must be {len(modality_configs['video'].delta_indices)}. Got {batched_video.shape[1]}"
            )

            # Verify channel dimension is 3 (RGB images)
            assert batched_video.shape[-1] == 3, (
                f"Video key '{video_key}'s channel 'C' must be 3. Got {batched_video.shape[-1]}"
            )

        # ===== STATE VALIDATION =====
        # Check state modalities with flat key format: 'state.state_name'
        for state_key in modality_configs["state"].modality_keys:
            # Construct flat key expected in RLDX sim environment
            parsed_key = f"state.{state_key}"
            assert parsed_key in observation, f"State key '{parsed_key}' must be in observation"

            batched_state = observation[parsed_key]

            # Verify data type is numpy array
            assert isinstance(batched_state, np.ndarray), (
                f"State key '{state_key}' must be a numpy array. Got {type(batched_state)}"
            )

            # Verify dtype is float32 (standard for continuous state values)
            assert batched_state.dtype == np.float32, (
                f"State key '{state_key}' must be a numpy array of type np.float32. Got {batched_state.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert batched_state.ndim == 3, (
                f"State key '{state_key}' must be a numpy array of shape (B, T, D), got {batched_state.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_state.shape[1] == len(modality_configs["state"].delta_indices), (
                f"State key '{state_key}'s horizon must be {len(modality_configs['state'].delta_indices)}. Got {batched_state.shape[1]}"
            )

        # ===== LANGUAGE VALIDATION =====
        # Check language modalities (special handling for DC environment compatibility)
        for language_key in modality_configs["language"].modality_keys:
            # PATCH: Legacy compatibility for DC environments
            # DC envs use 'annotation.human.coarse_action' instead of 'task'
            if language_key == "task" and "annotation.human.coarse_action" in observation:
                language_key = "annotation.human.coarse_action"
            # /PATCH

            # Check that the expected language key exists
            assert language_key in observation, (
                f"Language key '{language_key}' must be in observation"
            )

            # In RLDX sim format, language is a tuple of strings (B,)
            batched_language: tuple[str] | list[str] = observation[language_key]  # (B,)

            # Verify outer structure is a tuple (batch dimension)
            assert isinstance(batched_language, (tuple, list)), (
                f"Language key '{language_key}' must be a tuple or list. Got {type(batched_language)}"
            )

            # Verify each batch item is a string
            assert isinstance(batched_language[0], str), (
                f"Language batch item must be a string. Got {type(batched_language[0])}"
            )

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Transform RLDX sim observation format and compute actions.

        This method transforms the flat observation format from RLDX sim environments
        into the nested format expected by RLDXPolicy, computes actions, and transforms
        them back to the flat format expected by RLDX sim environments.

        Input format (RLDX sim):
            - Flat keys: 'video.camera_name', 'state.state_name'
            - Language: tuple[str] (B,)

        Output format (RLDX sim):
            - Flat keys: 'action.action_name'

        Args:
            observation: Flat observation dictionary from RLDX sim environment
            options: Optional parameters (currently unused)

        Returns:
            Tuple of (flat_actions_dict, info_dict)
        """
        # ===== LIBERO KEY MAPPING =====
        if "state.x" in observation:
            if all(key in observation for key in ["state.x", "state.y", "state.z"]):
                x, y, z = observation["state.x"], observation["state.y"], observation["state.z"]
                observation["state.eef_pos_absolute"] = np.concatenate([x, y, z], axis=-1)
            if all(key in observation for key in ["state.roll", "state.pitch", "state.yaw"]):
                roll, pitch, yaw = (
                    observation["state.roll"],
                    observation["state.pitch"],
                    observation["state.yaw"],
                )
                observation["state.eef_rot_absolute"] = np.concatenate([roll, pitch, yaw], axis=-1)
            if "state.gripper" in observation:
                observation["state.gripper_close"] = observation["state.gripper"]
            if "video.image" in observation:
                observation["video.front_view"] = observation["video.image"]
            if "video.wrist_image" in observation:
                observation["video.left_wrist_view"] = observation["video.wrist_image"]

        # ===== GR-1 / RoboCasa video key fallbacks =====
        # The GR-1 sim env (GrootRoboCasaEnv) and RoboCasa365 produce camera
        # keys with long suffixes (e.g. ``video.ego_view_pad_res256_freq20``),
        # but the policy's modality_config uses the short canonical names
        # (``video.ego_view``, ``video.left_view``, …). When ``--no-strict``
        # is set, ``check_observation`` is skipped, so the fallback that
        # used to live there must run unconditionally before the nested
        # transform below or _get_action raises KeyError on ``video.<short>``.
        for short_key, candidates in (
            (
                "video.ego_view",
                ("video.ego_view_bg_crop_pad_res256_freq20", "video.ego_view_pad_res256_freq20"),
            ),
            ("video.left_view", ("video.res256_image_side_0",)),
            ("video.right_view", ("video.res256_image_side_1",)),
            ("video.wrist_view", ("video.res256_image_wrist_0",)),
        ):
            if short_key not in observation:
                for c in candidates:
                    if c in observation:
                        observation[short_key] = observation[c]
                        break

        # Transform flat observation format to nested format expected by RLDXPolicy
        new_obs = {}
        for modality in ["video", "state", "language"]:
            new_obs[modality] = {}
            for key in self.policy.modality_configs[modality].modality_keys:
                if modality == "language":
                    # PATCH: Legacy compatibility for DC environments
                    if key == "task" and "annotation.human.coarse_action" in observation:
                        parsed_key = "annotation.human.coarse_action"
                    # /PATCH
                    else:
                        parsed_key = key
                else:
                    # Construct flat key (e.g., 'video.camera' or 'state.joints')
                    parsed_key = f"{modality}.{key}"

                arr = observation[parsed_key]

                # Transform to nested format
                if modality == "language":
                    # Convert from tuple[str] or list[str] (B,) to list[list[str]] (B, 1)
                    # Each element becomes a list with one string for temporal dimension
                    new_obs[modality][key] = [[str(item)] for item in arr]
                else:
                    # Video and state arrays are already in correct format (B, T, ...)
                    new_obs[modality][key] = arr

        # Compute actions using the underlying RLDXPolicy
        action, info = self.policy.get_action(new_obs, options)

        # Transform actions back to flat format
        is_libero = "state.x" in observation or "video.image" in observation

        if is_libero:
            flat_actions = {}
            if "eef_pos_delta" in action:
                pos_delta = action["eef_pos_delta"]
                flat_actions["action.x"] = pos_delta[..., 0:1]
                flat_actions["action.y"] = pos_delta[..., 1:2]
                flat_actions["action.z"] = pos_delta[..., 2:3]
            if "eef_rot_delta" in action:
                rot_delta = action["eef_rot_delta"]
                flat_actions["action.roll"] = rot_delta[..., 0:1]
                flat_actions["action.pitch"] = rot_delta[..., 1:2]
                flat_actions["action.yaw"] = rot_delta[..., 2:3]
            if "gripper_close" in action:
                flat_actions["action.gripper"] = 1.0 - action["gripper_close"]
        else:
            flat_actions = {f"action.{key}": action[key] for key in action}

        return flat_actions, info

    def check_action(self, action: dict[str, Any]) -> None:
        """Validate action in RLDX sim environment format.

        This validation is specific to the flat action format used by RLDX sim environments.
        Unlike RLDXPolicy.check_action which expects nested dicts, this expects flat keys.

        Expected action structure (RLDX sim format):
            - Flat keys like 'action.action_name': np.ndarray[np.float32, (B, T, D)]
                - B: batch size
                - T: action horizon (number of future action steps)
                - D: action dimension

        Args:
            action: Flat action dictionary for RLDX sim environment

        Raises:
            AssertionError: If any validation check fails
        """
        modality_configs = self.get_modality_config()

        # Validate each action key defined in the modality config
        for action_key in modality_configs["action"].modality_keys:
            # Construct flat key expected in RLDX sim environment (e.g., 'action.joints')
            parsed_key = f"action.{action_key}"
            assert parsed_key in action, f"Action key '{parsed_key}' must be in action"

            action_arr = action[parsed_key]

            # Verify data type is numpy array
            assert isinstance(action_arr, np.ndarray), (
                f"Action key '{action_key}' must be a numpy array. Got {type(action_arr)}"
            )

            # Verify dtype is float32 (standard for continuous actions)
            assert action_arr.dtype == np.float32, (
                f"Action key '{action_key}' must be a numpy array of type np.float32. Got {action_arr.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert action_arr.ndim == 3, (
                f"Action key '{action_key}' must be a numpy array of shape (B, T, D), got {action_arr.shape}"
            )

            # Verify action horizon matches the expected temporal dimension from config
            assert action_arr.shape[1] == len(modality_configs["action"].delta_indices), (
                f"Action key '{action_key}'s horizon must be {len(modality_configs['action'].delta_indices)}. Got {action_arr.shape[1]}"
            )

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        """Get the modality configuration from the underlying policy.

        Returns:
            Dictionary mapping modality names to their configurations
        """
        return self.policy.get_modality_config()
