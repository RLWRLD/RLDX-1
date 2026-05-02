# SPDX-License-Identifier: Apache-2.0
"""
ObservationValidator — wire-boundary validation for RLDXPolicy inputs.

Before this extraction, `check_observation` (170+ LOC) and `check_action`
(40 LOC) lived as methods on the RLDXPolicy god-class. Three issues with that
placement:
  - Policy's public surface was dominated by validation noise.
  - Options dict validation (sids / reset_mask / action_prefix) was scattered
    across `_get_action` — no single boundary.
  - Same validation logic would need duplication if another policy (e.g.
    a debugging wrapper) wanted to enforce the same contract.

ObservationValidator owns three related checks:
  - check_observation: obs dict shape/dtype (video / state / language / physics)
  - check_action:      action dict shape/dtype (returned by policy)
  - StepRequest.validate_shapes — delegated, invoked at `_get_action` entry

RLDXPolicy delegates check_observation/check_action here; `_get_action` calls
`decode_options_to_step_request(...)` which performs StepRequest shape
validation. Two validation entry points, one class responsible for both.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ObservationValidator:
    """Shape / dtype validator for policy inputs and outputs.

    Stateless w.r.t. runtime — the only "state" is the set of modality
    configs and physics requirements resolved at policy load time.
    """

    def __init__(
        self,
        modality_configs,
        require_physics: bool,
        physics_keys: list[str],
        use_memory: bool = False,
        expected_state_dims: dict[str, int] | None = None,
        expected_action_dims: dict[str, int] | None = None,
    ) -> None:
        self.modality_configs = modality_configs
        self.require_physics = require_physics
        self.physics_keys = physics_keys
        # use_memory toggles the memory+video buffering hint on video shape
        # mismatch; plain single-step policies (delta_indices=[0]) do not
        # get the hint even if use_memory=True because the contract is trivial.
        self.use_memory = use_memory
        self.expected_state_dims = expected_state_dims
        self.expected_action_dims = expected_action_dims

    # ------------------------------------------------------------------
    # check_observation
    # ------------------------------------------------------------------
    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate observation dict structure + dtypes + shapes.

        Expected observation structure:
            - video:    dict[str, np.ndarray[np.uint8, (B, T, H, W, C)]]
            - state:    dict[str, np.ndarray[np.float32, (B, T, D)]]
            - language: dict[str, list[list[str]]]  shape (B, T)
            - physics:  dict[str, ...] (optional unless require_physics)

        Raises:
            AssertionError on first violation.
        """
        # Top-level modality keys
        for modality in ("video", "state", "language"):
            assert modality in observation, f"Observation must contain a '{modality}' key"
            assert isinstance(observation[modality], dict), (
                f"Observation '{modality}' must be a dictionary. "
                f"Got {type(observation[modality])}: {observation[modality]}"
            )

        # Track batch size across modalities for consistency
        bs = -1

        # ----- VIDEO -----
        for video_key in self.modality_configs["video"].modality_keys:
            if bs == -1:
                bs = len(observation["video"][video_key])
            else:
                assert len(observation["video"][video_key]) == bs, (
                    f"Video key '{video_key}' must have batch size {bs}. "
                    f"Got {len(observation['video'][video_key])}"
                )
            assert video_key in observation["video"], (
                f"Video key '{video_key}' must be in observation"
            )
            batched_video = observation["video"][video_key]
            assert isinstance(batched_video, np.ndarray), (
                f"Video key '{video_key}' must be a numpy array. Got {type(batched_video)}"
            )
            assert batched_video.dtype == np.uint8, (
                f"Video key '{video_key}' must be a numpy array of type "
                f"np.uint8. Got {batched_video.dtype}"
            )
            assert batched_video.ndim == 5, (
                f"Video key '{video_key}' must be a numpy array of shape "
                f"(B, T, H, W, C), got {batched_video.shape}"
            )
            _video_deltas = self.modality_configs["video"].delta_indices
            if batched_video.shape[1] != len(_video_deltas):
                _msg = (
                    f"Video key '{video_key}' shape mismatch: "
                    f"got T={batched_video.shape[1]}, expected T={len(_video_deltas)} "
                    f"(delta_indices={_video_deltas})."
                )
                if self.use_memory and len(_video_deltas) > 1:
                    _msg += (
                        " This policy was loaded with memory+video enabled; "
                        "the client must buffer past frames at the action-step "
                        "offsets shown in delta_indices and send them all in "
                        "one call (oldest first, most-recent at index -1)."
                    )
                raise AssertionError(_msg)
            assert batched_video.shape[-1] == 3, (
                f"Video key '{video_key}'s channel 'C' must be 3. Got {batched_video.shape[-1]}"
            )

        # ----- STATE -----
        for state_key in self.modality_configs["state"].modality_keys:
            if bs == -1:
                bs = len(observation["state"][state_key])
            else:
                assert len(observation["state"][state_key]) == bs, (
                    f"State key '{state_key}' must have batch size {bs}. "
                    f"Got {len(observation['state'][state_key])}"
                )
            assert state_key in observation["state"], (
                f"State key '{state_key}' must be in observation"
            )
            batched_state = observation["state"][state_key]
            assert isinstance(batched_state, np.ndarray), (
                f"State key '{state_key}' must be a numpy array. Got {type(batched_state)}"
            )
            assert batched_state.dtype == np.float32, (
                f"State key '{state_key}' must be a numpy array of type "
                f"np.float32. Got {batched_state.dtype}"
            )
            assert batched_state.ndim == 3, (
                f"State key '{state_key}' must be a numpy array of shape "
                f"(B, T, D), got {batched_state.shape}"
            )
            assert batched_state.shape[1] == len(self.modality_configs["state"].delta_indices), (
                f"State key '{state_key}'s horizon must be "
                f"{len(self.modality_configs['state'].delta_indices)}. "
                f"Got {batched_state.shape[1]}"
            )
            if self.expected_state_dims is not None:
                expected = self.expected_state_dims[state_key]
                assert batched_state.shape[-1] == expected, (
                    f"State key '{state_key}' shape[-1]={batched_state.shape[-1]} "
                    f"!= expected_state_dim={expected}"
                )
            assert np.isfinite(batched_state).all(), f"Non-finite values in state key '{state_key}'"

        # ----- LANGUAGE -----
        for language_key in self.modality_configs["language"].modality_keys:
            if bs == -1:
                bs = len(observation["language"][language_key])
            else:
                assert len(observation["language"][language_key]) == bs, (
                    f"Language key '{language_key}' must have batch size "
                    f"{bs}. Got {len(observation['language'][language_key])}"
                )
            assert language_key in observation["language"], (
                f"Language key '{language_key}' must be in observation"
            )
            batched_language: list[list[str]] = observation["language"][language_key]
            assert isinstance(batched_language, list), (
                f"Language key '{language_key}' must be a list. Got {type(batched_language)}"
            )
            for batch_item in batched_language:
                assert len(batch_item) == len(self.modality_configs["language"].delta_indices), (
                    f"Language key '{language_key}'s horizon must be "
                    f"{len(self.modality_configs['language'].delta_indices)}. "
                    f"Got {len(batched_language)}"
                )
                assert isinstance(batch_item, list), (
                    f"Language batch item must be a list. Got {type(batch_item)}"
                )
                assert len(batch_item) == 1, (
                    f"Language batch item must have exactly one item. Got {len(batch_item)}"
                )
                assert isinstance(batch_item[0], str), (
                    f"Language batch item must be a string. Got {type(batch_item[0])}"
                )

        # ----- PHYSICS (optional) -----
        if self.require_physics and self.physics_keys:
            assert "physics" in observation, (
                f"Model requires physics input (physics_keys="
                f"{self.physics_keys}) but observation has no 'physics' key."
            )
            for pk in self.physics_keys:
                assert pk in observation["physics"], (
                    f"Physics key '{pk}' required but not found in "
                    f"observation['physics']. Available: "
                    f"{list(observation['physics'].keys())}"
                )
                physics_val = observation["physics"][pk]
                if isinstance(physics_val, np.ndarray):
                    assert np.isfinite(physics_val).all(), (
                        f"Non-finite values in physics key '{pk}'"
                    )

    # ------------------------------------------------------------------
    # check_action
    # ------------------------------------------------------------------
    def check_action(self, action: dict[str, Any]) -> None:
        """Validate action dict structure + dtypes + shapes.

        Expected action structure:
            - action: dict[str, np.ndarray[np.float32, (B, T, D)]]
                - B: batch size
                - T: action horizon
                - D: action dimension

        Raises:
            AssertionError on first violation.
        """
        for action_key in self.modality_configs["action"].modality_keys:
            assert action_key in action, f"Action key '{action_key}' must be in action"
            action_arr = action[action_key]
            assert isinstance(action_arr, np.ndarray), (
                f"Action key '{action_key}' must be a numpy array. Got {type(action_arr)}"
            )
            assert action_arr.dtype == np.float32, (
                f"Action key '{action_key}' must be a numpy array of type "
                f"np.float32. Got {action_arr.dtype}"
            )
            assert action_arr.ndim == 3, (
                f"Action key '{action_key}' must be a numpy array of shape "
                f"(B, T, D), got {action_arr.shape}"
            )
            assert action_arr.shape[1] == len(self.modality_configs["action"].delta_indices), (
                f"Action key '{action_key}'s horizon must be "
                f"{len(self.modality_configs['action'].delta_indices)}. "
                f"Got {action_arr.shape[1]}"
            )
            if self.expected_action_dims is not None:
                expected = self.expected_action_dims[action_key]
                assert action_arr.shape[-1] == expected, (
                    f"Action key '{action_key}' shape[-1]={action_arr.shape[-1]} "
                    f"!= expected_action_dim={expected}"
                )
            assert np.isfinite(action_arr).all(), f"Non-finite values in action key '{action_key}'"
