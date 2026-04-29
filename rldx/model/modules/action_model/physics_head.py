"""Physics conditioning + flow matching stream for RLDXActionModel."""

from typing import NamedTuple, Optional

import torch
from torch import nn
import torch.nn.functional as F

from rldx.model.modules.action_model.physics import (
    PhysicalSignalDecoder,
    PhysicalSignalEncoder,
    PhysicsNoiseEncoder,
)
from rldx.utils.dist import rank_zero_print as _print


class PhysicsInferenceState(NamedTuple):
    """Immutable state carried through Euler inference loop."""

    embs: Optional[torch.Tensor]  # current physics embeddings for MSAT
    hist_tok: Optional[torch.Tensor]  # fixed history tokens (computed once)
    fut: Optional[torch.Tensor]  # evolving future state (Euler updated)
    attn_mask: Optional[torch.Tensor]  # fixed attention mask


# --- State dict key remapping for backward compatibility ---

_PHYSICS_KEY_RENAMES = [
    ("physics_encoder.", "physics.physics_cond_encoder."),  # very old → new
    ("physics_cond_encoder.", "physics.physics_cond_encoder."),  # old → new
    ("physics_fut_encoder.", "physics.physics_fut_encoder."),  # old → new
    ("physics_decoder.", "physics.physics_decoder."),  # old → new
]


def remap_physics_keys(state_dict: dict) -> dict:
    """Remap legacy physics state_dict keys to PhysicsHead layout.

    Handles both very old (physics_encoder) and old (physics_cond_encoder) keys.
    Keys already in the new format (physics.physics_*) are left unchanged —
    including nested cases like `action_model.physics.physics_cond_encoder.*`,
    which the previous top-level `startswith("physics.")` guard missed and
    would double-prefix on every call.
    """
    remapped = {}
    renamed_count = 0
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in _PHYSICS_KEY_RENAMES:
            # Idempotence: if the target prefix is already present anywhere in
            # the key, this key is already in the new layout — do not rename.
            if new_prefix in key:
                break
            if old_prefix in key:
                new_key = key.replace(old_prefix, new_prefix, 1)
                renamed_count += 1
                break
        remapped[new_key] = value
    if renamed_count > 0:
        _print(f"[Physics] Remapped {renamed_count} legacy keys → physics.* layout")
    return remapped


class NoOpPhysicsHead(nn.Module):
    """No-op physics head. Used when use_physics=False."""

    def prepare_train(self, action_input, t_raw):
        return None, None, None

    def compute_loss(self, physics_model_output, physics_velocity, action_mask, physics_attn_mask):
        return None

    def prepare_inference(self, action_input, batch_size, device, dtype) -> PhysicsInferenceState:
        return PhysicsInferenceState(embs=None, hist_tok=None, fut=None, attn_mask=None)

    def build_tokens(self, state: PhysicsInferenceState, timesteps_tensor):
        return None

    def update_state(self, state: PhysicsInferenceState, model_output, dt) -> PhysicsInferenceState:
        return state


class PhysicsHead(nn.Module):
    """Owns physics encoder/decoder modules and related config.

    Registered as a submodule of RLDXActionModel (self.physics).
    """

    def __init__(
        self,
        physics_dim: int,
        embed_dim: int,
        msat_output_dim: int,
        physics_delta_indices: list[int] | None,
        physics_use_flow_matching: bool,
        physics_loss_weight: float,
        action_horizon: int,
        physics_dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.physics_dim = physics_dim
        self.embed_dim = embed_dim
        self.physics_loss_weight = physics_loss_weight
        self.physics_use_flow_matching = physics_use_flow_matching
        self.physics_dropout_prob = physics_dropout_prob

        delta = physics_delta_indices or []
        self.physics_hist_len = sum(1 for d in delta if d <= 0)
        self.physics_fut_len = sum(1 for d in delta if d > 0)

        self.physics_cond_encoder = PhysicalSignalEncoder(physics_dim, embed_dim, embed_dim)
        self.physics_fut_encoder = PhysicsNoiseEncoder(physics_dim, embed_dim, embed_dim)
        self.physics_decoder = PhysicalSignalDecoder(msat_output_dim, embed_dim, physics_dim)

        # Learned mask token used to replace dropped physics conditioning tokens.
        # Only created when dropout is on so checkpoints saved with prob=0 stay clean.
        self.physics_mask_token = (
            nn.Parameter(0.02 * torch.randn(1, 1, embed_dim)) if physics_dropout_prob > 0 else None
        )

        if self.physics_use_flow_matching:
            assert self.physics_fut_len == action_horizon, (
                f"physics_fut_len ({self.physics_fut_len}) must equal action_horizon "
                f"({action_horizon}) so that action_mask can be reused for physics loss masking"
            )
        else:
            _print(
                "[Physics] Flow matching disabled. "
                "Physics used as conditioning only (no prediction loss)"
            )

        _print(
            f"\n[Physics] Physics stream enabled (dim={physics_dim}, weight={physics_loss_weight})"
        )
        _print(
            f"[Physics] hist_len={self.physics_hist_len}, fut_len={self.physics_fut_len}, "
            f"flow_matching={self.physics_use_flow_matching}"
        )
        if physics_dropout_prob > 0:
            mode = "hist-only" if self.physics_use_flow_matching else "all-conditioning"
            _print(f"[Physics] physics_dropout_prob={physics_dropout_prob} ({mode})")

    def _maybe_dropout(self, tokens: torch.Tensor) -> torch.Tensor:
        """Per-sample dropout: replace a dropped sample's full token slice with
        the learned `physics_mask_token`. Active only during training."""
        if not (
            self.training
            and self.physics_dropout_prob > 0
            and self.physics_mask_token is not None
            and tokens.shape[1] > 0
        ):
            return tokens
        do_dropout = torch.rand(tokens.shape[0], device=tokens.device) < self.physics_dropout_prob
        do_dropout = do_dropout[:, None, None].to(dtype=tokens.dtype)
        return tokens * (1 - do_dropout) + self.physics_mask_token * do_dropout

    def prepare_train(self, action_input, t_raw):
        """Encode physics signal for training. Returns (physics_embs, physics_attn_mask, physics_velocity)."""
        physics_embs = None
        physics_attn_mask = None
        physics_velocity = None

        if not hasattr(action_input, "physics"):
            return physics_embs, physics_attn_mask, physics_velocity

        data_dim = action_input.physics.shape[-1]
        expected_dim = self.physics_cond_encoder.W1.in_features
        assert data_dim == expected_dim, (
            f"Physics dim mismatch: data has {data_dim} but model expects {expected_dim} "
            f"(from --physics-dims). Check that --physics-dims matches your dataset."
        )

        if hasattr(action_input, "physics_mask"):
            physics_attn_mask = action_input.physics_mask.view(-1)  # [B]

        if not self.physics_use_flow_matching:
            # Conditioning only: all tokens as conditioning. Dropout the whole sequence
            # since there's no flow-matching target to preserve.
            physics_embs = self.physics_cond_encoder(action_input.physics)
            physics_embs = self._maybe_dropout(physics_embs)
        else:
            # Flow matching: split hist/fut -> noise fut -> encode both
            physics_hist = action_input.physics[:, : self.physics_hist_len, :]  # (B, H, D)
            physics_fut_gt = action_input.physics[:, self.physics_hist_len :, :]  # (B, F, D)

            t_broad_p = t_raw[:, None, None]  # (B, 1, 1)
            physics_noise = torch.randn_like(physics_fut_gt)
            # noisy_fut = (1-t) * noise + t * gt  (flow matching interpolation)
            noisy_physics_fut = (1 - t_broad_p) * physics_noise + t_broad_p * physics_fut_gt
            physics_velocity = physics_fut_gt - physics_noise

            if self.physics_hist_len > 0:
                physics_hist_tok = self.physics_cond_encoder(physics_hist)
            else:
                physics_hist_tok = torch.zeros(
                    physics_hist.shape[0],
                    0,
                    self.embed_dim,
                    dtype=physics_hist.dtype,
                    device=physics_hist.device,
                )
            # Hist-only dropout: future tokens are the prediction target and must NOT be masked.
            physics_hist_tok = self._maybe_dropout(physics_hist_tok)
            physics_fut_tok = self.physics_fut_encoder(noisy_physics_fut, t_raw)
            physics_embs = torch.cat([physics_hist_tok, physics_fut_tok], dim=1)

        return physics_embs, physics_attn_mask, physics_velocity

    def compute_loss(self, physics_model_output, physics_velocity, action_mask, physics_attn_mask):
        """Compute physics prediction loss (flow matching only). Returns physics_loss or None."""
        if not (
            self.physics_use_flow_matching
            and physics_model_output is not None
            and physics_velocity is not None
        ):
            return None

        physics_hidden_fut = physics_model_output[:, -self.physics_fut_len :, :]
        physics_pred_vel = self.physics_decoder(physics_hidden_fut)

        # Combine per-step action_mask (episode boundary) with per-sample physics_attn_mask
        step_mask = action_mask.any(dim=-1).float()  # (B, T) — per-step validity from action
        if physics_attn_mask is not None:
            step_mask = step_mask * physics_attn_mask.unsqueeze(1)  # (B, T) * (B, 1)
        mask_3d = step_mask.unsqueeze(-1)  # (B, T, 1)

        loss_unreduced = F.mse_loss(physics_pred_vel, physics_velocity, reduction="none")
        n_valid = mask_3d.sum() * physics_pred_vel.shape[-1]
        physics_loss = (loss_unreduced * mask_3d).sum() / (n_valid + 1e-6)
        return physics_loss

    def prepare_inference(self, action_input, batch_size, device, dtype) -> PhysicsInferenceState:
        """Initialize physics state for inference."""
        embs = None
        hist_tok = None
        fut = None
        attn_mask = None

        has_physics_input = action_input is not None and hasattr(action_input, "physics")

        if has_physics_input and hasattr(action_input, "physics_mask"):
            attn_mask = action_input.physics_mask.view(-1)

        if not self.physics_use_flow_matching:
            # Conditioning only: encode all tokens once (fixed outside loop)
            if has_physics_input:
                embs = self.physics_cond_encoder(action_input.physics)
        else:
            # Flow matching: hist conditioning + fut noise init
            if has_physics_input and self.physics_hist_len > 0:
                physics_hist = action_input.physics[:, : self.physics_hist_len, :]
                hist_tok = self.physics_cond_encoder(physics_hist)
            else:
                hist_tok = torch.zeros(
                    batch_size,
                    0,
                    self.embed_dim,
                    dtype=dtype,
                    device=device,
                )
            fut = torch.randn(
                batch_size,
                self.physics_fut_len,
                self.physics_dim,
                dtype=dtype,
                device=device,
            )

        return PhysicsInferenceState(embs=embs, hist_tok=hist_tok, fut=fut, attn_mask=attn_mask)

    def build_tokens(self, state: PhysicsInferenceState, timesteps_tensor) -> torch.Tensor:
        """Build physics tokens for one Euler step. Returns updated physics_embs."""
        if state.hist_tok is not None and state.fut is not None:
            fut_tok = self.physics_fut_encoder(state.fut, timesteps_tensor)
            return torch.cat([state.hist_tok, fut_tok], dim=1)
        return state.embs

    def update_state(self, state: PhysicsInferenceState, model_output, dt) -> PhysicsInferenceState:
        """Euler update for physics future state. Returns new state with updated fut."""
        if state.fut is not None and isinstance(model_output, dict) and "physics" in model_output:
            physics_hidden_fut = model_output["physics"][:, -self.physics_fut_len :, :]
            physics_pred_vel = self.physics_decoder(physics_hidden_fut)
            return state._replace(fut=state.fut + dt * physics_pred_vel)
        return state
