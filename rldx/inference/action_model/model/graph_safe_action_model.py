"""Graph-safe wrapper for the full v1.4 action head pipeline.

Composes GraphSafeMSAT with state/action encoders/decoders and runs
the denoising loop.  Pre-determines data-dependent variables (position IDs,
timestep schedule, dt) in __init__.

Data-dependent operations replaced:
  - torch.arange(action_horizon) per forward → static_pos_ids buffer
  - timestep schedule computation → static_timesteps buffer
  - dt = 1/N → Python float

Physics support (v1.4-all):
  When action_model.use_physics is True, the denoising loop is extended with
  physics conditioning (history) and flow-matching (future) streams.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rldx.utils.dist import rank_zero_print as _print

from .graph_safe_msat import GraphSafeMSAT


class GraphSafeActionModel(nn.Module):
    """Graph-safe full action head pipeline.

    Without physics (v1.4-v):
      vlln + state_enc(1x) + N_steps x [action_enc + MSAT + action_dec + Euler]

    With physics (v1.4-all):
      vlln + state_enc(1x) + physics_hist_enc(1x) + init physics_fut noise
      + N_steps x [action_enc + physics_fut_enc + MSAT(+physics) + action_dec
                   + Euler + physics_dec + physics_Euler]

    Accepts either:
      - action_model: RLDXActionModel (extracts sub-modules automatically)
      - Individual components via kwargs (for standalone benchmark)
    """

    def __init__(
        self,
        action_model=None,
        *,
        msat=None,
        state_encoder=None,
        action_encoder=None,
        action_decoder=None,
        position_embedding=None,
        vlln=None,
        n_vl,
        n_sa_pure,
        action_horizon,
        action_dim,
        num_inference_timesteps,
        device,
        dtype=torch.bfloat16,
        # Physics kwargs (used when action_model is None)
        physics_cond_encoder=None,
        physics_fut_encoder=None,
        physics_decoder=None,
        physics_hist_len=0,
        physics_fut_len=0,
        physics_dim=0,
        # RTC trained-mode prefix length (0 = disabled). When > 0
        # the forward applies hard prefix inpaint at initial noise,
        # before the model forward, and after the Euler step;
        # ``forward(prefix_actions=...)`` must supply the frozen
        # prefix tensor of shape ``(B, prefix_len, action_dim)``.
        prefix_len: int = 0,
    ):
        super().__init__()

        # --- Extract sub-modules from action_model or use provided kwargs ---
        # The MSAT diffusion model is registered as ``action_model.model``.
        if action_model is not None:
            msat = action_model.model
            vlln = action_model.vlln
            state_encoder = action_model.state_encoder
            action_encoder = action_model.action_encoder
            action_decoder = action_model.action_decoder
            position_embedding = action_model.position_embedding

        # --- Physics sub-modules ---
        # PhysicsHead nests them under ``action_model.physics``. Allow
        # kwargs to override dimensions for standalone benchmarks.
        use_physics = False
        if action_model is not None and getattr(action_model, "use_physics", False):
            use_physics = True
            physics = action_model.physics
            if physics_cond_encoder is None:
                physics_cond_encoder = physics.physics_cond_encoder
            if physics_fut_encoder is None:
                physics_fut_encoder = physics.physics_fut_encoder
            if physics_decoder is None:
                physics_decoder = physics.physics_decoder
            # Use kwarg values if non-default (allow override of checkpoint's 0-length physics)
            if physics_hist_len == 0:
                physics_hist_len = physics.physics_hist_len
            if physics_fut_len == 0:
                physics_fut_len = physics.physics_fut_len
            if physics_dim == 0:
                physics_dim = physics.physics_dim
        elif physics_cond_encoder is not None:
            use_physics = True

        self.use_physics = use_physics
        n_physics = physics_hist_len + physics_fut_len if use_physics else 0

        # --- GraphSafeMSAT (shared by all engine paths) ---
        self.gs_msat = GraphSafeMSAT(msat, n_vl, n_sa_pure, device, n_physics=n_physics)
        # Cache inner_dim for physics zero-tensor creation (avoids gs_msat._msat
        # access at forward time, which breaks when gs_msat is swapped for CUDA Graph/compile)
        self._inner_dim = msat.inner_dim

        # vlln: backbone output normalization
        self.vlln = vlln if vlln is not None else nn.Identity()

        # Sub-modules (weight references)
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.action_decoder = action_decoder
        self.position_embedding = position_embedding

        # Physics sub-modules
        if use_physics:
            self.physics_cond_encoder = physics_cond_encoder
            self.physics_fut_encoder = physics_fut_encoder
            self.physics_decoder = physics_decoder
            self.physics_hist_len = physics_hist_len
            self.physics_fut_len = physics_fut_len
            self.physics_dim = physics_dim

        # Static config
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.num_inference_timesteps = num_inference_timesteps
        self.dt = 1.0 / num_inference_timesteps
        self.prefix_len = int(prefix_len)

        # Static position IDs for position_embedding lookup
        pos_ids = torch.arange(action_horizon, dtype=torch.long, device=device)
        self.register_buffer("static_pos_ids", pos_ids)

        # Static timestep schedule
        timesteps = torch.tensor(
            [t / float(num_inference_timesteps) for t in range(num_inference_timesteps)],
            dtype=dtype,
            device=device,
        )
        self.register_buffer("static_timesteps", timesteps)

        _print(
            f"  [GraphSafeActionModel] pos_ids={list(pos_ids.shape)}, "
            f"timesteps={list(timesteps.shape)}, "
            f"action_horizon={action_horizon}, "
            f"num_steps={num_inference_timesteps}, dt={self.dt}"
        )
        if use_physics:
            _print(
                f"  [GraphSafeActionModel] physics: dim={physics_dim}, "
                f"hist_len={physics_hist_len}, fut_len={physics_fut_len}, "
                f"n_physics={n_physics}"
            )
        if self.prefix_len > 0:
            _print(f"  [GraphSafeActionModel] RTC trained: prefix_len={self.prefix_len}")

    def forward(
        self,
        vl_embs,
        state,
        embodiment_id,
        init_noise=None,
        physics_hist=None,
        physics_init_noise=None,
        prefix_actions=None,
    ):
        """Graph-safe action head forward.

        Args:
            vl_embs: (B, n_vl, vl_dim) — VL embeddings from backbone
            state: (B, state_dim) — current state observation
            embodiment_id: (B,) — embodiment IDs
            init_noise: (B, action_horizon, action_dim) or None — action noise
            physics_hist: (B, hist_len, physics_dim) or None — physics history
            physics_init_noise: (B, fut_len, physics_dim) or None — physics noise
            prefix_actions: (B, prefix_len, action_dim) frozen actions from
                the previous chunk (RTC trained mode). Required when
                ``prefix_len > 0``; ignored otherwise.

        Returns:
            current_state: (B, action_horizon, action_dim) — predicted actions
        """
        # Backbone output normalization
        vl_embs = self.vlln(vl_embs)

        B = vl_embs.shape[0]
        d = self.prefix_len

        # State encoding (once, before loop)
        state_features = self.state_encoder(state, embodiment_id)

        # Position embedding (static IDs)
        pos_embs = self.position_embedding(self.static_pos_ids).unsqueeze(0)

        # Init action noise; trained-mode prefix is locked to the frozen
        # ground-truth prefix from the previous chunk.
        if init_noise is not None:
            current_state = init_noise.clone() if d > 0 else init_noise
        else:
            current_state = torch.randn(
                (B, self.action_horizon, self.action_dim),
                dtype=vl_embs.dtype,
                device=vl_embs.device,
            )
        if d > 0:
            current_state[:, :d] = prefix_actions

        # Physics setup (if applicable)
        physics_hist_tok = None
        physics_fut = None
        if self.use_physics:
            # Encode history (once, before loop)
            if physics_hist is not None and self.physics_hist_len > 0:
                physics_hist_tok = self.physics_cond_encoder(physics_hist)
            else:
                physics_hist_tok = torch.zeros(
                    B,
                    0,
                    self._inner_dim,
                    dtype=vl_embs.dtype,
                    device=vl_embs.device,
                )

            # Init future noise
            if physics_init_noise is not None:
                physics_fut = physics_init_noise
            else:
                physics_fut = torch.randn(
                    B,
                    self.physics_fut_len,
                    self.physics_dim,
                    dtype=vl_embs.dtype,
                    device=vl_embs.device,
                )

        # Denoising loop
        for t in range(self.num_inference_timesteps):
            t_scalar = self.static_timesteps[t].expand(B)

            # Per-token time tensor — always (B, action_horizon) so any
            # downstream fused variant of ``action_encoder`` is forced to
            # honour position-wise time.  In trained mode the prefix uses
            # t=1 (clean ground truth) and the postfix the standard
            # t_scalar; without prefix it is a uniform broadcast.
            t_tok = t_scalar.unsqueeze(1).expand(-1, self.action_horizon).clone()
            if d > 0:
                t_tok[:, :d] = 1.0
                # Re-lock prefix before the model sees the trajectory —
                # required for both numerical stability and to prevent
                # step-wise drift from leaking into the frozen window.
                current_state[:, :d] = prefix_actions

            action_features = self.action_encoder(
                current_state,
                t_tok,
                embodiment_id,
            )
            action_features = action_features + pos_embs

            # Concatenate SA tokens: [state(1), action(N_action)]
            sa_embs = torch.cat([state_features, action_features], dim=1)

            # Physics encoding (per step — flow matching)
            physics_embs = None
            if physics_hist_tok is not None and physics_fut is not None:
                physics_fut_tok = self.physics_fut_encoder(physics_fut, t_scalar)
                physics_embs = torch.cat([physics_hist_tok, physics_fut_tok], dim=1)

            # GraphSafeMSAT forward
            model_output = self.gs_msat(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=t_scalar,
                physics_embs=physics_embs,
            )

            # Action decoding + Euler step
            if isinstance(model_output, dict):
                action_output = model_output["action"]
            else:
                action_output = model_output

            pred = self.action_decoder(action_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon :]
            current_state = current_state + self.dt * pred_velocity
            if d > 0:
                current_state[:, :d] = prefix_actions

            # Physics decoding + Euler step (flow matching)
            if (
                physics_fut is not None
                and isinstance(model_output, dict)
                and "physics" in model_output
            ):
                physics_hidden_fut = model_output["physics"][:, -self.physics_fut_len :]
                physics_pred_vel = self.physics_decoder(physics_hidden_fut)
                physics_fut = physics_fut + self.dt * physics_pred_vel

        return current_state

    def __getattr__(self, name):
        """Delegate attribute access to GraphSafeMSAT."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.gs_msat, name)
