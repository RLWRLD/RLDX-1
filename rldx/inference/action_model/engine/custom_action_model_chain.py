"""CustomActionHeadChain: 2-way action head pipeline (v1.4-v, no physics).

Pre-bakes at init:
  - pos_embs:    position_embedding(static_pos_ids) — same every forward
  - tembs:       timestep_encoder(t) for each denoising step — same every forward
  - time_tokens: time_token_proj(temb) for each step — same every forward

Forward only runs data-dependent ops: action_enc, MSAT (DS+SS), action_dec, Euler.

For v1.4-all (physics), see custom_expanded_action_model_chain.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rldx.utils.dist import rank_zero_print as _print

from .custom_msat_chain import CustomOpMSAT


class CustomActionHeadChain(nn.Module):
    """Unified 2-way action head chain with baked static values (v1.4-v)."""

    def __init__(self, gs_action_model, device, dtype=torch.bfloat16):
        super().__init__()

        self.action_horizon = gs_action_model.action_horizon
        self.action_dim = gs_action_model.action_dim
        self.num_inference_timesteps = gs_action_model.num_inference_timesteps
        self.dt = gs_action_model.dt
        # RTC trained-mode prefix length, baked into the chain at build
        # time. Slice indices ``[:, :prefix_len]`` resolve to compile-time
        # constants under ``torch.compile``.
        self.prefix_len = getattr(gs_action_model, "prefix_len", 0)

        self.register_buffer("static_timesteps", gs_action_model.static_timesteps)

        self.vlln = gs_action_model.vlln
        self.state_encoder = gs_action_model.state_encoder
        self.action_encoder = gs_action_model.action_encoder
        self.action_decoder = gs_action_model.action_decoder

        # --- Bake static values at init ---
        msat_raw = gs_action_model.gs_msat._msat
        num_temb = msat_raw.num_temb_tokens

        with torch.no_grad():
            # pos_embs: position_embedding(static_pos_ids) — always same
            pos_embs = gs_action_model.position_embedding(gs_action_model.static_pos_ids).unsqueeze(
                0
            )
            self.register_buffer("static_pos_embs", pos_embs)

            # tembs + time_tokens: per denoising step, always same
            tembs = []
            time_tokens = []
            for t_val in gs_action_model.static_timesteps:
                temb = msat_raw.timestep_encoder(t_val.unsqueeze(0))
                tt = msat_raw.time_token_proj(temb).unsqueeze(1)
                tt = tt.repeat(1, num_temb, 1)
                tembs.append(temb)
                time_tokens.append(tt)
            self.register_buffer("static_tembs", torch.stack(tembs))  # (N_steps, temb_dim)
            self.register_buffer(
                "static_time_tokens", torch.stack(time_tokens)
            )  # (N_steps, num_temb, sa_dim)

        # MSAT (no timestep_encoder/time_token_proj — baked above)
        n_vl = gs_action_model.gs_msat.n_vl
        n_sa_pure = gs_action_model.gs_msat.n_sa_pure
        self.msat = CustomOpMSAT(
            gs_action_model.gs_msat,
            n_sa_pure=n_sa_pure,
            n_vl=n_vl,
            device=device,
            dtype=dtype,
        ).eval()

    def forward(
        self, vl_embs, state, embodiment_id, init_noise=None, prefix_actions=None, **kwargs
    ):
        vl_embs = self.vlln(vl_embs)
        B = vl_embs.shape[0]
        d = self.prefix_len

        state_features = self.state_encoder(state, embodiment_id)

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
            # In RTC trained mode the prefix is frozen across all
            # denoising steps — its inputs (``prefix_actions``,
            # ``t_tok = 1.0``, ``embodiment_id``) are step-invariant by
            # contract.  ``MultiEmbodimentActionEncoder`` is a per-token
            # MLP (verified slice-invariant with ``max_diff = 0`` for
            # both fp32 and bf16), so its output for the prefix region
            # is also step-invariant — compute it once and concat with
            # the freshly-encoded postfix every step.
            prefix_t_tok = torch.ones(
                B,
                d,
                dtype=current_state.dtype,
                device=current_state.device,
            )
            prefix_features = self.action_encoder(
                prefix_actions,
                prefix_t_tok,
                embodiment_id,
            )
            prefix_features = prefix_features + self.static_pos_embs[:, :d]

        for t in range(self.num_inference_timesteps):
            t_scalar = self.static_timesteps[t].expand(B)
            temb = self.static_tembs[t].expand(B, -1)  # baked
            time_token = self.static_time_tokens[t].expand(B, -1, -1)  # baked

            # Always pass per-token time of shape (B, action_horizon) so
            # any future fused variant of ``action_encoder`` is forced to
            # honour position-wise time and the trained-mode prefix slice
            # write cannot be silently broadcast away.
            t_tok = t_scalar.unsqueeze(1).expand(-1, self.action_horizon).clone()
            if d > 0:
                # Prefix encoder output is cached; only encode postfix.
                t_tok = t_tok[:, d:]
                postfix_features = self.action_encoder(current_state[:, d:], t_tok, embodiment_id)
                postfix_features = postfix_features + self.static_pos_embs[:, d:]
                action_features = torch.cat([prefix_features, postfix_features], dim=1)
            else:
                action_features = self.action_encoder(current_state, t_tok, embodiment_id)
                action_features = action_features + self.static_pos_embs  # baked

            sa_embs = torch.cat([state_features, action_features], dim=1)

            action_out = self.msat(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                temb=temb,
                time_token=time_token,
            )

            if d > 0:
                # The prefix slot's predicted velocity is discarded by
                # the re-lock contract, so don't decode it.  The decoder
                # is a per-token MLP (also verified slice-invariant), so
                # postfix-only decode is bit-equivalent to decode-then-
                # discard.
                postfix_pred = self.action_decoder(
                    action_out[:, -(self.action_horizon - d) :, :], embodiment_id
                )
                current_state = torch.cat(
                    [
                        prefix_actions,
                        current_state[:, d:] + self.dt * postfix_pred,
                    ],
                    dim=1,
                )
            else:
                pred = self.action_decoder(action_out, embodiment_id)
                pred_velocity = pred[:, -self.action_horizon :]
                current_state = current_state + self.dt * pred_velocity

        return current_state


def build_custom_action_model_chain(action_head_model, device, dtype=torch.bfloat16):
    """Build a CustomActionHeadChain from a GraphSafeActionModel (no compilation)."""
    return CustomActionHeadChain(action_head_model, device=device, dtype=dtype).eval()


def compile_custom_action_model_chain(
    chain, sample_inputs, compile_mode="max-autotune", fullgraph=True
):
    """Compile a CustomActionHeadChain with torch.compile and trigger compilation."""
    import time as _time

    _print(f"  [ActionHeadChain] Compiling ({compile_mode}, fullgraph={fullgraph})...")
    compiled_chain = torch.compile(chain, mode=compile_mode, fullgraph=fullgraph)

    vl_embs, state, embodiment_id, init_noise = sample_inputs
    t0 = _time.time()
    with torch.no_grad():
        compiled_chain(vl_embs, state, embodiment_id, init_noise=init_noise)
    torch.cuda.synchronize()
    compile_time_s = _time.time() - t0
    _print(f"  [ActionHeadChain] Compilation: {compile_time_s:.1f}s")

    return compiled_chain, compile_time_s
