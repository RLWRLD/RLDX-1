"""Real-Time Chunking (RTC) for flow-matching action heads.

Implements two mutually-reinforcing variants:

- **Training RTC** (arXiv 2512.05964): Per-sample inference-delay simulation.
  At training time, sample a delay d_i in [0, max_delay] for each batch item;
  set t=1 on the first d_i action-horizon positions (they stay clean ground
  truth), sample normal t for the rest, and mask loss on the clean prefix.
  The model learns to predict the postfix conditioned on a clean prefix.

- **Inference RTC**
    * ``trained`` (2512.05964): Hard-inpaint the frozen prefix at each
      Euler step and feed the model per-token time with t=1 on the prefix.
      Only valid for models trained with training-RTC.
    * ``guided`` (2506.07339 Eq. 2): Jacobian universal-guidance modification
      of the velocity field via the vector-Jacobian product of the predicted
      clean trajectory ``Â¹ = x_τ + (1-τ)·v`` with respect to ``x_τ``. Works
      with any pre-trained flow-matching VLA, no retraining required.

The soft mask ``W`` (Eq. 5) ramps from 1 on the frozen prefix down to 0 on the
free postfix with an exponential taper across the middle band.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional

import torch


_EPS_TAU = 1e-2  # τ clip range for 1/(τ(1-τ)) guidance scale stability
_EPS_DENOM = 1e-8


@dataclass
class RTCConfig:
    """Parameters controlling RTC behaviour.

    Defaults disable RTC entirely (training_max_delay=0, inference_mode="none").
    """

    # Training-time RTC. Per-sample prefix length d ~ U[0, max_delay].
    # ``max_delay = 0`` disables training RTC (no extra prob knob — d=0 is a
    # valid sample that reduces to standard flow-matching training).
    training_max_delay: int = 0

    # Inference-time RTC
    inference_mode: str = "none"  # "none" | "trained" | "guided"
    inference_delay: int = 0  # d (frozen prefix length)
    inference_exec_horizon: int = 0  # s (execution horizon). 0 => H - d

    # Jacobian-specific.
    #
    # ``jacobian_beta`` clips the guidance scale ``min(β, 1/(τ(1-τ)))``. Note
    # that ``1/(τ(1-τ)) ≥ 4`` for τ ∈ (0,1), so β ≤ 4 disables the paper's
    # adaptive τ schedule entirely (scale collapses to β everywhere).
    # Default β=5 lets the schedule activate near the τ→0 and τ→1 boundaries
    # while keeping guidance stable mid-range.
    jacobian_beta: float = 5.0
    # Apply guidance only on the first ``N`` inference steps. Jacobian
    # autograd through the full DiT multiplies inference memory several-fold
    # and can OOM on full-size VLAs. ``None`` = guide every step.
    jacobian_steps_only: Optional[int] = None

    def enabled_training(self) -> bool:
        return self.training_max_delay > 0

    def enabled_inference(self) -> bool:
        return self.inference_mode != "none" and self.inference_delay > 0

    def validate(self, horizon: int) -> None:
        if self.inference_mode not in ("none", "trained", "guided"):
            raise ValueError(f"invalid inference_mode: {self.inference_mode}")
        # Reject max_delay == horizon: it lets a sample have every position in
        # the frozen prefix, producing an empty postfix and a zero-gradient
        # loss contribution. Keep one position of postfix guaranteed.
        if self.training_max_delay < 0 or self.training_max_delay >= horizon:
            raise ValueError(
                f"training_max_delay={self.training_max_delay} must be in [0, {horizon - 1}]"
            )
        # inference_mode='trained' requires the checkpoint to have been
        # trained with rtc_training_max_delay > 0 (paper 2512.05964 Alg. 1).
        # Otherwise the model has never seen clean-prefix / t=1 conditioning
        # and will produce degenerate postfixes. The explicit check fails
        # fast at init instead of silently-wrong inference.
        if self.inference_mode == "trained" and self.training_max_delay == 0:
            raise ValueError(
                "inference_mode='trained' requires a checkpoint trained with "
                "rtc_training_max_delay > 0. This config has training_max_delay=0 "
                "(no RTC training signal). For RTC on an untrained checkpoint, "
                "use inference_mode='guided' (Jacobian universal guidance; "
                "arXiv 2506.07339, no retraining needed)."
            )
        d = self.inference_delay
        s = self.inference_exec_horizon or (horizon - d)
        if self.inference_mode != "none":
            if not (0 <= d <= s <= horizon - d):
                raise ValueError(
                    f"must satisfy 0 <= d ({d}) <= s ({s}) <= H - d ({horizon - d}); H={horizon}"
                )


def sample_training_prefix(
    batch_size: int,
    horizon: int,
    max_delay: int,
    *,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample a boolean prefix mask for training RTC (Algorithm 1 of
    arXiv 2512.05964).

    For each sample draws ``d_i ~ U[0, max_delay]`` and marks positions
    ``[0, d_i)`` as frozen prefix. ``d_i = 0`` naturally recovers the
    standard flow-matching case, so no extra probability gate is needed.
    """
    if max_delay <= 0:
        return torch.zeros(batch_size, horizon, dtype=torch.bool, device=device)

    kwargs = {"device": device}
    if generator is not None:
        kwargs["generator"] = generator

    delays = torch.randint(0, max_delay + 1, (batch_size,), **kwargs)
    positions = torch.arange(horizon, device=device).unsqueeze(0)  # (1, H)
    return positions < delays.unsqueeze(1)  # (B, H)


def build_per_token_time(
    scalar_tau: torch.Tensor,
    prefix_mask: torch.Tensor,
) -> torch.Tensor:
    """Construct per-token time with t=1 on the prefix, scalar τ elsewhere.

    Args:
        scalar_tau: ``(B,)`` per-sample noise time.
        prefix_mask: ``(B, H)`` boolean mask marking clean-prefix positions.
    Returns:
        ``(B, H)`` per-token time tensor.
    """
    ones = torch.ones_like(prefix_mask, dtype=scalar_tau.dtype)
    tau_bh = scalar_tau.unsqueeze(1).expand_as(prefix_mask).to(dtype=scalar_tau.dtype)
    return torch.where(prefix_mask, ones, tau_bh)


def build_noisy_trajectory_rtc(
    actions: torch.Tensor,
    noise: torch.Tensor,
    t_per_token: torch.Tensor,
) -> torch.Tensor:
    """Flow-matching linear-interpolation noise schedule with per-token t.

    ``x_t = t * action + (1 - t) * noise`` so that prefix positions
    (t = 1) stay at the ground-truth clean action.

    Args:
        actions: ``(B, H, D)``.
        noise: ``(B, H, D)``.
        t_per_token: ``(B, H)`` time per action step.
    """
    t_exp = t_per_token.unsqueeze(-1).to(dtype=actions.dtype)
    return t_exp * actions + (1.0 - t_exp) * noise


def compute_soft_mask_weights(
    horizon: int,
    d: torch.Tensor | int,
    s: torch.Tensor | int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Eq. 5 of arXiv 2506.07339 — soft mask weights W_i.

    The three regions (frozen / ramp / free) are produced vectorised so that
    per-sample ``d`` and ``s`` are supported. Returns ``(B, H)``.
    """
    if isinstance(d, int):
        d = torch.tensor([d], device=device)
    if isinstance(s, int):
        s = torch.tensor([s], device=device)
    if d.dim() == 0:
        d = d.unsqueeze(0)
    if s.dim() == 0:
        s = s.unsqueeze(0)
    B = int(max(d.shape[0], s.shape[0]))
    if d.shape[0] == 1 and B > 1:
        d = d.expand(B)
    if s.shape[0] == 1 and B > 1:
        s = s.expand(B)

    d = d.to(device=device, dtype=torch.long)
    s = s.to(device=device, dtype=torch.long)
    positions = torch.arange(horizon, device=device).unsqueeze(0).expand(B, -1)  # (B, H)

    # Regions
    frozen = positions < d.unsqueeze(1)  # i < d
    free = positions >= (horizon - s).unsqueeze(1)  # i >= H - s
    ramp = (~frozen) & (~free)  # d <= i < H - s

    # c_i = (H - s - i) / (H - s - d + 1)
    numer = (horizon - s).unsqueeze(1) - positions  # (B, H)
    denom = (horizon - s - d + 1).unsqueeze(1).clamp_min(1)  # avoid /0 when s+d=H+1
    c = numer.to(dtype=dtype) / denom.to(dtype=dtype)
    # c_i should be in [0, 1] within ramp band; clamp for numerical safety
    c = c.clamp(min=0.0, max=1.0)

    ramp_val = c * (torch.exp(c) - 1.0) / (math.e - 1.0)

    W = torch.zeros(B, horizon, device=device, dtype=dtype)
    W = torch.where(frozen, torch.ones_like(W), W)
    W = torch.where(ramp, ramp_val, W)
    return W


def guidance_scale(tau: float, beta: float) -> float:
    """min(β, (1-τ)/(τ·r_τ²)) per Eq. 2 + Eq. 4 of arXiv 2506.07339.

    Algebraic equivalent: min(β, (τ²+(1-τ)²)/(τ(1-τ))).
    """
    t = min(max(tau, _EPS_TAU), 1.0 - _EPS_TAU)
    inv_r2 = (t * t + (1.0 - t) * (1.0 - t)) / ((1.0 - t) * (1.0 - t))
    c = (1.0 - t) / t
    return float(min(c * inv_r2, beta))


def guided_velocity(
    velocity_fn: Callable[[torch.Tensor], torch.Tensor],
    x_tau: torch.Tensor,
    tau: float,
    Y: torch.Tensor,
    W: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Jacobian-guided velocity update (Eq. 2 of arXiv 2506.07339).

    ``v_guided = v + c · VJP[(Y − Â¹_t)ᵀ diag(W), ∂Â¹_t/∂x_τ]``

    where ``Â¹_t = x_τ + (1−τ) v`` and
    ``c = min(β, 1 / (τ(1−τ)))``.

    Args:
        velocity_fn: maps ``x_τ → v`` (must be differentiable wrt the input).
        x_tau: ``(B, H, D)`` current noisy trajectory.
        tau: scalar flow-matching time in (0, 1).
        Y: ``(B, H, D)`` inpainting target (frozen prefix for the prefix
            region; value in the ramp/free region is irrelevant because W=0).
        W: ``(B, H)`` or ``(H,)`` soft mask weights (Eq. 5).
        beta: guidance scale clip constant.
    Returns:
        ``(B, H, D)`` guided velocity, detached from the autograd graph.
    """
    if W.dim() == 1:
        W = W.unsqueeze(0).expand(x_tau.shape[0], -1)

    x_in = x_tau.detach().requires_grad_(True)
    with torch.enable_grad():
        v = velocity_fn(x_in)
        a_hat = x_in + (1.0 - tau) * v
        diff = Y.detach() - a_hat.detach()  # (B, H, D), no grad through diff
        grad_outputs = W.unsqueeze(-1).to(dtype=a_hat.dtype) * diff  # (B, H, D)
        vjp = torch.autograd.grad(
            outputs=a_hat,
            inputs=x_in,
            grad_outputs=grad_outputs,
            retain_graph=False,
            create_graph=False,
        )[0]

    c = guidance_scale(float(tau), beta)
    return (v.detach() + c * vjp.detach()).to(dtype=x_tau.dtype)


def rtc_config_from_rldx(cfg) -> RTCConfig:
    """Build an RTCConfig from the flat fields on an RLDXConfig instance."""
    return RTCConfig(
        training_max_delay=int(getattr(cfg, "rtc_training_max_delay", 0) or 0),
        inference_mode=str(getattr(cfg, "rtc_inference_mode", "none")),
        inference_delay=int(getattr(cfg, "rtc_inference_delay", 0) or 0),
        inference_exec_horizon=int(getattr(cfg, "rtc_inference_exec_horizon", 0) or 0),
        jacobian_beta=float(getattr(cfg, "rtc_jacobian_beta", 5.0)),
        jacobian_steps_only=(
            None
            if getattr(cfg, "rtc_jacobian_steps_only", None) is None
            else int(cfg.rtc_jacobian_steps_only)
        ),
    )


__all__ = [
    "RTCConfig",
    "rtc_config_from_rldx",
    "sample_training_prefix",
    "build_per_token_time",
    "build_noisy_trajectory_rtc",
    "compute_soft_mask_weights",
    "guided_velocity",
    "guidance_scale",
]
