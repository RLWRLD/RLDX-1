from typing import Tuple

from rldx.configs.model.rldx import RLDXConfig
from rldx.model.modules.action_model.msat import MSAT
from rldx.model.modules.action_model.physics import init_physics_params_near_zero
from rldx.model.modules.action_model.physics_head import NoOpPhysicsHead, PhysicsHead
from rldx.model.modules.action_model.rtc import (
    RTCConfig,
    build_noisy_trajectory_rtc,
    build_per_token_time,
    compute_soft_mask_weights,
    guidance_scale,
    rtc_config_from_rldx,
    sample_training_prefix,
)
from rldx.model.modules.backbone.adapter import VTCQwen3VLBackbone
from rldx.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
from rldx.model.modules.memory import TransformerMemory
from rldx.utils.dist import rank_zero_print as _print
import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree


class RLDXActionModel(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: RLDXConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        # Initialize MSAT from config
        config.diffusion_model_cfg.setdefault("attention_head_dim", 64)
        config.diffusion_model_cfg.setdefault("depth_multi_stream", 4)
        config.diffusion_model_cfg.setdefault("depth_single_stream", 8)
        config.diffusion_model_cfg.setdefault("dropout", 0.2)
        config.diffusion_model_cfg.setdefault("final_dropout", True)
        config.diffusion_model_cfg.setdefault("num_attention_heads", 24)
        config.diffusion_model_cfg.setdefault("output_dim", 1024)
        config.diffusion_model_cfg.setdefault("positional_embeddings", "rope_sa_only")
        config.diffusion_model_cfg.setdefault("rope_theta", 10000.0)
        config.diffusion_model_cfg.setdefault("temb_type", "input_token")
        config.diffusion_model_cfg.setdefault("use_swiglu", True)
        config.diffusion_model_cfg.setdefault("action_model_max_seq_len", 1024)
        config.diffusion_model_cfg.setdefault("pre_norm", "layer_norm")
        config.diffusion_model_cfg.setdefault("qk_norm", "rms_norm")
        config.diffusion_model_cfg.setdefault("sa_dim", config.input_embedding_dim)
        config.diffusion_model_cfg.setdefault("vl_dim", config.backbone_embedding_dim)
        # Remove deprecated triple-stream keys (may exist in old checkpoints)
        for _key in (
            "set_triple_stream_for_mq",
            "set_triple_stream_for_state",
            "state_dim",
            "action_dim",
            "mq_dim",
            "state_mlp_ratio",
            "action_mlp_ratio",
            "mq_mlp_ratio",
        ):
            config.diffusion_model_cfg.pop(_key, None)
        # Inject physics config
        config.diffusion_model_cfg["use_physics"] = getattr(config, "use_physics", False)
        config.diffusion_model_cfg["physics_dim"] = getattr(config, "physics_dim", 0)
        self.model = MSAT(
            **config.diffusion_model_cfg,
        )
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob
        self.mask_token = (
            nn.Parameter(0.02 * torch.randn(1, 1, self.input_embedding_dim))
            if self.state_dropout_prob > 0
            else None
        )

        # State noise parameters
        self.state_additive_noise_scale = config.state_additive_noise_scale

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        # self.beta_dist = Beta(
        #     torch.tensor(config.noise_beta_alpha, dtype=torch.float32),
        #     torch.tensor(config.noise_beta_beta, dtype=torch.float32),
        # )
        self.num_timestep_buckets = config.num_timestep_buckets

        # Real-Time Chunking.
        self._rtc: RTCConfig = rtc_config_from_rldx(config)
        self._rtc.validate(self.action_horizon)
        if self._rtc.enabled_training() or self._rtc.enabled_inference():
            _print(
                f"[RTC] enabled: training_max_delay={self._rtc.training_max_delay}, "
                f"inference_mode={self._rtc.inference_mode}, "
                f"inference_delay={self._rtc.inference_delay}"
            )

        # Physics (tactile/torque) stream
        self.use_physics = getattr(config, "use_physics", False)
        physics_dim = getattr(config, "physics_dim", 0)

        if self.use_physics and physics_dim > 0:
            embed_dim = self.input_embedding_dim
            msat_output_dim = config.diffusion_model_cfg.get("output_dim", 1024)
            config_flow_matching = getattr(config, "physics_use_flow_matching", True)
            physics_delta = getattr(config, "physics_delta_indices", None) or []
            effective_flow_matching = (
                config_flow_matching and sum(1 for d in physics_delta if d > 0) > 0
            )

            self.physics = PhysicsHead(
                physics_dim=physics_dim,
                embed_dim=embed_dim,
                msat_output_dim=msat_output_dim,
                physics_delta_indices=physics_delta,
                physics_use_flow_matching=effective_flow_matching,
                physics_loss_weight=getattr(config, "physics_loss_weight", 0.1),
                action_horizon=self.action_horizon,
                physics_dropout_prob=getattr(config, "physics_dropout_prob", 0.0),
            )
            _print("[Physics] Applying near-zero (exit-zero) initialization...")
            init_physics_params_near_zero(self)
        else:
            self.physics = NoOpPhysicsHead()

        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        # Issue #54: when LoRA is on, the diffusion model is no longer
        # full-tuned — LoRA adapters are the only trainable surface inside
        # ``self.model``. We override the flag here (instead of asking the
        # caller to flip it) so legacy callers that pass
        # ``tune_diffusion_model=True`` from the config default still get
        # LoRA-only behaviour, with a single source of truth.
        use_lora = getattr(self.config, "action_model_use_lora", False)
        if use_lora:
            tune_diffusion_model = False

        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            self.physics.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
            if self.state_dropout_prob > 0:
                self.mask_token.requires_grad_(False)

        if use_lora:
            # Replaces the unconditional ``self.model.requires_grad_(False)``:
            # _apply_action_model_lora freezes the DiT first and then PEFT
            # marks only the injected LoRA params trainable.
            self._apply_action_model_lora()
        elif not tune_diffusion_model:
            self.model.requires_grad_(False)

        if not tune_vlln:
            self.vlln.requires_grad_(False)

        _print(f"[MSAT] Tune action model projector: {self.tune_projector}")
        _print(f"[MSAT] Tune action model diffusion model: {self.tune_diffusion_model}")
        _print(f"[MSAT] Tune action model vlln: {self.tune_vlln}")
        _print(f"[MSAT] Action model LoRA: {use_lora}")

        # Check if any parameters are still trainable. If not, _print a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln and not use_lora:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    _print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            _print("Warning: No action model trainable parameters found.")

    def _apply_action_model_lora(self):
        """Inject PEFT LoRA adapters into the MSAT diffusion model — Issue #54.

        Freezes the entire MSAT first, then wraps the target Linear layers
        listed in ``config.action_model_lora_target_modules`` (PEFT marks the
        injected LoRA weights ``requires_grad=True``). Target names that don't
        exist in the current MSAT (e.g. ``p_qkv``/``p_proj`` when
        ``use_physics=False``) are filtered before the PEFT call so PEFT
        doesn't raise on a missing target.
        """
        try:
            from peft import LoraConfig, inject_adapter_in_model
        except ImportError as e:
            raise ImportError(
                "peft is required for action_model_use_lora=True. Install with `pip install peft`."
            ) from e

        target_modules = list(
            getattr(
                self.config,
                "action_model_lora_target_modules",
                ["vl_qkv", "vl_proj", "sa_qkv", "sa_proj", "p_qkv", "p_proj", "linear1", "linear2"],
            )
        )

        # Keep only target names that actually appear in the MSAT. PEFT
        # matches by exact name OR ".{name}" suffix of the fully qualified
        # module path — mirror that here so we don't pass dead targets
        # (e.g. p_qkv when physics is disabled).
        module_names = [name for name, _ in self.model.named_modules()]

        def _present(target: str) -> bool:
            dot_target = f".{target}"
            return any(n == target or n.endswith(dot_target) for n in module_names)

        filtered = [t for t in target_modules if _present(t)]
        skipped = [t for t in target_modules if t not in filtered]
        if skipped:
            _print(f"[ActionModel LoRA] Skipping absent target modules: {skipped}")
        if not filtered:
            raise ValueError(
                f"[ActionModel LoRA] None of the requested target modules "
                f"{target_modules} exist in the MSAT."
            )

        # Freeze the entire MSAT; LoRA weights will be marked trainable by PEFT.
        self.model.requires_grad_(False)

        lora_config = LoraConfig(
            r=int(getattr(self.config, "action_model_lora_rank", 16)),
            lora_alpha=int(getattr(self.config, "action_model_lora_alpha", 32)),
            lora_dropout=float(getattr(self.config, "action_model_lora_dropout", 0.0)),
            bias="none",
            target_modules=filtered,
        )
        inject_adapter_in_model(lora_config, self.model)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        ratio = (100.0 * trainable / total) if total > 0 else 0.0
        _print(
            f"[ActionModel LoRA] target_modules={filtered}, "
            f"r={lora_config.r}, alpha={lora_config.lora_alpha}, "
            f"dropout={lora_config.lora_dropout}"
        )
        _print(f"[ActionModel LoRA] trainable params: {trainable} / {total} ({ratio:.2f}%)")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                self.physics.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action model.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Dropout state features.
        if self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout) + self.mask_token * do_dropout

        # Add Gaussian noise to state features.
        if self.training and self.state_additive_noise_scale > 0:
            _print(
                f"Adding Gaussian noise to state features with scale {self.state_additive_noise_scale}"
            )
            noise = torch.randn_like(state_features) * self.state_additive_noise_scale
            state_features = state_features + noise

        # Embed noised action trajectory.
        actions = action_input.action
        batch_size = actions.shape[0]
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t_raw = self.sample_time(batch_size, device=actions.device, dtype=actions.dtype)  # (B,)

        # Training RTC: per-sample prefix with clean ground-truth actions at t=1.
        if self.training and self._rtc.enabled_training():
            prefix_mask = sample_training_prefix(
                batch_size,
                self.action_horizon,
                self._rtc.training_max_delay,
                device=actions.device,
            )  # (B, H) bool
            t_tok = build_per_token_time(t_raw, prefix_mask)  # (B, H)
            noisy_trajectory = build_noisy_trajectory_rtc(actions, noise, t_tok)
        else:
            prefix_mask = None
            t_tok = t_raw.unsqueeze(1).expand(-1, self.action_horizon).contiguous()
            t = t_raw[:, None, None]
            noisy_trajectory = (1 - t) * noise + t * actions

        velocity = actions - noise
        action_features = self.action_encoder(noisy_trajectory, t_tok, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)

        encoder_attention_mask = backbone_output.get("backbone_attention_mask", None)
        # Only pass mask to MSAT when there are actually masked positions
        if encoder_attention_mask is not None and encoder_attention_mask.all():
            encoder_attention_mask = None

        # Encode physics signal
        physics_embs, physics_attn_mask, physics_velocity = self.physics.prepare_train(
            action_input, t_raw
        )

        # MSAT global temb uses the scalar postfix τ per sample. Per-token
        # time has already been threaded through action_encoder above.
        model_output, _ = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embeds,
            timestep=t_raw,
            return_all_hidden_states=True,
            encoder_attention_mask=encoder_attention_mask,
            physics_embs=physics_embs,
            physics_attention_mask=physics_attn_mask,
        )

        # When physics is enabled, model_output is a dict {"action": ..., "physics": ...}
        if isinstance(model_output, dict):
            action_model_output = model_output["action"]
            physics_model_output = model_output["physics"]
        else:
            action_model_output = model_output
            physics_model_output = None

        pred = self.action_decoder(action_model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Action loss. Training RTC masks out the clean-prefix positions so
        # the model is only graded on postfix reconstruction.
        action_mask = action_input.action_mask
        loss_mask = action_mask
        if prefix_mask is not None:
            postfix = (~prefix_mask).to(dtype=action_mask.dtype).unsqueeze(-1)
            loss_mask = action_mask * postfix
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * loss_mask
        loss = action_loss.sum() / (loss_mask.sum() + 1e-6)

        results = {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

        # Physics prediction loss (flow matching only; conditioning-only mode has no physics loss)
        physics_loss = self.physics.compute_loss(
            physics_model_output, physics_velocity, action_mask, physics_attn_mask
        )
        if physics_loss is not None:
            loss = loss + self.physics.physics_loss_weight * physics_loss
            results["loss"] = loss
            results["physics_loss"] = physics_loss

        return results

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action model.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, state_horizon, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
            action_input: Optional, used for physics conditioning and for RTC
                prefix inputs. When Real-Time Chunking is enabled by config,
                ``action_input`` may carry:
                - ``action_prefix``: [B, d, action_dim] frozen actions from the
                  previous chunk.
                - ``rtc_prefix_len``: int (defaults to
                  ``config.rtc_inference_delay``).
                Missing prefix at episode start falls back to standard sampling.
        """
        vl_embeds = backbone_features

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        dtype = vl_embeds.dtype
        horizon = self.config.action_horizon

        # ─── RTC setup ──────────────────────────────────────────────────────
        rtc = self._rtc
        rtc_mode = "none"
        prefix_actions = None
        prefix_len = 0
        soft_mask_W = None
        postfix_target = None
        if rtc.enabled_inference() and action_input is not None:
            maybe_prefix = action_input.get("action_prefix", None)
            cfg_len = int(action_input.get("rtc_prefix_len", rtc.inference_delay) or 0)
            if maybe_prefix is not None and cfg_len > 0:
                if not isinstance(maybe_prefix, torch.Tensor):
                    maybe_prefix = torch.as_tensor(maybe_prefix, dtype=dtype, device=device)
                else:
                    maybe_prefix = maybe_prefix.to(device=device, dtype=dtype)
                expected_dim = self.action_dim
                if maybe_prefix.dim() != 3 or maybe_prefix.shape[-1] != expected_dim:
                    raise ValueError(
                        f"action_prefix must have shape (B, >=d, {expected_dim}); "
                        f"got {tuple(maybe_prefix.shape)}"
                    )
                if not torch.isfinite(maybe_prefix).all():
                    raise ValueError("action_prefix contains NaN or Inf values")
                if maybe_prefix.shape[1] >= cfg_len:
                    prefix_actions = maybe_prefix[:, :cfg_len].contiguous()
                    prefix_len = cfg_len
                    rtc_mode = rtc.inference_mode
                    if rtc_mode == "guided":
                        s = rtc.inference_exec_horizon or (horizon - prefix_len)
                        soft_mask_W = compute_soft_mask_weights(
                            horizon, prefix_len, s, device=device, dtype=dtype
                        )  # (1, H); will broadcast to (B, H)
                        # Optional Y target for the new chunk's [d, H) region —
                        # the previous chunk's predictions sliced to the matching
                        # absolute-time positions (Eq. 5 of arXiv 2506.07339).
                        # Server stores this in SessionRegistry.rtc_chunk and
                        # PolicyRuntime injects it as ``action_postfix_target``.
                        # Cold start path leaves it None and the Jacobian VJP
                        # falls back to the prefix-only signal.
                        maybe_postfix = action_input.get("action_postfix_target", None)
                        if maybe_postfix is not None:
                            if not isinstance(maybe_postfix, torch.Tensor):
                                maybe_postfix = torch.as_tensor(
                                    maybe_postfix, dtype=dtype, device=device
                                )
                            else:
                                maybe_postfix = maybe_postfix.to(device=device, dtype=dtype)
                            if (
                                maybe_postfix.dim() != 3
                                or maybe_postfix.shape[0] != batch_size
                                or maybe_postfix.shape[-1] != expected_dim
                            ):
                                raise ValueError(
                                    f"action_postfix_target must have shape "
                                    f"(B={batch_size}, *, {expected_dim}); got "
                                    f"{tuple(maybe_postfix.shape)}"
                                )
                            if not torch.isfinite(maybe_postfix).all():
                                raise ValueError("action_postfix_target contains NaN or Inf values")
                            postfix_target = maybe_postfix.contiguous()

        # Per-call RTC trace so deployers can see what happened on each
        # inference. One line per invocation.
        _rtc_parts = [f"mode={rtc_mode}", f"prefix_len={prefix_len}"]
        if rtc_mode != "none":
            _rtc_s = rtc.inference_exec_horizon or (horizon - prefix_len)
            _rtc_parts.append(f"s={_rtc_s}")
            if rtc_mode == "guided":
                _rtc_parts.append(f"β={rtc.jacobian_beta}")
                if rtc.jacobian_steps_only is not None:
                    _rtc_parts.append(f"steps_only={rtc.jacobian_steps_only}")
        elif rtc.enabled_inference():
            _rtc_parts.append("cold_start=True")  # config wants RTC but no prefix arrived
        _print(f"[RTC] B={batch_size} H={horizon} " + " ".join(_rtc_parts))

        # ─── Physics ────────────────────────────────────────────────────────
        with torch.no_grad():
            phys_state = self.physics.prepare_inference(action_input, batch_size, device, dtype)

        # ─── Initial noise ──────────────────────────────────────────────────
        actions = torch.randn(
            size=(batch_size, horizon, self.action_dim),
            dtype=dtype,
            device=device,
        )
        # Trained mode hard-inpaints prefix; guided mode relies on VJP only.
        if rtc_mode == "trained" and prefix_len > 0:
            actions[:, :prefix_len] = prefix_actions

        # Use custom denoising timesteps if set, otherwise uniform spacing.
        if hasattr(self, "denoising_timesteps") and self.denoising_timesteps is not None:
            timesteps_list = list(self.denoising_timesteps) + [1.0]
        else:
            n = self.num_inference_timesteps
            timesteps_list = [t / float(n) for t in range(n)] + [1.0]

        encoder_attention_mask = backbone_output.get("backbone_attention_mask", None)
        if encoder_attention_mask is not None and encoder_attention_mask.all():
            encoder_attention_mask = None

        def _dit_forward(x_tau: torch.Tensor, t_scalar: torch.Tensor, t_tok: torch.Tensor):
            """One MSAT forward at the current Euler step."""
            af = self.action_encoder(x_tau, t_tok, embodiment_id)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(af.shape[1], dtype=torch.long, device=device)
                af = af + self.position_embedding(pos_ids).unsqueeze(0)
            sa = torch.cat((state_features, af), dim=1)
            phy_embs = self.physics.build_tokens(phys_state, t_scalar)
            mo = self.model(
                hidden_states=sa,
                encoder_hidden_states=vl_embeds,
                timestep=t_scalar,
                encoder_attention_mask=encoder_attention_mask,
                physics_embs=phy_embs,
                physics_attention_mask=phys_state.attn_mask,
            )
            return mo

        # Run denoising steps.
        for i in range(len(timesteps_list) - 1):
            t_cont = float(timesteps_list[i])
            dt = float(timesteps_list[i + 1] - timesteps_list[i])

            t_scalar = torch.full((batch_size,), t_cont, device=device, dtype=dtype)
            t_tok = t_scalar.unsqueeze(1).expand(-1, horizon).clone()
            # Per-token time t=1 on the prefix is a *training-RTC* trick
            # (arXiv 2512.05964): the model has to be trained to expect a
            # clean prefix marked by t=1. Applying it under ``guided`` to a
            # checkpoint without RTC-training pushes the prefix tokens out
            # of the model's training distribution and produces garbage
            # velocity at the chunk boundary.
            # Trained mode: t=1 + hard inpaint on prefix (paper 2512.05964).
            # Guided mode skips both — VJP handles the prefix attraction.
            if rtc_mode == "trained" and prefix_len > 0:
                t_tok[:, :prefix_len] = 1.0
                actions[:, :prefix_len] = prefix_actions

            use_jacobian = rtc_mode == "guided" and (
                rtc.jacobian_steps_only is None or i < rtc.jacobian_steps_only
            )

            if use_jacobian:
                # Compute v_guided = v + c · VJP[(Y−Â¹)ᵀ diag(W), ∂Â¹/∂x].
                x_g = actions.detach().requires_grad_(True)
                with torch.enable_grad():
                    mo = _dit_forward(x_g, t_scalar, t_tok)
                    ao = mo["action"] if isinstance(mo, dict) else mo
                    v = self.action_decoder(ao, embodiment_id)[:, -horizon:]
                    a_hat = x_g + (1.0 - t_cont) * v
                    # Inpainting target Y (Eq. 5 of arXiv 2506.07339).
                    #
                    # When the caller supplied ``action_postfix_target``
                    # (server cache of the previous chunk's predictions),
                    # the new chunk's ramp positions get a non-zero Y →
                    # the soft-mask W's exponential decay produces a
                    # smooth corrective gradient profile across the ramp.
                    #
                    # When that target is absent (cold start: first chunk
                    # of an episode), Y collapses to the current ``a_hat``
                    # outside the frozen prefix so the residual is zero
                    # there. In that mode the soft-mask is dead weight in
                    # the ramp band and the only postfix-side signal is
                    # the leak from the prefix VJP through MSAT's
                    # self-attention — sufficient to lock the prefix but
                    # with no explicit ramp smoothing.
                    if postfix_target is not None:
                        Y = a_hat.detach().clone()
                        Y[:, :prefix_len] = prefix_actions
                        ramp_len = min(postfix_target.shape[1], horizon - prefix_len)
                        if ramp_len > 0:
                            Y[:, prefix_len : prefix_len + ramp_len] = postfix_target[:, :ramp_len]
                    else:
                        Y = a_hat.detach().clone()
                        Y[:, :prefix_len] = prefix_actions
                    W_b = soft_mask_W.to(dtype=a_hat.dtype)
                    if W_b.shape[0] == 1 and batch_size > 1:
                        W_b = W_b.expand(batch_size, -1)
                    grad_outputs = W_b.unsqueeze(-1) * (Y - a_hat.detach())
                    vjp = torch.autograd.grad(
                        outputs=a_hat,
                        inputs=x_g,
                        grad_outputs=grad_outputs,
                        retain_graph=False,
                        create_graph=False,
                    )[0]
                c = guidance_scale(t_cont, rtc.jacobian_beta)
                pred_velocity = (v.detach() + c * vjp.detach()).to(dtype=dtype)
                if isinstance(mo, dict):
                    model_output = {k: val.detach() for k, val in mo.items()}
                else:
                    model_output = mo.detach()
            else:
                with torch.no_grad():
                    mo = _dit_forward(actions, t_scalar, t_tok)
                    ao = mo["action"] if isinstance(mo, dict) else mo
                    pred_velocity = self.action_decoder(ao, embodiment_id)[:, -horizon:]
                    model_output = mo

            # Euler step.
            with torch.no_grad():
                actions = actions + dt * pred_velocity
                # Re-lock prefix between Euler steps (trained mode only).
                if rtc_mode == "trained" and prefix_len > 0:
                    actions[:, :prefix_len] = prefix_actions
                phys_state = self.physics.update_state(phys_state, model_output, dt)

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action model."""
        return BatchFeature(data=batch)


class RLDX(PreTrainedModel):
    """RLDX: Vision-Language-Action model with backbone."""

    config_class = RLDXConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: RLDXConfig,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize RLDX model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config
        kwargs = {}
        kwargs["use_cog_tokens"] = True
        kwargs["cog_mode"] = "cog_only"
        kwargs["n_cog_tokens"] = getattr(self.config, "n_cog_tokens", 64)
        _print(f"\n[MSAT Configs] n_cog_tokens: {kwargs['n_cog_tokens']}")

        # Build motion module config if enabled
        if getattr(self.config, "use_motion", False):
            kwargs["motion_config"] = {
                "use_motion": True,
                "motion_insert_layer": getattr(self.config, "motion_insert_layer", 9),
                "motion_d_hid": getattr(self.config, "motion_d_hid", 512),
                "motion_window": tuple(getattr(self.config, "motion_window", (5, 9, 9))),
                "motion_ext_chnls": tuple(getattr(self.config, "motion_ext_chnls", (256,))),
                "motion_int_chnls": tuple(
                    getattr(self.config, "motion_int_chnls", (256, 256, 512))
                ),
                "motion_corr_func": getattr(self.config, "motion_corr_func", "cosine"),
                "motion_n_encoders": getattr(self.config, "motion_n_encoders", 1),
                "motion_use_layerscale": getattr(self.config, "motion_use_layerscale", False),
                "motion_layerscale_init": getattr(self.config, "motion_layerscale_init", 1e-5),
                "motion_use_layernorm": getattr(self.config, "motion_use_layernorm", False),
                "motion_use_syncbn": getattr(self.config, "motion_use_syncbn", False),
                "motion_injection_point": getattr(
                    self.config, "motion_injection_point", "vision_encoder"
                ),
                "motion_pool_type": getattr(self.config, "motion_pool_type", "avg"),
                "motion_drop": getattr(self.config, "motion_drop", True),
                "motion_gradient_check": getattr(self.config, "motion_gradient_check", False),
            }
            _print(f"[motion module] Enabled with config: {kwargs['motion_config']}")

        if config.backbone_model_type != "vtc_qwen3_vl":
            raise ValueError(
                f"Unsupported backbone_model_type={config.backbone_model_type!r}; "
                "only 'vtc_qwen3_vl' is supported since Issue #36."
            )
        self.backbone = VTCQwen3VLBackbone(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
            skip_pretrained_weights=transformers_loading_kwargs.pop(
                "skip_pretrained_weights", False
            ),
            **kwargs,
        )

        # Freeze cog_emb if configured (after backbone init which sets requires_grad=True)
        if getattr(config, "freeze_cog_tokens", False):
            if hasattr(self.backbone, "cog_emb"):
                self.backbone.cog_emb.requires_grad_(False)
                _print("[i] freeze_cog_tokens: cog_emb frozen (requires_grad=False)")

        # Initialize action model
        self.action_model = RLDXActionModel(config)

        # Issue #55 — backbone (Qwen3 LLM) LoRA. Runs AFTER the action model
        # is built so its requires_grad bookkeeping (top-N freeze) is already
        # done; the LoRA injection then freezes the entire backbone, lets
        # PEFT mark only the adapter params trainable, and casts those to
        # fp32 for NaN safety on the first optimizer step.
        if getattr(config, "backbone_use_lora", False):
            self._apply_backbone_lora()

        from .processing_rldx import RLDXDataCollator

        self.collator = RLDXDataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Memory module (optional, enabled by config.use_memory)
        self.use_memory = getattr(config, "use_memory", False)
        self.memory = None
        self._cached_mq = None

        if self.use_memory:
            self._init_memory(config)

    def _init_memory(self, config: RLDXConfig):
        """Initialize memory module for temporal context aggregation."""
        # Switch backbone to full mode (return VL + cognition tokens)
        if hasattr(self.backbone, "cog_mode"):
            self.backbone.cog_mode = "full"

        self._memory_length = getattr(config, "memory_length", 4)
        self._n_cog_tokens = getattr(self.backbone, "n_cog_tokens", 8)

        raw_mem_nq = getattr(config, "memory_n_cog_tokens", None)
        self._memory_n_cog_tokens = raw_mem_nq if raw_mem_nq is not None else self._n_cog_tokens

        assert self._memory_n_cog_tokens <= self._n_cog_tokens, (
            f"memory_n_cog_tokens ({self._memory_n_cog_tokens}) must be "
            f"<= n_cog_tokens ({self._n_cog_tokens})"
        )

        self._concat_memory = getattr(config, "concat_memory", False)
        self._memory_dropout_ratio = getattr(config, "memory_dropout_prob", 0.0)
        assert not (self._memory_dropout_ratio > 0.0 and not self._concat_memory), (
            "memory_dropout_prob > 0.0 requires concat_memory=True"
        )

        # Build memory module
        memory_cfg = dict(getattr(config, "memory_cfg", {}))
        backbone_hidden_size = self._get_backbone_hidden_size()
        if memory_cfg.get("hidden_size") != backbone_hidden_size:
            _print(
                f"[i] Updating memory hidden_size from {memory_cfg.get('hidden_size')} to {backbone_hidden_size}"
            )
            memory_cfg["hidden_size"] = backbone_hidden_size
            memory_cfg["intermediate_size"] = backbone_hidden_size * 4

        n_mq_mem = self._memory_n_cog_tokens
        memory_cfg["block_attn_size"] = n_mq_mem
        memory_cfg["max_position_embeddings"] = self._memory_length * n_mq_mem

        self.memory = TransformerMemory(**memory_cfg)

        n_out = (self._n_cog_tokens + n_mq_mem) if self._concat_memory else self._n_cog_tokens
        _print(
            f"\n[i] Memory enabled: length={self._memory_length}, "
            f"n_mq_mem={n_mq_mem}, concat={self._concat_memory}, "
            f"dropout={self._memory_dropout_ratio}, output_tokens={n_out}"
        )

        # Memory trainable by default
        for param in self.memory.parameters():
            param.requires_grad = True

    def _get_backbone_hidden_size(self) -> int:
        if hasattr(self.backbone, "qwen_model"):
            return self.backbone.qwen_model.model.language_model.config.hidden_size
        return 4096

    def _apply_backbone_lora(self):
        """Inject PEFT LoRA adapters into the backbone LLM layers (top-N or all).

        Issue #55 — sibling of ``RLDXActionModel._apply_action_model_lora``.
        Mirrors the same plumbing but targets the Qwen3 LLM layers held under
        ``self.backbone.qwen_model.model.language_model.layers``. The
        ``backbone_lora_num_layers`` knob picks the suffix to adapt:
        ``-1`` (or any negative) and any value ``>= total`` ⇒ all layers,
        ``0`` ⇒ no-op skip (logged), ``N > 0`` ⇒ last ``N`` layers only.

        The function freezes the entire backbone first, then PEFT marks only
        the injected LoRA params trainable. Adapter params are immediately
        cast bf16 → fp32 to avoid NaN losses on the first optimizer step
        (mirrors VTC's ``trainable_params_fp32`` policy).
        """
        try:
            from peft import LoraConfig, inject_adapter_in_model
        except ImportError as e:
            raise ImportError(
                "peft is required for backbone_use_lora=True. Install with `pip install peft`."
            ) from e

        config = self.config
        num = int(getattr(config, "backbone_lora_num_layers", -1))
        layers = self.backbone.qwen_model.model.language_model.layers
        total = len(layers)

        if num == 0:
            _print("[Backbone LoRA] backbone_lora_num_layers=0, skipping injection")
            return

        if num < 0 or num >= total:
            layers_to_transform = list(range(total))
        else:
            layers_to_transform = list(range(total - num, total))

        target_modules = list(
            getattr(
                config,
                "backbone_lora_target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        )

        self.backbone.requires_grad_(False)

        lora_cfg = LoraConfig(
            r=int(getattr(config, "backbone_lora_rank", 16)),
            lora_alpha=int(getattr(config, "backbone_lora_alpha", 32)),
            lora_dropout=float(getattr(config, "backbone_lora_dropout", 0.0)),
            bias="none",
            target_modules=target_modules,
            layers_to_transform=layers_to_transform,
            layers_pattern="layers",
        )
        inject_adapter_in_model(lora_cfg, self.backbone)

        # fp32 contract for LoRA adapter params (NaN safety: bf16 AdamW state
        # underflows on the first step). Filter on the ``lora_`` name segment
        # so a future non-LoRA trainable backbone param is not silently
        # promoted. Cast unconditionally — owning the contract here decouples
        # it from ``adapter.py:186-190``'s top-N pre-cast.
        n_cast = 0
        for pname, p in self.backbone.named_parameters():
            if not p.requires_grad:
                continue
            if not any(seg.startswith("lora_") for seg in pname.split(".")):
                continue
            p.data = p.data.to(torch.float32)
            n_cast += 1
        _print(f"[Backbone LoRA] Ensured fp32 dtype on {n_cast} LoRA parameter tensors")

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total_p = sum(p.numel() for p in self.backbone.parameters())
        ratio = (100.0 * trainable / total_p) if total_p > 0 else 0.0
        _print(
            f"[Backbone LoRA] layers_to_transform={layers_to_transform} (total={total}), "
            f"r={lora_cfg.r}, alpha={lora_cfg.lora_alpha}, "
            f"dropout={lora_cfg.lora_dropout}, target_modules={target_modules}"
        )
        _print(f"[Backbone LoRA] trainable params: {trainable} / {total_p} ({ratio:.3f}%)")

    @property
    def _n_output_tokens(self) -> int:
        if not self.use_memory:
            return getattr(self.backbone, "n_cog_tokens", 64)
        if self._concat_memory:
            return self._n_cog_tokens + self._memory_n_cog_tokens
        return self._n_cog_tokens

    def reset_memory(self):
        """Reset recurrent memory state (call at start of new episode)."""
        self._cached_mq = None

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action model."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_model.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            # Non-tensor scalars (int ``rtc_prefix_len``, str, None, etc.)
            # flow through unchanged. ``torch.is_floating_point`` rejects
            # non-Tensor inputs with TypeError, so without this guard the
            # RTC dispatch (which carries an int ``rtc_prefix_len`` next
            # to the action_prefix tensor) crashes inside
            # ``tree.map_structure`` before reaching the model.
            if not torch.is_tensor(x):
                return x
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)

        if self.use_memory:
            backbone_outputs = self._apply_memory_training(backbone_outputs)

        action_outputs = self.action_model(backbone_outputs, action_inputs)
        return action_outputs

    def get_action(self, inputs: dict = None, **kwargs) -> BatchFeature:
        if inputs is None:
            inputs = kwargs
        elif kwargs:
            # PolicyRuntime stuffs top-level keys like ``action_prefix`` and
            # ``rtc_prefix_len`` alongside the wrapped ``inputs`` modality
            # dict, so when this method is called as
            # ``model.get_action(**collated)`` they land in ``kwargs``
            # rather than inside ``inputs``. Without this merge they are
            # silently dropped before the action model's RTC check, so RTC
            # always sees ``action_prefix=None`` and runs cold-start every
            # chunk regardless of what PolicyRuntime injected.
            #
            # Loud-fail on key collision: today the inputs / kwargs key
            # namespaces are disjoint (collator emits modalities; runtime
            # injects RTC/memory control keys), but a silent overwrite
            # would be hard to diagnose if a future modality reuses one
            # of those names — surface the conflict here instead.
            collision = set(inputs).intersection(kwargs)
            if collision:
                raise ValueError(
                    "get_action: keys collide between `inputs` dict and "
                    f"top-level kwargs: {sorted(collision)}. Pass each key "
                    "in exactly one place."
                )
            inputs = {**inputs, **kwargs}

        reset_memory = inputs.pop("reset_memory", None) if self.use_memory else None

        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)

        if self.use_memory:
            backbone_outputs = self._apply_memory_inference(backbone_outputs, reset_memory)

        action_outputs = self.action_model.get_action(backbone_outputs, action_inputs)
        return action_outputs

    # ── Memory processing ──

    def _apply_memory_training(self, backbone_outputs: BatchFeature) -> BatchFeature:
        """Process backbone features through memory for training."""
        backbone_features = backbone_outputs["backbone_features"]  # [B*K, T, d]
        BK, T, d = backbone_features.shape
        K = self._memory_length
        B = BK // K
        n_q = self._n_cog_tokens
        n_mq_mem = self._memory_n_cog_tokens
        n_mq_pass = n_q - n_mq_mem

        mq_all = backbone_features[:, -n_q:, :].contiguous().view(B, K, n_q, d)
        mq_original = mq_all[:, -1, :, :]  # [B, n_q, d]
        mq_for_memory = mq_all[:, :, n_mq_pass:, :]  # [B, K, n_mq_mem, d]

        mq_mem_seq = mq_for_memory.contiguous().view(B, K * n_mq_mem, d)
        mq_memory_out = self.memory(inputs_embeds=mq_mem_seq).last_hidden_state
        mq_augmented = mq_memory_out.view(B, K, n_mq_mem, d)[:, -1, :, :]

        if self._concat_memory:
            backbone_outputs["backbone_features"] = torch.cat([mq_original, mq_augmented], dim=1)
        else:
            if n_mq_pass > 0:
                mq_pass = mq_original[:, :n_mq_pass, :]
                backbone_outputs["backbone_features"] = torch.cat([mq_pass, mq_augmented], dim=1)
            else:
                backbone_outputs["backbone_features"] = mq_augmented

        # Rebuild attention mask
        if "backbone_attention_mask" in backbone_outputs:
            attn_mask = backbone_outputs["backbone_attention_mask"]
            attn_mask_last = attn_mask.view(B, K, -1)[:, -1, :]
            mq_mask = attn_mask_last[:, -n_q:]
            if self._concat_memory:
                mem_mask = mq_mask[:, -n_mq_mem:]
                backbone_outputs["backbone_attention_mask"] = torch.cat([mq_mask, mem_mask], dim=1)
            else:
                backbone_outputs["backbone_attention_mask"] = mq_mask

        # Memory dropout (concat mode only)
        if self.training and self._memory_dropout_ratio > 0.0 and self._concat_memory:
            if "backbone_attention_mask" in backbone_outputs:
                B_out = backbone_outputs["backbone_features"].shape[0]
                attn_mask = backbone_outputs["backbone_attention_mask"].clone()
                do_dropout = torch.rand(B_out, device=attn_mask.device) < self._memory_dropout_ratio
                dropout_mask = do_dropout[:, None].expand(-1, n_mq_mem)
                attn_mask[:, -n_mq_mem:] = attn_mask[:, -n_mq_mem:].masked_fill(dropout_mask, 0)
                backbone_outputs["backbone_attention_mask"] = attn_mask

        return backbone_outputs

    def _apply_memory_inference(
        self, backbone_outputs: BatchFeature, reset_memory=None
    ) -> BatchFeature:
        """Process backbone features through memory for inference (single timestep)."""
        backbone_features = backbone_outputs["backbone_features"]
        B, T, d = backbone_features.shape
        n_q = self._n_cog_tokens
        n_mq_mem = self._memory_n_cog_tokens
        n_mq_pass = n_q - n_mq_mem

        mq_all = backbone_features[:, -n_q:, :]
        mq_current = mq_all[:, n_mq_pass:, :]

        # Manage recurrent cache
        if self._cached_mq is None or self._cached_mq.shape[0] != B:
            self._cached_mq = mq_current.repeat(1, self._memory_length, 1)
        else:
            if reset_memory is not None and reset_memory.any():
                reset_defaults = mq_current.repeat(1, self._memory_length, 1)
                shifted_cache = torch.cat([self._cached_mq[:, n_mq_mem:, :], mq_current], dim=1)
                reset_expanded = reset_memory.view(B, 1, 1).expand(
                    B, self._memory_length * n_mq_mem, d
                )
                self._cached_mq = torch.where(reset_expanded, reset_defaults, shifted_cache)
            else:
                self._cached_mq = torch.cat([self._cached_mq[:, n_mq_mem:, :], mq_current], dim=1)

        mq_memory_out = self.memory(inputs_embeds=self._cached_mq).last_hidden_state
        mq_augmented = mq_memory_out[:, -n_mq_mem:, :]

        if self._concat_memory:
            backbone_outputs["backbone_features"] = torch.cat([mq_all, mq_augmented], dim=1)
        else:
            if n_mq_pass > 0:
                mq_pass = mq_all[:, :n_mq_pass, :]
                backbone_outputs["backbone_features"] = torch.cat([mq_pass, mq_augmented], dim=1)
            else:
                backbone_outputs["backbone_features"] = mq_augmented

        # Rebuild attention mask
        if "backbone_attention_mask" in backbone_outputs:
            attn_mask = backbone_outputs["backbone_attention_mask"]
            mq_mask = attn_mask[:, -n_q:]
            if self._concat_memory:
                mem_mask = mq_mask[:, -n_mq_mem:]
                backbone_outputs["backbone_attention_mask"] = torch.cat([mq_mask, mem_mask], dim=1)
            else:
                backbone_outputs["backbone_attention_mask"] = mq_mask

        return backbone_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("RLDX-1", RLDXConfig)
AutoModel.register(RLDXConfig, RLDX)
