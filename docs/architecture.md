# Architecture

High-level walkthrough of the RLDX-1 model. This is the "what is in the
box" view; configuration knobs are cross-linked to source and to
[`training.md`](training.md) for hands-on use.

```
                ┌──────────────────────────────────────────┐
                │              Image pipeline              │
                │  aspect-area resize → optional crop       │
                │  → albumentations stack → uint8 (C,H,W)   │
                └──────────────────────┬───────────────────┘
                                       │
        text instruction               │ multi-view uint8 frames
        ────────────────┐              │
                        ▼              ▼
                ┌──────────────────────────────────────────┐
                │     Backbone (Qwen3-VL / VTC)      │
                │   VLM hidden states + learnable tokens    │
                └──────────────────────┬───────────────────┘
                                       │
                         ┌─────────────┼─────────────┐
                         ▼             ▼             ▼
                 Cognition tokens   State tokens   (optional)
                                    from cat-MLP   Physics tokens
                                       │             │
                                       ▼             ▼
                ┌──────────────────────────────────────────┐
                │       MSAT action model (flow matching)  │
                │     triple-stream self+cross attention    │
                │         RoPE on SA stream by default      │
                └──────────────────────┬───────────────────┘
                                       │
                                       ▼
                ┌──────────────────────────────────────────┐
                │   Action decoder (category-specific MLP) │
                │   16-step continuous action chunk         │
                └──────────────────────────────────────────┘
```

## The model at a glance

- **Total size:** 6.9B params (base) / 8.0B params (with memory + motion +
  physics add-ons), bf16.
- **Input:** one or more camera views (each a short clip of PIL frames),
  a low-dimensional proprioceptive state vector, an optional physics
  vector (tactile + torque), and a natural-language task instruction.
- **Output:** an **action chunk** of `action_horizon = 16` steps of
  continuous actions, decoded per-embodiment. Actions are produced by
  iterative flow-matching denoising (default 4 inference timesteps).
- **Embodiments:** a single checkpoint supports every embodiment seen
  during training via category-conditional MLP encoders and decoders.
  Registered tags live in `rldx/data/embodiment_tags.py`.

## Stage 1 — image pipeline

Code: [`rldx/data/augmentations.py`](../rldx/data/augmentations.py)
Config: `image_max_area`, `image_resize_m`, `random_crop_fraction`,
`random_rotation_angle`, `color_jitter_params` on `RLDXConfig`.

Every camera frame goes through the same two-stage pipeline:

1. **AspectAreaResizeAndCrop** (deterministic). Downscales so the total
   area is ≤ `image_max_area`, keeping aspect ratio, then center-crops
   both dims to multiples of `image_resize_m`. The default `(max_area=65536,
   m=32)` maps `(480, 640) → (192, 256)` and leaves `(256, 256)`
   untouched.
2. **Fractional crop + resize-back** (optional). When
   `random_crop_fraction` is set, a `fraction * H × fraction * W`
   sub-region is cropped (**random** position during training,
   **center** at eval time) and then resized back to the pre-crop shape
   so downstream stages always see a fixed spatial size. The stage is
   skipped entirely when the fraction is `None` (the default).
3. **Train-only photometric / geometric augmentation.** Rotate
   (`random_rotation_angle`) and ColorJitter (`color_jitter_params`)
   are added to the train transform only.

The train and eval transforms are built once at processor construction
and selected based on `ProcessorMixin.training` at call time
(`processor.train()` / `processor.eval()`). A replay mechanism keeps
all camera views in a batch on the same random parameters.

## Stage 2 — backbone

Code: [`rldx/model/modules/backbone/`](../rldx/model/modules/backbone/)

Two concrete backbones, both Qwen3-VL derivatives:

- **`vtc_qwen3_vl`** — Qwen3-VL-8B (from `Qwen/Qwen3-VL-8B-Instruct`)
  wrapped in [VTC](../rldx/model/modules/backbone/) for temporal
  fusion across multiple frames per view. This is the sole supported
  backbone since Issue #36 (the vanilla `qwen3_vl` variant was
  production-dead and removed).

The [`VTCQwen3VLBackbone`](../rldx/model/modules/backbone/adapter.py)
adapter exposes:

- `encode(vlm_inputs, prompt_embeds, ...)` → `(vlm_hidden, cognition_tokens)`
- `embed_tokens(...)` → injects the learnable cognition-token embedding
  before the transformer stack, so the backbone produces dedicated token
  positions the action model can read from.

Relevant `RLDXConfig` fields:
  - `model_name`, `model_revision` — backbone weights to load
  - `backbone_model_type` — fixed at `"vtc_qwen3_vl"`; any other value
    raises at RLDX construction time
  - `select_layer` — which hidden layer feeds the action model
  - `backbone_embedding_dim`, `input_embedding_dim` — projection targets
  - `n_cog_tokens` (default 64) — number of learnable cognition tokens
  - `tune_llm`, `tune_visual`, `tune_top_llm_layers` — trainable slices
  - `freeze_cog_tokens` — freeze the cognition-token embedding to
    prevent VLM backprop

## Stage 3 — cognition tokens

The cognition-token bridge is what lets the action model read VLM hidden
state without being coupled to the language-modelling head. At init,
`n_cog_tokens` learnable embeddings are appended to the backbone
input. The backbone sees them like any other token and updates them
through every layer. The action model later attends to (or receives
directly) those cognition tokens.

Two downstream modes for cognition tokens:

- **Single-step** (`use_memory=False`, default). Each forward pass runs
  one backbone call; the action model uses the cognition tokens from
  that single pass.
- **History-aware** (`use_memory=True`). Past timesteps' cognition
  tokens are cached and fused via a Transformer memory module
  ([`rldx/model/modules/memory.py`](../rldx/model/modules/memory.py))
  before reaching the action model. Controls: `memory_length`,
  `memory_n_cog_tokens`, `concat_memory`, `memory_dropout_prob`.

## Stage 4 — MSAT action model

Code: [`rldx/model/modules/action_model/`](../rldx/model/modules/action_model/)
(`msat.py` `blocks.py` `attention.py` `ops.py`)

The action model is a Multi-Stream Action Transformer (MSAT) trained
with **flow matching**: given a noised action chunk, the model predicts
the velocity field that transports it back to the clean chunk. At
inference time we integrate that field through
`num_inference_timesteps = 4` Euler steps (tunable).

Three token streams flow through every MSAT block:
- **VL stream** — projected cognition tokens + optional raw VLM hidden
  states
- **SA (state/action) stream** — noised action chunk + state tokens
- **P stream** — present when `use_physics=True`: projected physics
  tokens (tactile / torque / both)

Block variants:
- `DoubleStreamBlock` — VL + SA only (default)
- `TripleStreamBlock` — VL + SA + P, used when physics is on
- `SingleStreamBlock` / `ExpandedSingleStreamBlock` — late layers after
  the cross-stream mixing

Each block does modulated pre-norm, multi-head self-attention across
the concatenated streams, and a SwiGLU MLP. Positional information is
provided by **RoPE** on the SA stream (`rope_sa_only`, the default) or
optionally also on the VL stream (`rope_vl_sa`).

Key `RLDXConfig.diffusion_model_cfg` entries (all are overridable via
the CLI or from a YAML config):

| Key | Default | What |
|---|---|---|
| `depth_multi_stream` | 4 | `DoubleStreamBlock` / `TripleStreamBlock` count |
| `depth_single_stream` | 8 | `SingleStreamBlock` count |
| `num_attention_heads` | 24 | MSAT attention heads |
| `attention_head_dim` | 64 | per-head dim |
| `output_dim` | 1024 | action-head hidden size |
| `positional_embeddings` | `rope_sa_only` | `rope_sa_only` / `rope_vl_sa` / None |
| `temb_type` | `input_token` | how the timestep embedding is injected |
| `use_swiglu` | `True` | (only SwiGLU path is supported) |

## Stage 5 — action encoder / decoder

Code: [`rldx/model/modules/embodiment_conditioned_mlp.py`](../rldx/model/modules/embodiment_conditioned_mlp.py)

A pair of category-specific MLPs bridge embodiment-specific action
spaces and the shared MSAT hidden space:

- `MultiEmbodimentActionEncoder`: `(B, action_horizon, max_action_dim)` →
  `(B, action_horizon, input_embedding_dim)`. One linear head per
  embodiment index.
- `CategorySpecificMLP` decoder: MSAT output →
  `(B, action_horizon, max_action_dim)`. One linear head per embodiment
  index.

The state encoder is the same pattern: a `CategorySpecificMLP` maps
`max_state_dim` raw proprioception into `hidden_size`. State is padded
to `max_state_dim = 64` and action is padded to `max_action_dim = 64`;
the per-embodiment head selects the live slice.

## Optional modules

- **Memory** (`use_memory=True`): multi-timestep cognition-token
  aggregation via `rldx/model/modules/memory.py`.
- **Motion module** (`use_motion=True`): motion-aware vision feature
  extractor injected into the Qwen3-VL encoder. Aimed at fine-grained
  temporal dynamics. Source:
  `rldx/model/modules/vtc/motion.py`.
- **Physics** (`use_physics=True`, `physics_keys=[...]`, `physics_dims=[...]`):
  tactile / torque stream joined as a third MSAT stream. Flow matching
  supervises both action and future physics signals.

## Training vs inference

- **Training**:
  - `processor.train()` selects the randomised image transform.
  - `RLDX.forward` samples a flow-matching timestep, noises the ground-
    truth action chunk, and computes the velocity-MSE loss.
  - Distributed via DeepSpeed ZeRO-2 (default) or ZeRO-3.
- **Inference**:
  - `processor.eval()` selects the deterministic transform.
  - `RLDX.get_action` integrates the learned velocity field for
    `num_inference_timesteps` Euler steps, starting from Gaussian noise,
    and returns the decoded `action_horizon`-step chunk.
  - `RLDXPolicy` wraps that into a stateless policy interface used by
    the ZeroMQ server (`rldx/eval/run_rldx_server.py`) and the local
    rollout runner (`rldx/eval/rollout_policy.py`).

See [`training.md`](training.md) and [`inference_server.md`](inference_server.md)
for the CLI and server recipes.
