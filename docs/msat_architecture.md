# MSAT Architecture: Original N1.5 → RLDX MSAT → WM Variant

## Overview

이 문서는 action model의 어텐션 아키텍처 변천사를 다룹니다.

- **Original GR00T N1.5**: Cross-Attention DiT (일방향)
- **RLDX MSAT**: Flux-style DoubleStreamBlock (양방향 joint attention)
- **RLDX MSAT v1**: cognition tokens로 VL 토큰 대체
- **RLDX MSAT + WM**: TripleStreamBlock + World Modeling + Memory

---

## 1. Original GR00T N1.5: Cross-Attention DiT

NVIDIA 원본 GR00T N1.5 action model (`cross_attention_dit.py`).
`BasicTransformerBlock` x 12 레이어, diffusers 기반.

```
  VL tokens ─────────────────────────────────┐ (frozen K, V)
  (backbone_features)                         │
                                              │
  SA tokens ──→ ┌───────────────────────────┐ │
  (state+action) │  BasicTransformerBlock x12│ │
                 │                           │ │
  Timestep ─┐   │   ┌──────────────┐        │ │
     t       │   │   │  AdaLayerNorm │        │ │
             ▼   │   │  (timestep    │        │ │
          AdaLN──│──→│   modulation) │        │ │
                 │   └──────┬───────┘        │ │
                 │          ▼                 │ │
                 │   ┌──────────────┐        │ │
                 │   │Self-Attention │        │ │
                 │   │  SA ↔ SA     │        │ │
                 │   │  (SA끼리만)   │        │ │
                 │   └──────┬───────┘        │ │
                 │          ▼                 │ │
                 │   ┌──────────────┐        │ │
                 │   │Cross-Attention│←───────┘ │
                 │   │  Q=SA, K,V=VL│  VL은 K,V로만 참여
                 │   │  (SA→VL 일방향)│  VL 자체는 업데이트 안됨
                 │   └──────┬───────┘        │
                 │          ▼                 │
                 │   ┌──────────────┐        │
                 │   │     FFN      │        │
                 │   │   (GELU)     │        │
                 │   └──────┬───────┘        │
                 │          ▼                 │
                 │       SA_out              │
                 └──────────┬────────────────┘
                            ▼
                  Linear → action_pred
```

**핵심 특징**:
- **VL은 K, V로만 참여** → SA가 VL을 참조하지만, VL 자체는 변하지 않음 (일방향)
- **Self-attention과 cross-attention이 분리**된 sequential 구조
- Timestep 조건: **AdaLayerNorm** (feature-wise multiplicative modulation)
- Positional embedding: **Sinusoidal**
- 단일 stream, 단순 구조

---

## 2. RLDX MSAT v0: Flux-style DoubleStreamBlock

`joint_attention_dit.py`. Flux에서 영감받은 dual-stream joint attention.
이 단계에서는 backbone의 VL 토큰 전체가 action model에 진입.

```
                         ┌──────────────────────────────────────┐
  Inputs                 │         DoubleStreamBlock x N        │
  ───────                │         (Multi-Stream Phase)         │
                         │                                      │
  VL tokens ─────────┐   │   ┌─────────┐     ┌─────────┐       │
  (backbone_features) │   │   │ VL Norm │     │ SA Norm │       │
                      │   │   │ + Mod   │     │ + Mod   │       │
  SA tokens ─────┐   │   │   └────┬────┘     └────┬────┘       │
  (state+action)  │   │   │        │               │            │
                  │   │   │        ▼               ▼            │
  Timestep ──┐   │   │   │     VL QKV          SA QKV          │
     t       │   │   │   │        │               │            │
             ▼   │   │   │        └──────┬────────┘            │
          ┌──────┴┐  │   │               ▼                     │
          │ Temb  │  │   │     ┌──────────────────┐            │
          │Encoder│  │   │     │  Joint Attention  │            │
          └───┬───┘  │   │     │ Q=cat([VL,SA])    │            │
              │      │   │     │ K=cat([VL,SA])    │  ← 양방향! │
              │      │   │     │ → SDPA → split    │            │
              │      │   │     └────────┬─────────┘            │
              ▼      ▼   │          ┌───┴───┐                  │
          temb ──→ Mod   │          ▼       ▼                  │
                         │      VL attn  SA attn               │
                         │          │       │                  │
                         │      VL Proj  SA Proj               │
                         │      + Gate   + Gate                │
                         │          │       │                  │
                         │      VL MLP   SA MLP                │
                         │      + Gate   + Gate                │
                         │          │       │                  │
                         │          ▼       ▼                  │
                         │     VL_out    SA_out  ← VL도 업데이트!│
                         └──────────┬───────┬──────────────────┘
                                    │       │
                                    ▼       ▼
                         ┌──────────────────────────────────────┐
                         │       SingleStreamBlock x M          │
                         │       (Single-Stream Phase)          │
                         │                                      │
                         │  x = cat([VL_proj, time_tok, SA])    │
                         │         ┌──────────┐                 │
                         │         │ Pre-Norm │                 │
                         │         │  + Mod   │                 │
                         │         └────┬─────┘                 │
                         │              │                       │
                         │    ┌─────────┴─────────┐             │
                         │    ▼                   ▼             │
                         │  QKV (parallel)     MLP_in           │
                         │    │                   │             │
                         │    ▼                   ▼             │
                         │  Self-Attention    SwiGLU/GELU       │
                         │    │                   │             │
                         │    └─────────┬─────────┘             │
                         │              ▼                       │
                         │         linear2 + Gate               │
                         │              │                       │
                         │              ▼                       │
                         │       x (residual updated)           │
                         └──────────────┬───────────────────────┘
                                        │
                                        ▼
                              AdaLN-zero → proj_out
                                        │
                                        ▼
                                  action_pred
```

**핵심 특징**:
- **VL과 SA가 양방향 joint attention** → VL도 매 레이어 업데이트됨
- 각 stream마다 **독립된 QKV projection, MLP, Modulation**
- 어텐션은 모든 토큰을 **concat해서 한번에** 수행 (Joint Attention) → 결과를 다시 split
- Timestep 조건: **Modulation** (Flux-style AdaLN) 또는 **input_token** (in-context)
- Positional embedding: **RoPE** (multi-axis)
- FFN: SwiGLU 지원
- QK Norm: RMSNorm 지원

---

## 2.1. RLDX MSAT v1: cognition tokens로 VL 대체

MSAT v1부터 backbone의 VL 토큰 전체 대신 **learnable cognition tokens (cog)**가 도입.
backbone에서 cog 토큰만 추출하여 action model에 전달 → VL 토큰은 action model에 진입하지 않음.

```
  Backbone output: [ VL tokens (T-n_q개) | cog tokens (n_q개) ]
                                            │
                                            ▼
                              (VL 토큰 버림, cog만 사용)
                                            │
                                            ▼
                     DoubleStreamBlock:  cog  |  SA
                                         ↕ joint attention
```

**v0 → v1 차이점**:
- Stream 1이 VL 전체 (~수백 토큰) → **cog (8~16 토큰)**으로 교체
- 어텐션 시퀀스 길이 대폭 감소 → 연산 효율 향상
- cog는 backbone 내부에서 VL 정보를 압축한 learnable query
- `cog_mode`에 따라 다양한 조합 가능 (full, meta_only, meta_n_meta)

---

## 3. RLDX MSAT + WM: TripleStreamBlock + Memory

`joint_attention_dit_wm.py`. 현재 설정 (`set_triple_stream_for_wm=True` + `cog_mode=cog_only`).

**중요**: Stream 1에 들어가는 것은 VL 토큰이 아니라 **cog\* (memory-augmented cognition tokens)**.

```
  Backbone: [ VL tokens | cog tokens ] x (memory_length + 1) timesteps
                            │
              ┌─────────────┴─────────────┐
              │    TransformerMemory       │
              │  과거 cog들 aggregation      │
              │  → cog* (memory-augmented)  │
              └─────────────┬─────────────┘
                            │
              (VL 토큰 버림, cog*만 사용)
                            ▼
```

```
                         ┌─────────────────────────────────────────────────┐
  Inputs                 │           TripleStreamBlock x N                 │
  ───────                │           (Multi-Stream Phase)                  │
                         │                                                 │
  cog* tokens ─────┐      │   ┌────────┐   ┌────────┐   ┌────────┐        │
  (memory-        │      │   │cog* Norm│   │WM Norm │   │SA Norm │        │
   augmented,     │      │   │ + Mod  │   │ + Mod  │   │ + Mod  │        │
   8 tokens)      │      │   └───┬────┘   └───┬────┘   └───┬────┘        │
                  │      │       │             │             │             │
  WM tokens ──┐  │      │    cog* QKV       WM QKV        SA QKV          │
  (DINO embs) │  │      │       │             │             │             │
              │  │      │       └─────────────┼─────────────┘             │
  SA tokens ┐ │  │      │                     ▼                           │
  (S+A)     │ │  │      │        ┌─────────────────────────┐              │
            │ │  │      │        │     Joint Attention      │              │
  t ──→ temb│ │  │      │        │  Q=cat([cog*, WM, SA])    │              │
  t_wm→    ││ │  │      │        │  K=cat([cog*, WM, SA])    │  ← 3자 양방향│
   wm_temb ││ │  │      │        │  → SDPA → split          │              │
            ▼▼ ▼  ▼      │        └──────┬────┬────┬────────┘              │
                         │               ▼    ▼    ▼                      │
                         │          cog*_attn WM_attn SA_attn              │
                         │               │    │    │                      │
                         │          cog*_proj WM_proj SA_proj              │
                         │           +Gate  +Gate  +Gate                  │
                         │               │    │    │                      │
                         │          cog*_MLP WM_MLP SA_MLP                 │
                         │           +Gate  +Gate  +Gate                  │
                         │               │    │    │                      │
                         │               ▼    ▼    ▼                      │
                         │          cog*_out WM_out SA_out                 │
                         └───────────────┬────┬────┬──────────────────────┘
                                         │    │    │
                    ┌────────────────────┘    │    └──────────────────┐
                    │                         │                       │
                    ▼                         ▼                       ▼
          ┌─────────────────────┐  ┌─────────────────────┐
          │ SA SingleStreamBlock│  │ WM SingleStreamBlock│   ← 별도 블록!
          │       x M           │  │       x M           │
          │                     │  │                     │
          │ x = cat(            │  │ wm_x = cat(         │
          │  [cog*_proj,         │  │  [cog*_proj,         │
          │   time_tok,         │  │   wm_time_tok,      │
          │   SA])              │  │   WM])              │
          │                     │  │                     │
          │ Self-Attention      │  │ Self-Attention      │
          │ + SwiGLU MLP        │  │ + SwiGLU MLP        │
          │                     │  │                     │
          └─────────┬───────────┘  └─────────┬───────────┘
                    │                         │
                    ▼                         ▼
          AdaLN-zero → proj_out     AdaLN-zero → wm_proj_out
                    │                         │
                    ▼                         ▼
              action_pred              wm_pred (DINO)
```

---

## 4. 3세대 비교 (핵심 차이)

### 4.1 Attention Mechanism

```
=== Original N1.5 (Cross-Attention DiT) ===

  x12 layers:
    SA → SA  (self-attention)
    SA → VL  (cross-attention, 일방향)
              VL은 frozen K,V — 업데이트 안 됨


=== RLDX MSAT v0 (DoubleStream, VL 토큰) ===

  Multi-Stream (x4 layers):
    VL ←──joint_attn──→ SA          # VL과 SA가 양방향 attend
                                     # VL도 매 레이어 업데이트!
  Single-Stream (x8 layers):
    [VL + SA] → self_attn           # 합쳐서 self-attention


=== RLDX MSAT v1 (DoubleStream, cognition tokens) ===

  Multi-Stream (x4 layers):
    cog ←──joint_attn──→ SA          # VL 대신 cog (8~16개)가 참여
                                     # 어텐션 시퀀스 길이 대폭 감소
  Single-Stream (x8 layers):
    [cog + SA] → self_attn


=== RLDX MSAT + WM (TripleStream, cog* + Memory) ===

  Multi-Stream (x4 layers):
    cog* ←──joint_attn──→ WM         # cog*, WM, SA 3자가 서로 attend
    cog* ←──joint_attn──→ SA         # (하나의 큰 attention에서 동시에)
    WM  ←──joint_attn──→ SA         # cog* = memory-augmented cognition tokens

  Single-Stream (x8 layers):
    [cog* + SA] → self_attn          # SA: cog* context 참조하며 action refine
    [cog* + WM] → self_attn          # WM: cog* context 참조하며 DINO refine
                                     # (SA↔WM 간 직접 attention 없음)
```

### 4.2 종합 비교표

| Aspect | Original N1.5 | MSAT v0 | MSAT v1 (cog) | MSAT + WM (현재) |
|--------|:-------------:|:--------:|:-------------:|:-----------------:|
| **File** | `cross_attention_dit.py` | `joint_attention_dit.py` | `joint_attention_dit.py` | `joint_attention_dit_wm.py` |
| **Block 타입** | `BasicTransformerBlock` | `DoubleStream` + `SingleStream` | `DoubleStream` + `SingleStream` | `TripleStream` + 2x `SingleStream` |
| **Attention 방향** | SA→VL 일방향 | VL↔SA 양방향 | cog↔SA 양방향 | cog\*↔WM↔SA 양방향 |
| **조건 토큰** | VL 전체 (~수백) | VL 전체 (~수백) | **cog (8~16개)** | **cog\* (8개, memory-augmented)** |
| **조건 토큰 업데이트** | X (frozen K,V) | O (매 레이어) | O (매 레이어) | O (매 레이어) |
| **Stream 수** | 1 (SA only) | 2 (VL, SA) → 1 | 2 (cog, SA) → 1 | 3 (cog\*, WM, SA) → 2 병렬 |
| **Timestep 조건** | AdaLayerNorm | Modulation / input_token | Modulation / input_token | Modulation / input_token (x2) |
| **Pos. Embedding** | Sinusoidal | RoPE (2D) | RoPE (2D) | RoPE (3D multimodal) |
| **FFN** | GELU | GELU / SwiGLU | GELU / SwiGLU | GELU / SwiGLU |
| **QK Norm** | None | RMSNorm / LayerNorm | RMSNorm / LayerNorm | RMSNorm / LayerNorm |
| **출력** | action_pred | action_pred | action_pred | action_pred + wm_pred |
| **Default depth** | 12 layers | 4 multi + 8 single | 4 multi + 8 single | 4 multi + 8 single (x2) |

### 4.3 Positional Embedding (RoPE)

```
rope_multimodal_3d (head_dim=64, WM variant):

  ┌──────────────────────────────────────────────────┐
  │  Axis 0 (16d): Time                              │
  │    VL tokens: T=0 (context)                      │
  │    WM tokens: T=0 (current/future frame)         │
  │    SA tokens: (unused, =0)                        │
  │                                                    │
  │  Axis 1 (24d): DINO Height / SA sequential        │
  │    VL tokens: (unused, =0)                        │
  │    WM tokens: y_grid (0..H-1, repeating W times)  │
  │    SA tokens: 0, 1, 2, ..., N_sa-1                │
  │                                                    │
  │  Axis 2 (24d): DINO Width                         │
  │    VL tokens: (unused, =0)                        │
  │    WM tokens: x_grid (0..W-1, repeating H times)  │
  │    SA tokens: (unused, =0)                        │
  └──────────────────────────────────────────────────┘
```

### 4.4 Single-Stream Phase 차이 (WM의 가장 큰 구조적 특징)

**MSAT v0/v1**: Multi-stream 후 모든 것을 하나로 concat → single-stream attention

```
  v0: [VL_proj | time_tok | SA]  → Self-Attention → action
  v1: [MQ_proj | time_tok | SA]  → Self-Attention → action
```

**WM**: SA와 WM가 **별도의 single-stream block**을 가짐 (lockstep 실행)

```
  SA Stream:  [cog*_proj | time_tok | SA]      → Self-Attention → action
  WM Stream:  [cog*_proj | wm_time_tok | WM]   → Self-Attention → wm_pred
              ↑ 이 둘은 서로 attention 공유 없음 (독립)
```

Multi-stream (TripleStreamBlock)에서는 joint attention으로 서로 정보를 교환하지만,
Single-stream에서는 각자 cog* context만 참조하여 독립적으로 refine.

---

## 5. 현재 설정 — 구체적 수치

| Parameter | Value |
|-----------|-------|
| `num_attention_heads` | 24 |
| `attention_head_dim` | 64 |
| `inner_dim` (= heads * head_dim) | 1536 |
| `depth_multi_stream` (TripleStreamBlock) | 4 |
| `depth_single_stream` (SingleStreamBlock) | 8 |
| `sa_dim` | 1536 |
| `vl_dim` (backbone embedding) | 4096 (Qwen3-VL 8B) |
| `wm_dim` | 1536 |
| `positional_embeddings` | `rope_multimodal_3d` |
| `temb_type` | `input_token` |
| `qk_norm` | `rms_norm` |
| `use_swiglu` | True |
| `pre_norm` | `layer_norm` |
