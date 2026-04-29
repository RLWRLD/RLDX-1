# Inference server

`rldx/eval/run_rldx_server.py` is the canonical way to serve an RLDX
checkpoint. It holds the model on the GPU and answers policy queries
over a simple TCP / ZeroMQ REQ/REP protocol with msgpack payloads, so
any client — simulator rollout, real-robot driver, tele-op bridge —
can use it as long as it speaks the protocol.

This doc covers the server CLI, the client-side handshake, and two
deployment recipes (simulator eval, real robot).

> **Alternative transport (WebSocket).**
> A second server binary `rldx/eval/run_rldx_server_pi.py` exists for
> openpi-compatible WebSocket clients (`openpi_client.websocket_client_policy`);
> it wraps the same `RLDXPolicy` as the canonical ZeroMQ path. It is the
> transport used today by `rldx/eval/sim/CALVIN/eval_taskwise.py`. Whether
> to ship it, retire it, or consolidate both behind `--transport` is a
> release-level open item tracked in
> [`TODO_BEFORE_RELEASE.md`](../TODO_BEFORE_RELEASE.md). Everything below
> describes the canonical ZeroMQ server; when you need WebSocket, use
> `run_rldx_server_pi.py` with an openpi client.

## Quick start

```bash
# Start the server (holds the GPU)
uv run python rldx/eval/run_rldx_server.py \
    --model-path RLWRLD/RLDX-1-FT-ROBOCASA \
    --embodiment-tag GENERAL_EMBODIMENT \
    --use-sim-policy-wrapper \
    --host 127.0.0.1 --port 5555
```

On first launch the server downloads the backbone, loads the action
head, runs the MSAT init, and starts listening. Expect ~30-60 s for
an 8B Qwen3-VL backbone on a warm disk cache, up to several minutes on
a cold cache.

Once you see

```
Starting RLDX inference server...
  Embodiment tag: EmbodimentTag.GENERAL_EMBODIMENT
  Model path: RLWRLD/RLDX-1-FT-ROBOCASA
  Device: cuda
  Host: 127.0.0.1
  Port: 5555
```

followed by the MSAT init log, the server is ready to accept clients.

## Server CLI reference

All flags come from `ServerConfig` in `rldx/eval/run_rldx_server.py`
and are parsed by `tyro`.

### Loading the policy

| Flag | Default | Notes |
|---|---|---|
| `--model-path` | None | HF hub id or absolute path to a trained RLDX checkpoint. Processor is loaded from `{model_path}/processor`. Mutually exclusive with `--dataset-path`. |
| `--dataset-path` | None | Path to a LeRobot dataset. Loads a [`ReplayPolicy`](../rldx/policy/replay_policy.py) that answers queries with recorded actions instead of model inference. Useful for debugging the wire protocol without loading the real model. |
| `--modality-config-path` | None | When using `--dataset-path`, optional JSON with the modality config. Defaults to the entry in `MODALITY_CONFIGS` keyed by `--embodiment-tag`. |
| `--embodiment-tag` | `GENERAL_EMBODIMENT` | Category-specific MLP head to use. Complete list in `rldx/data/embodiment_tags.py`. |
| `--device` | `cuda` | Pass `cuda:1` to pin to a specific GPU. |
| `--strict` | True | Validate input/output shapes against the modality config. Turn off only for debugging. |

### Inference-time sampling knobs

| Flag | Default | Notes |
|---|---|---|
| `--sample-timestep-from-beta-dist` | False | Use a Beta-distributed flow-matching timestep at training time instead of uniform. Inference-only; used for the beta-dist ablation. |
| `--denoising-timesteps` | None | Fixed denoising schedule, e.g. `0.0 0.1 0.3 0.6`. Overrides the default `num_inference_timesteps = 4` Euler steps. |
| `--deactivate-memory` | False | Load a memory-trained checkpoint as if it were a non-memory model. Memory weights are skipped on load. Useful for "does the memory module actually help" ablations. |
| `--execution-horizon` | None | Only consulted for `ReplayPolicy`; sets how many steps of the replayed action chunk to return per query. |

### Server / protocol

| Flag | Default | Notes |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address. Use `0.0.0.0` for LAN access. |
| `--port` | `5555` (`DEFAULT_MODEL_SERVER_PORT`) | Server port. |
| `--use-sim-policy-wrapper` | False | Wraps the policy in [`RLDXSimPolicyWrapper`](../rldx/policy/rldx_policy.py). The wrapper adapts observation and action shapes to what the simulator rollout worker expects. **Always enable for simulator eval.** |
| `--verbose` | False | Print per-query debug logs (first-step shape dump, action chunk dtype, etc). |

## Client protocol (`PolicyServer` / `PolicyClient`)

The server is built on
[`rldx/policy/server_client.py`](../rldx/policy/server_client.py).
`PolicyServer` is a ZeroMQ REQ/REP server (`zmq.REP` socket bound to
`tcp://{host}:{port}`); `PolicyClient` is the matching `zmq.REQ`
client. Each request is a msgpack-encoded dict
`{"endpoint": "<name>", "data": {...}, "api_token": "..."}`; the reply
is the endpoint's return value, also msgpack-encoded.

Default endpoints registered by `PolicyServer.__init__`:

| Endpoint | Purpose |
|---|---|
| `ping` | Liveness check; returns immediately. No input. |
| `kill` | Stop the server. No input. |
| `get_action` | Forward a single observation batch and return the predicted action chunk. |
| `reset` | Drop per-session state (memory cache, RTC chunk cache). |

Numpy arrays and `ModalityConfig` are tunneled through msgpack via the
custom `MsgSerializer.encode_custom_classes` hook in `server_client.py`.

### Python client

```python
import numpy as np
from rldx.policy.server_client import PolicyClient

client = PolicyClient(host="127.0.0.1", port=5555)

observations = {
    "video.primary": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
    "video.secondary": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
    "state.joint_position": np.zeros((1, 1, 16), dtype=np.float32),
    "annotation.human.action.task_description": ["pick up the cup"],
}
options = {
    "session_ids": ["my-session-id"],
    "reset_memory": [True],      # first step of an episode
}

result = client.get_action(observations, options=options)
# result["action.joint_position"]: (1, 16, 16) float32 — 16 chunk steps
```

`PolicyClient.get_action(obs, options=…)` is the idiomatic call — every
in-tree client (`rollout_policy.py`, `droid_deploy.py`, `eval_so100.py`)
uses it. It wraps `call_endpoint("get_action", …)` above; use the
low-level form only when reaching a custom endpoint.

Observation keys follow the `{modality_type}.{modality_key}` convention
defined in the embodiment's `ModalityConfig`. Video arrays are uint8
`(B, T, H, W, 3)`; state is float32 `(B, T, dim)`; language is a list
of strings one per batch element. The exact set of keys depends on the
embodiment — consult the modality config you trained with.

### Options dict

| Key | Type | Purpose |
|---|---|---|
| `session_ids` | `list[str]` | Unique id per parallel rollout stream. Required for memory models so past cog caches do not cross contaminate. |
| `reset_memory` | `list[bool]` | Mark the first step of each episode so the server drops the memory cache. |
| `action_pred` | `dict` | Previously predicted action chunk, used by server-side RTC (real-time chunking) to blend across chunk boundaries. |

### RTC variants

Two real-time-chunking flavors are implemented and selected per
checkpoint at load time (see `rldx/model/modules/action_model/rtc.py`):

- **Inference-time RTC** (Black et al. 2025b): no training change.
  At each denoising step the flow-matching velocity is augmented with a
  guidance term that pulls the new chunk toward the still-overlapping
  positions of the previous one, hard-fixing the first `d` already-
  executed entries. Selected via `rtc_inference_mode != "none"` on the
  checkpoint config.
- **Training-time RTC** (Black et al. 2025c): a delay
  `d ~ U{0, …, rtc_training_max_delay}` is sampled at each training
  step and the model is conditioned on a clean prefix of length `d`,
  so inference reduces to the standard denoising loop with no
  guidance term and no extra latency. Selected when the loaded
  checkpoint has `rtc_training_max_delay > 0`.

Inference-time RTC adds latency (extra forward passes per denoising
step), training-time RTC does not. If both are configured on the same
checkpoint, the training-time path takes precedence because the model
was already trained for clean-prefix conditioning.

## Two canonical deployments

### Simulator eval

`run_rldx_server.py` + `rollout_policy.py` pair. See
[`evaluation.md`](evaluation.md) for the full recipe. Key points:

- The rollout worker runs in a separate simulator-specific venv that
  does not import torch/flash-attn at all.
- `--use-sim-policy-wrapper` must be on for the observation/action
  reshape.
- Bind to `127.0.0.1` since both processes live on the same box.

### Real-robot (DROID, Allex, ...)

`run_scripts/deploy/droid_deploy.sh` shows the DROID pattern:

```bash
uv run python rldx/eval/run_rldx_server.py \
    --model-path /path/to/your/checkpoint \
    --embodiment-tag GENERAL_EMBODIMENT \
    # drop --use-sim-policy-wrapper below when the client is a real robot
    --use-sim-policy-wrapper \
    --host 0.0.0.0 --port 5555
```

On the robot side, wire the DROID driver into a `PolicyClient`
(ZMQ REQ over TCP) and step at whatever rate the controller can
sustain. The server's action chunk is 16 steps at the training
timestep dt; the client decides how many to execute before asking for
the next chunk (`execution_horizon`).

Caveats for real-robot deployment:

- **Memory resets.** For episodic tasks you must pass
  `options["reset_memory"] = [True]` on the first step of every
  episode, otherwise the memory cache from the previous episode will
  bleed into the new one.
- **Session ids.** Use a stable unique string per controller; if you
  regenerate it every query the memory cache will be empty every
  step.
- **Latency.** The default 4 denoising steps + MSAT forward is
  ~15-30 ms on an A100 for the 8B model. Chunks of 16 steps mean you
  can amortise that over 16 controller ticks; keep
  `execution_horizon ≤ 16` to stay inside a single chunk.

## Embedded vs server use

If your client lives in the same Python process (so no wire protocol
is needed), you can skip `run_rldx_server.py` entirely and use
`RLDXPolicy` directly:

```python
import rldx
from rldx.policy.rldx_policy import RLDXPolicy

policy = RLDXPolicy(
    model_path="RLWRLD/RLDX-1-FT-ROBOCASA",
    embodiment_tag=rldx.EmbodimentTag.GENERAL_EMBODIMENT,
    device="cuda",
)
actions, _ = policy.get_action(observations, options={"session_ids": ["s0"], "reset_memory": [True]})
```

The server and the embedded path go through the exact same
`RLDXPolicy.get_action` code path, just with or without the ZMQ
round-trip.

## Troubleshooting

### `Connection refused` from the client

The server takes 30-60 s to load the backbone on a warm cache, much
longer on cold. Poll until the server prints the final `[MSAT]` init
line before connecting, or wrap your client startup in a retry loop.

### `Unrecognized processing class`

Checkpoint processor files are not inside a `processor/` subdir. Fix
with `scripts/patch_checkpoint.py` or re-upload the checkpoint with
the current layout — see
[`evaluation.md`](evaluation.md#troubleshooting) and the
`CheckpointFormatCallback` section of [`training.md`](training.md#checkpoint-format).

### Memory cache not resetting between episodes

Either `session_ids` is being regenerated every query (so every step
looks like a new session to the server and the cache is never hit),
or `reset_memory` is never set. The server logs each unique session id
and whether its memory cache was hit — enable `--verbose` on the
server to see them.

### Action chunks look random at inference

Most likely the processor is running in `train` mode and applying
random crops. `RLDXPolicy` calls `processor.eval()` on load, so this
only happens if you are using `RLDXPolicy` directly and accidentally
called `processor.train()` later in your code.

## Where to next

- [`installation.md`](installation.md) — env setup
- [`evaluation.md`](evaluation.md) — simulator benchmark recipes
- [`training.md`](training.md) — produce a checkpoint to serve
- [`architecture.md`](architecture.md) — model internals
