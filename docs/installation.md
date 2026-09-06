# Installation

This guide walks through a fresh install of RLDX-1 on a Linux machine with
an NVIDIA GPU. The short version is at the top; subsequent sections cover
simulator environments, developer tooling, and common pitfalls.

## TL;DR

There are two install paths. Pick the one that matches your GPU:

| Path | Tool |
|---|---|
| [Standard install](#standard-install) | `uv` |
| [SM_120 (RTX 5090 / Blackwell)](#rtx-5090--blackwell-sm_120) | `pixi` |

```bash
# Standard install
git clone https://github.com/RLWRLD/RLDX-1.git
cd RLDX-1
uv sync --python 3.10
uv pip install -e .
python -c "import rldx; print(rldx.__version__)"
```

```bash
# RTX 5090 / Blackwell (SM_120) — flash-attn must be built from source
git clone https://github.com/RLWRLD/RLDX-1.git
cd RLDX-1
pixi install
pixi run --environment rldx postinstall
pixi run --environment rldx python -c "import rldx; print(rldx.__version__)"
```

Printing the version only checks package discovery; it does not load model
dependencies. Run the model registry check below to verify the model installation.
Simulator eval stacks install separately
(see [Simulator environments](#simulator-environments)).

## Prerequisites

| Requirement | Version | Why |
|---|---|---|
| Linux | x86_64 | The model has only been exercised on Linux. |
| Python | `3.10.*` (pinned) | pyproject.toml constrains `requires-python`. |
| CUDA toolkit | 12.x | Needed to build `flash-attn==2.7.4.post1`. |
| NVIDIA driver | supports CUDA 12 | For `torch==2.7.0` + `flash-attn`. |
| [uv](https://github.com/astral-sh/uv) | `>= 0.8.4` | Dependency + virtualenv manager (Standard path). |
| [pixi](https://pixi.sh) | `>= 0.40` | Conda-backed env manager (SM_120 path only). |
| `git` / `git-lfs` | recent | Submodules + HF LFS assets. |

Install `uv` if you do not already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install `pixi` only if you are on the SM_120 path:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

## Standard install

Default path. flash-attn ships prebuilt wheels and `uv sync` installs
them directly without compilation. Use the
[SM_120 path](#rtx-5090--blackwell-sm_120) instead if you're on
RTX 5090 / Blackwell.

```bash
git clone https://github.com/RLWRLD/RLDX-1.git
cd RLDX-1
uv sync --python 3.10          # creates .venv and resolves deps
uv pip install -e .            # installs the rldx package editable
```

`uv sync` materialises the environment described by `pyproject.toml`,
including the `[tool.uv.extra-build-dependencies]` override that pins
`torch==2.7.0 + numpy==1.26.4` at build time for `flash-attn`. The
resulting virtualenv lives at `.venv/` and is activated automatically
by every `uv run` call.

### Verify

```bash
# Python import smoke test — should print the installed rldx version (no ImportError)
uv run python -c "import rldx; print(rldx.__version__)"

# HuggingFace registry smoke test — should print "<class 'rldx.model.core.processing_rldx.RLDXProcessor'>"
uv run python -c "
import rldx.model
from transformers.models.auto.processing_auto import PROCESSOR_MAPPING
print(PROCESSOR_MAPPING._extra_content[rldx.RLDXConfig])
"
```

### Model-free simulator client

The native ZMQ client can be used from a simulator runtime without installing
the full model stack. Keep the model server in the Python 3.10 environment
above. This source-based client path has been tested with Isaac's Python 3.12;
it does **not** make the full distribution compatible with Python 3.12.

Clone this repository and check out a reviewed, pinned commit containing the
model-free package initialization change. Until that change is merged, stock
`main` is not sufficient. Do not replace an existing model checkout.

```bash
git clone https://github.com/RLWRLD/RLDX-1.git RLDX-1-client
cd RLDX-1-client
git checkout --detach <reviewed-client-commit>
# Set this to the Python launcher supplied by your simulator.
SIM_PYTHON=/path/to/simulator/python.sh
"$SIM_PYTHON" -c 'import numpy, torch, msgpack, zmq'
RLDX_CLIENT_SRC="$(pwd)"
PYTHONPATH="$RLDX_CLIENT_SRC${PYTHONPATH:+:$PYTHONPATH}" "$SIM_PYTHON" -c \
  'from rldx.policy.server_client import PolicyClient; print(PolicyClient.__module__)'
```

Use that same command-scoped `PYTHONPATH` when launching your simulator's
policy runner. It exposes the pinned native source; it does not patch imports,
copy the transport, or change global shell settings. The client still needs
NumPy, PyTorch, msgpack and pyzmq. Use the simulator's existing compatible
packages; do not install the full RLDX dependency stack over its PyTorch/CUDA
runtime. If dependencies are missing, provision them in an isolated runtime
and repeat the import and native round-trip tests before deployment.

`pip install --no-deps -e .` is not a workaround for `requires-python`:
the full package still declares Python 3.10. A separately installable client
wheel is not supplied by this change.

### Import compatibility

`import rldx` now only discovers the package and its lazy public symbols.
Code that relied on this import alone registering Hugging Face Auto classes
must explicitly use `import rldx.model` before `AutoConfig`, `AutoModel` or
`AutoProcessor` loading. Accessing model symbols still requires the full model
environment. Native server entrypoints import their model implementations;
the wire protocol and inference math are unchanged. Validate custom HF loaders
and the full model/server environment before adopting this change there.

## RTX 5090 / Blackwell (SM_120)

flash-attn upstream does not publish prebuilt wheels for `sm_120`. The
Standard path's `uv sync` therefore fails on RTX 5090 with a missing
wheel error. The pixi path solves this by providing CUDA 12.8 toolkit
through the `nvidia` conda channel and compiling flash-attn from source
with `TORCH_CUDA_ARCH_LIST=sm_120` baked into the env activation.

```bash
git clone https://github.com/RLWRLD/RLDX-1.git
cd RLDX-1

# Resolve and install the pixi-managed conda + PyPI environment.
pixi install

# Run the chained postinstall task — builds flash-attn from source for
# sm_120 and editable-installs the rldx package against the pixi torch.
pixi run --environment rldx postinstall
```

The `postinstall` task chains `install-flash-attn` (source build) before
`pip install -e . --no-deps` so a single command bootstraps the entire
env. `TORCH_CUDA_ARCH_LIST=sm_120` is set both in
`[feature.rldx.tasks.install-flash-attn]` and in
`[feature.rldx.activation]` so any subsequent `pixi run` keeps the same
arch target.

### Verify (pixi)

```bash
# Python import smoke test — should print the installed rldx version (no ImportError)
pixi run --environment rldx python -c "import rldx; print(rldx.__version__)"

# flash-attn was built against the pixi torch and targets sm_120
pixi run --environment rldx python -c "
import torch
from flash_attn import flash_attn_func
print(torch.__version__, torch.cuda.get_device_capability())
"
```

### Notes

- `pixi.toml` pins **torch 2.8.0 + cu128** (vs the uv path's `torch 2.7.0`).
  The two environments are deliberately separate; do not mix them.
- The first `pixi run --environment rldx postinstall` takes 10-20 minutes
  because flash-attn compiles from source. Subsequent runs reuse the cached
  build.
- If you are not on Blackwell, prefer the Standard install — pixi will
  work but pulls a larger conda payload (CUDA 12.8 toolkit) than uv needs.

## Simulator environments

Simulator eval stacks are maintained as git submodules under
`external_dependencies/`. They are optional — you only need the ones
that match the benchmarks you want to run — and most of them have their
own Python environments to avoid dependency conflicts with the main
training env.

```bash
# Pull the submodules you need
git submodule update --init external_dependencies/robocasa
git submodule update --init external_dependencies/LIBERO
git submodule update --init external_dependencies/SimplerEnv
```

### RoboCasa

```bash
bash rldx/eval/sim/robocasa/setup_RoboCasa.sh
```

The setup script creates a separate `robocasa_uv/.venv` under
`rldx/eval/sim/robocasa/` and installs robocasa, robosuite, and mujoco
into that venv. The robocasa eval scripts (for example
`run_scripts/eval/robocasa_kitchen/eval_robocasa.sh`) explicitly invoke
that venv's Python so the main training env is untouched.

### LIBERO

```bash
bash rldx/eval/sim/LIBERO/setup_libero.sh
```

Separate venv at `rldx/eval/sim/LIBERO/libero_uv/.venv`.

### SimplerEnv

```bash
bash rldx/eval/sim/SimplerEnv/setup_SimplerEnv.sh
```

### GR00T whole-body control (BEHAVIOR / robocasa365)

```bash
bash rldx/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh
```

Requires `git-lfs` and pulls a large amount of assets. The section
and script keep their upstream `GR00T-WholeBodyControl` name because
this is an unmodified external dependency that RLDX-1 only consumes at eval time.

## Developer install

For anyone editing the code rather than only consuming it, also install
the optional `dev` extra and set up the pre-commit hooks:

```bash
uv pip install -e ".[dev]"
uv tool install ruff
uv tool install pre-commit
pre-commit install
```

The pre-commit config (`.pre-commit-config.yaml`) runs the same
`ruff check` and `ruff format --check` steps that CI enforces, scoped to
`rldx/`. Run them against the whole tree ad-hoc with:

```bash
pre-commit run --all-files
```

## Common pitfalls

### `flash-attn` build fails with "nvcc not found"

The wheel is compiled on first install and needs the CUDA toolkit,
not just a matching driver. Make sure `nvcc --version` works on the
machine, and that the version matches the version `torch` was built
against. On RTX 5090 (Blackwell) follow the
[SM_120 install](#rtx-5090--blackwell-sm_120) — the pixi env ships
the CUDA 12.8 toolkit so this error cannot occur.

### `flash-attn` takes 10-20 minutes to install

Expected on first install: the wheel is compiled from source against
your local torch build. Subsequent installs reuse the cached wheel.
On RTX 5090 the SM_120 source build always runs (no prebuilt wheel
exists upstream), so plan for the 10-20 minute first install.

### `flash-attn` import fails with "no kernel image is available for execution on the device" on RTX 5090

The default `uv sync` pulled the prebuilt `flash-attn` wheel which only
ships kernels for SM_80 and SM_90. On Blackwell that wheel imports but
crashes at the first kernel launch. Switch to the
[SM_120 install](#rtx-5090--blackwell-sm_120) (pixi) — it builds
flash-attn from source with `TORCH_CUDA_ARCH_LIST=sm_120`.

### `ImportError: libcuda.so.1: cannot open shared object file`

The driver is missing or the `LD_LIBRARY_PATH` does not point at the
NVIDIA userspace libraries. On a typical Ubuntu install the fix is to
add `/usr/lib/x86_64-linux-gnu` to `LD_LIBRARY_PATH`, or to invoke the
Python interpreter inside an NVIDIA container.

### `HF_HUB_OFFLINE=1` + first-time model load

Several code paths download the Qwen3-VL backbone or the reference
RLDX-1 checkpoint from the Hugging Face Hub on first use. Either run the
initial pull online, or pre-download the checkpoint to
`$HF_HOME/hub/models--...` before switching to offline mode.

### `Could not find the Transformers classes you have set` at processor load

The RLDX Hugging Face `AutoProcessor` registry is populated by explicitly
importing `rldx.model` in the model environment. Package-only `import rldx`
intentionally leaves the model stack unloaded. Custom HF loading code should
use `import rldx.model` before `AutoProcessor.from_pretrained(...)`.

## Where to next

- [`training.md`](training.md) — run fine-tuning and mid-training
- [`evaluation.md`](evaluation.md) — run LIBERO / RoboCasa / SIMPLER / GR-1 evals
- [`inference_server.md`](inference_server.md) — serve a checkpoint (ZeroMQ canonical / WebSocket alt)
- [`architecture.md`](architecture.md) — high-level model walkthrough
