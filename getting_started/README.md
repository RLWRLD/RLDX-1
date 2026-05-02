# Getting Started with RLDX

A guided tour of the RLDX codebase via five notebooks. Read them in
order the first time — each one assumes the previous.

| # | Notebook | Covers |
|---|---|---|
| 1 | [`01_quickstart.ipynb`](01_quickstart.ipynb) | Install check, load a pretrained model, run a single inference step on a dummy observation. |
| 2 | [`02_dataset_preparation.ipynb`](02_dataset_preparation.ipynb) | LeRobot dataset layout, modality configs, embodiment registration, what the processor expects. |
| 3 | [`03_finetuning.ipynb`](03_finetuning.ipynb) | `launch_train.py` CLI, core flags, checkpoint layout, a one-step validation run. |
| 4 | [`04_inference_server.ipynb`](04_inference_server.ipynb) | Starting the inference server, connecting a client, and the sim-policy wrapper. |
| 5 | [`05_module_guide.ipynb`](05_module_guide.ipynb) | Opt-in modules: video, memory module, motion module, physics, cognition tokens. Flags, data requirements, composition matrix. |

## Prerequisites

- A working install per [`docs/installation.md`](../docs/installation.md):
  ```bash
  uv sync --python 3.10 && uv pip install -e .
  ```
- A CUDA-capable GPU for notebooks 1, 3, and 4. Notebook 2 is CPU-only.
- ~30 GB of free disk for the pretrained-checkpoint cache.
- Optional: a LeRobot-format dataset on local disk for the training /
  dataset notebooks.

## Running the notebooks

Jupyter is not declared as a dev dependency of `rldx`. Install it into
the existing `uv` environment:

```bash
uv pip install jupyterlab
uv run jupyter lab getting_started/
```

Or open individual notebooks in VS Code / Cursor and select
`.venv/bin/python` as the interpreter.

## Placeholders

All notebooks refer to the public pretrained checkpoint as
`YOUR_MODEL_PLACEHOLDER`. Replace it with the HuggingFace repo id or
local path of the checkpoint you want to load. The official release
identifier will be wired in at release time — search for `TODO(release)`
in this directory.
