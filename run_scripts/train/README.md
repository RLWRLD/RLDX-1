# RLDX Training Scripts

All scripts launch via `uv run torchrun ... rldx/experiment/launch_train.py`.
The default `BASE_MODEL_PATH` for fine-tune scripts is
`RLWRLD/RLDX-1-PT`; override with `BASE_MODEL_PATH=<path>` to start from a
different base.

## Directory layout

| Directory | What's inside |
|---|---|
| [`pretrain/`](pretrain/) | Video pretrain on `rldx_mix_v4` / OXE — produces the `RLWRLD/RLDX-1-PT` base. |
| [`midtrain/`](midtrain/) | Mid-train recipes that turn `RLDX-1-PT` into full-add-on (memory + motion + physics) checkpoints `RLWRLD/RLDX-1-MT-ALLEX` / `RLDX-1-MT-DROID`. |
| [`benchmarks/`](benchmarks/) | One script per benchmark — the recipes that produced each released `RLWRLD/RLDX-1-FT-*` checkpoint (LIBERO, GR-1, RoboCasa, RoboCasa365, SIMPLER). |

## Per-benchmark fine-tune

The benchmark guides under [`run_scripts/eval/<bench>/README.md`](../eval/)
each link to the matching script in `benchmarks/`. As a quick
reference:

| Benchmark | Script |
|---|---|
| LIBERO | `benchmarks/finetune_libero_bs256.sh` |
| GR-1 Tabletop | `benchmarks/finetune_gr1_tabletop_bs1024_sd05.sh` |
| RoboCasa Kitchen (300 demo) | `benchmarks/finetune_rldx1_robocasa_kitchen300_lr4_bs1024.sh` |
| RoboCasa Kitchen (1000 demo) | `benchmarks/finetune_rldx1_robocasa_kitchen1000_lr4_bs1024.sh` |
| RoboCasa365 | `benchmarks/finetune_rldx1_robocasa_365.sh` |
| SIMPLER Google | `benchmarks/finetune_simpler_google_bs1024_sd05.sh` |
| SIMPLER WidowX | `benchmarks/finetune_simpler_widowx_bs1024.sh` |
