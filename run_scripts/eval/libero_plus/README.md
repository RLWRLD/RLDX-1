# LIBERO-Plus

Robustness benchmark built on LIBERO. Evaluates a single LIBERO checkpoint
across 10,300 perturbation tasks (object layout, camera viewpoint, robot
init, language, light, background, sensor noise).

| Field | Value |
|---|---|
| Embodiment tag | `GENERAL_EMBODIMENT` |
| HuggingFace checkpoint | [`RLWRLD/RLDX-1-FT-LIBERO`](https://huggingface.co/RLWRLD/RLDX-1-FT-LIBERO) (same as LIBERO) |
| Reported success rate | 84.3 % |
| Simulator venv | `rldx/eval/sim/LIBERO_PLUS/libero_plus_uv/.venv` |
| Source | [Sylvest/LIBERO-plus](https://huggingface.co/datasets/Sylvest/LIBERO-plus) |

## 1. Setup (one-time)

```bash
bash run_scripts/eval/libero_plus/setup_libero_plus.sh
```

Builds the isolated venv and downloads the 6.4 GB perturbation asset zip
from the HuggingFace dataset repo.

## 2. Finetune

LIBERO-Plus reuses the LIBERO checkpoint — see the LIBERO finetune recipe
at [`../libero/README.md`](../libero/README.md).

## 3. Run evaluation

```bash
LIBERO_PLUS_DATA_DIR=/path/to/libero_plus \
bash run_scripts/eval/libero_plus/eval_libero_plus.sh RLWRLD/RLDX-1-FT-LIBERO
```

Optional second argument restricts to a single LIBERO suite (e.g.
`libero_10`). Outputs land in
`output_final/libero_plus/<ckpt>/<suite>/<task>/`.
