# RLDX-1 Video-only recipes

Pretrain and fine-tune scripts for the **video-only** stage of RLDX-1.
This is the recipe that produced `RLWRLD/RLDX-1-PT` and the base for all
RoboCasa fine-tunes.

| Script | Purpose |
|---|---|
| `pretrain_rldx1_mixv4_multinode.sh` | Full pretrain on the `rldx_mix_v4` mixture (Droid + OXE + Galaxea + AgiBot + GR-1, etc.). Multi-node distributed launch. |
| `pretrain_rldx1_oxe_multinode.sh` | OXE-only pretrain variant. |
| `finetune_rldx1_robocasa_kitchen.sh` | Fine-tune on RoboCasa Kitchen 300-demo dataset. |
| `finetune_rldx1_robocasa_365.sh` | Fine-tune on RoboCasa365 multitask. |

All scripts default to `BASE_MODEL_PATH=RLWRLD/RLDX-1-PT` (or the raw VLM
backbone for pretraining). The video module is enabled by default through
`--video-length 4` — there is no separate `--use-video` flag.
