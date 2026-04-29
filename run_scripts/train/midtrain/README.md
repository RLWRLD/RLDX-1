# RLDX-1 All-add-on recipes

Mid-train and fine-tune scripts that build on `RLWRLD/RLDX-1-PT` and add
**all** functional capabilities — memory, motion, physics — on top of the
video backbone.

| Script | Purpose |
|---|---|
| `midtrain_rldx1_droid.sh` | Mid-train on the DROID dataset to produce `RLWRLD/RLDX-1-MT-DROID`. |
| `midtrain_rldx1_allex.sh` | Mid-train on ALLEX humanoid data to produce `RLWRLD/RLDX-1-MT-ALLEX`. |
| `finetune_rldx1_all_droid.sh` | Direct fine-tune on DROID with all add-ons enabled (memory + motion + physics) — alternative path that skips a separate mid-train stage. |

These scripts add the following CLI flags relative to the video-only
recipes (see [`../pretrain/README.md`](../pretrain/README.md)):

```bash
--use-memory --memory-length 4 --memory-stride 16 --concat-memory --memory-n-cog-tokens 16
--use-motion --motion-insert-layer 9
--use-physics --physics-keys torque --physics-dims 48 --allow-missing-physics
--new-param-warmup-steps 2000
```

Default `BASE_MODEL_PATH` is `RLWRLD/RLDX-1-PT`.
