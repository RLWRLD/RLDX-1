"""Migrate RLWRLD HF checkpoints to the Issue #70 naming convention.

The script downloads the *inference-only* subset of files from each source
repo, applies ``scripts/patch_checkpoint.py`` to migrate legacy config /
state_dict field names to the current schema, then re-uploads to the new
``RLWRLD/`` repo as **private**. Source repos are left untouched — verify
the new private repos load correctly, then delete or archive the old ones
manually.

Patch step (mandatory, non-dry-run only):
  After ``snapshot_download`` but before ``upload_folder``, all
  ``patch_checkpoint.py`` handlers are applied in order:
    field_renames_v1, embodiment_rename, moss_enable_to_use_moss,
    yaml_rename, metaquery_to_cog_token, mmdit_to_msat, moss_to_motion,
    action_head_to_action_model, contextvla_to_vtc,
    drop_block_causal_masking, drop_rope_complex_buffers,
    drop_legacy_image_pipeline_kwargs, physics_flat_to_container,
    vlm_backbone_reference, legacy_qwen3_vl_backbone.
  All handlers are idempotent (sentinel-gated), so re-running on an
  already-patched checkpoint (e.g. production 4) is safe.
  If patch application raises an exception, the upload is aborted.

Naming map covered (8 active mappings; 2 deferred to a follow-up issue):
  Production (4 repos):
    RLWRLD/RLDX-1-8B          -> RLWRLD/RLDX-1-PT
    RLWRLD/RLDX-1-vlm         -> RLWRLD/RLDX-1-VLM
    RLWRLD/RLDX-1-ft-robocasa -> RLWRLD/RLDX-1-FT-ROBOCASA
    RLWRLD/RLDX-1-mt-allex    -> RLWRLD/RLDX-1-MT-ALLEX

  Sandbox finetunes from rldx-benchmark.csv (4 repos):
    kimtaey/g0_8kpt_ft_libero_bsz64_60k                                       -> RLWRLD/RLDX-1-FT-LIBERO
    huiwon/g0_v1d_simpler_google_bsz1024_20k_sd05                             -> RLWRLD/RLDX-1-FT-SIMPLER-GOOGLE
    huiwon/g0_v1d_simpler_widowx_bsz1024_60k                                  -> RLWRLD/RLDX-1-FT-SIMPLER-WIDOWX
    glory-hyeok/g0_v1d4d1v_mixv5_bs8192_100k_ft_gr1_1000demos_sd05            -> RLWRLD/RLDX-1-FT-GR1
        (CSV references the checkpoint-60000 subdir; the mapping pins it via the `subdir=` arg)

  Deferred to follow-up issue (tracked separately; not in MAPPINGS below):
    kimtaey/g0_midtrain_droid_physicsfixed_bsz1024_ac16_step25000             -> RLWRLD/RLDX-1-MT-DROID
        (already published once during PR #71; re-upload from clean source paused
         pending owner sign-off on regenerated patch + DROID modality config drift)
    happyhappy-jun/rldx_g0_ft_rc365_human300_bs192_250k_16gpu                 -> RLWRLD/RLDX-1-FT-RC365
        (source private; dry-run 404 without owner token)

Usage::

    # 1. Set up auth (worker node only — never on login node)
    export HF_TOKEN=<your token with RLWRLD org write access>

    # 2. Dry run first to inspect file lists per repo (patch NOT applied):
    python scripts/migrate_hf_naming.py --dry-run

    # 3. Migrate one repo (downloads, patches, uploads):
    python scripts/migrate_hf_naming.py --only RLWRLD/RLDX-1-FT-LIBERO

    # 4. Migrate everything:
    python scripts/migrate_hf_naming.py --all

    # 5. Verify the new repo loads (after success):
    python -c "import rldx; from transformers import AutoConfig; \
               print(AutoConfig.from_pretrained('RLWRLD/RLDX-1-PT'))"

The "inference-only" file list is the union of:
  - top-level ``config.json`` (model config)
  - sharded weight files: ``model-*.safetensors``, ``model.safetensors.index.json``,
    ``pytorch_model-*.bin`` (legacy), ``pytorch_model.bin.index.json``
  - ``processor/**`` subdir (RLDX layout) OR top-level ``processor_config.json``
    + ``embodiment_id.json`` + ``statistics.json`` (legacy flat layout)
  - tokenizer files inherited from the backbone: ``tokenizer*.json``,
    ``vocab.json``, ``merges.txt``, ``special_tokens_map.json``

Excluded by default:
  - optimizer state: ``optimizer.pt``, ``scheduler.pt``, ``trainer_state.json``
  - DeepSpeed ZeRO shards: ``global_step*/``, ``*_optim_states.pt``, ``*_model_states.pt``
  - logs: ``train.log``, ``wandb_config.json``, ``runs/``, ``logs/``
  - tensorboard event files: ``events.out.tfevents.*``

Override the include / exclude pattern lists at the top of this file if a
specific source repo needs something extra.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError as e:
    raise SystemExit(
        "huggingface_hub is required. Install with: uv pip install huggingface_hub"
    ) from e

# Import patch_checkpoint handlers for in-process patching.
# We locate the script relative to this file so it works from any cwd.
_PATCH_SCRIPT = Path(__file__).parent / "patch_checkpoint.py"
if not _PATCH_SCRIPT.exists():
    raise SystemExit(f"patch_checkpoint.py not found at {_PATCH_SCRIPT}")

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("patch_checkpoint", _PATCH_SCRIPT)
_patch_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_patch_mod)

# Convenience aliases for the public patch functions and sentinel helpers.


def _apply_all_patches(ckpt_path: Path) -> None:
    """Run all patch_checkpoint handlers against *ckpt_path* in order.

    Delegates the entire sequence to ``patch_checkpoint.apply_all_patches``
    — same single source of truth as the standalone CLI. Raises on any
    unexpected exception so the caller can abort the upload.
    """
    total_changes = _patch_mod.apply_all_patches(ckpt_path)
    if total_changes:
        print(f"  patch: {total_changes} file(s)/blob(s) modified.")
    else:
        print("  patch: no changes needed (checkpoint already compatible).")


INCLUDE_PATTERNS: list[str] = [
    "config.json",
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model-*.bin",
    "pytorch_model.bin.index.json",
    "processor/**",
    "processor_config.json",
    "embodiment_id.json",
    "statistics.json",
    "tokenizer*.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "generation_config.json",
    "README.md",
    "*.txt",  # often LICENSE / NOTICE / license info
]

EXCLUDE_PATTERNS: list[str] = [
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "global_step*/**",
    "*_optim_states.pt",
    "*_model_states.pt",
    "runs/**",
    "logs/**",
    "events.out.tfevents.*",
    "wandb_config.json",
    "train.log",
    "experiment_cfg/**",   # training metadata; not used at inference
    "rng_state_*.pth",
    "latest",
    "zero_to_fp32.py",
]


@dataclass(frozen=True)
class RepoMapping:
    source: str
    target: str
    subdir: str | None = None
    note: str = ""


# (source, target, optional source subdir, note)
MAPPINGS: list[RepoMapping] = [
    # Production 4
    RepoMapping("RLWRLD/RLDX-1-8B",          "RLWRLD/RLDX-1-PT",            note="pretrained base"),
    RepoMapping("RLWRLD/RLDX-1-vlm",         "RLWRLD/RLDX-1-VLM",           note="VLM mirror"),
    RepoMapping("RLWRLD/RLDX-1-ft-robocasa", "RLWRLD/RLDX-1-FT-ROBOCASA",   note="RoboCasa finetune"),
    RepoMapping("RLWRLD/RLDX-1-mt-allex",    "RLWRLD/RLDX-1-MT-ALLEX",      note="ALLEX midtrain"),
    # Sandbox finetunes from rldx-benchmark.csv (4 repos)
    RepoMapping("kimtaey/g0_8kpt_ft_libero_bsz64_60k",
                "RLWRLD/RLDX-1-FT-LIBERO",
                note="LIBERO finetune (97.40%)"),
    RepoMapping("huiwon/g0_v1d_simpler_google_bsz1024_20k_sd05",
                "RLWRLD/RLDX-1-FT-SIMPLER-GOOGLE",
                note="SIMPLER Google-VM/VA shared finetune"),
    RepoMapping("huiwon/g0_v1d_simpler_widowx_bsz1024_60k",
                "RLWRLD/RLDX-1-FT-SIMPLER-WIDOWX",
                note="SIMPLER WidowX finetune (71.90%)"),
    RepoMapping("glory-hyeok/g0_v1d4d1v_mixv5_bs8192_100k_ft_gr1_1000demos_sd05",
                "RLWRLD/RLDX-1-FT-GR1",
                subdir="checkpoint-60000",
                note="GR-1 Tabletop finetune (58.70%) — CSV pins checkpoint-60000"),
    # MT-DROID (kimtaey/...) and RC365 (happyhappy-jun/...) deferred to a
    # separate follow-up issue. MT-DROID was published once during PR #71 but
    # is paused pending owner sign-off on patch regeneration; RC365's source
    # is private and 404s without the owning token. See the follow-up issue
    # tracked in the docstring above.
]


def find(args_target: str) -> list[RepoMapping]:
    if args_target == "all":
        return list(MAPPINGS)
    matches = [m for m in MAPPINGS if args_target in (m.source, m.target)]
    if not matches:
        raise SystemExit(f"No mapping matches {args_target!r}")
    return matches


def migrate_one(api: HfApi, m: RepoMapping, dry_run: bool, keep_local: bool) -> None:
    print(f"\n=== {m.source} -> {m.target} ===")
    if m.note:
        print(f"  note: {m.note}")
    if m.subdir:
        print(f"  source subdir: {m.subdir}")

    src_files = api.list_repo_files(m.source)
    if m.subdir:
        prefix = m.subdir.rstrip("/") + "/"
        src_files = [f[len(prefix):] for f in src_files if f.startswith(prefix)]

    print(f"  source has {len(src_files)} files")

    # Decide which files to download. INCLUDE_PATTERNS uses fnmatch-style
    # globs via huggingface_hub.snapshot_download(allow_patterns=...).
    # Showing the projected file list helps catch surprises in dry-run.
    import fnmatch

    def keep(f: str) -> bool:
        if any(fnmatch.fnmatch(f, p) for p in EXCLUDE_PATTERNS):
            return False
        return any(fnmatch.fnmatch(f, p) for p in INCLUDE_PATTERNS)

    keep_list = [f for f in src_files if keep(f)]
    skip_list = [f for f in src_files if not keep(f)]
    print(f"  -> would upload {len(keep_list)} files, skip {len(skip_list)}")
    if dry_run:
        print("  --- keep ---")
        for f in keep_list:
            print(f"    {f}")
        print("  --- skip ---")
        for f in skip_list:
            print(f"    {f}")
        return

    # Real run.
    workdir = tempfile.mkdtemp(prefix=f"hf-migrate-{m.target.replace('/', '_')}-")
    print(f"  download dir: {workdir}")
    # When the source pins a subdir (e.g. checkpoint-60000/...), the keep
    # patterns must be prefixed with that subdir or fnmatch will not match
    # any of the source files (each repo file path is "subdir/<base>").
    if m.subdir:
        prefix = m.subdir.rstrip("/") + "/"
        allow = [prefix + p for p in INCLUDE_PATTERNS]
        ignore = [prefix + p for p in EXCLUDE_PATTERNS]
    else:
        allow = list(INCLUDE_PATTERNS)
        ignore = list(EXCLUDE_PATTERNS)
    snapshot_download(
        repo_id=m.source,
        local_dir=workdir,
        allow_patterns=allow,
        ignore_patterns=ignore,
    )
    upload_root = os.path.join(workdir, m.subdir) if m.subdir else workdir

    # Apply all patch_checkpoint handlers before upload so that
    # legacy field names (model_type=AlinVLAv0, backbone=contextvla_qwen3_vl,
    # etc.) are rewritten to their current schema equivalents on disk.
    # Handlers are idempotent (sentinel-gated), so this is safe even for
    # production checkpoints that are already patched.
    print(f"  applying patch_checkpoint handlers to {upload_root} ...")
    try:
        _apply_all_patches(Path(upload_root))
    except Exception as exc:
        raise RuntimeError(
            f"patch_checkpoint failed for {m.source} -> {m.target}: {exc}"
        ) from exc

    # Reset the target repo so the resulting HF history is a single
    # "RLDX-1 Release" commit (no linear chain of migration / re-upload
    # / fix commits). delete_repo is idempotent — missing_ok=True covers
    # first-time creation.
    try:
        api.delete_repo(repo_id=m.target, missing_ok=True)
    except Exception as exc:
        raise SystemExit(f"delete_repo({m.target}) failed: {exc}") from exc
    try:
        api.create_repo(repo_id=m.target, private=True, exist_ok=False)
    except Exception as exc:
        raise SystemExit(f"create_repo({m.target}) failed: {exc}") from exc

    print(f"  uploading {upload_root} -> {m.target} (private, fresh repo)")
    api.upload_folder(
        folder_path=upload_root,
        repo_id=m.target,
        commit_message="RLDX-1 Release",
    )
    print(f"  done. https://huggingface.co/{m.target}")
    if not keep_local:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="Migrate every mapping")
    g.add_argument("--only", metavar="REPO", help="Migrate one mapping (match source or target)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print include / exclude file lists per repo, do not download or upload")
    p.add_argument("--keep-local", action="store_true",
                   help="Do not delete the temp download dir after upload")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token) if token else HfApi()
    if not args.dry_run:
        try:
            who = api.whoami()
            print(f"Authenticated as {who.get('name', '<unknown>')}")
        except Exception as exc:
            raise SystemExit(
                "HF authentication missing. Set HF_TOKEN or run "
                "`huggingface-cli login` (token must have RLWRLD org write access). "
                f"whoami() error: {exc}"
            ) from exc

    targets = find("all" if args.all else args.only)
    print(f"Will process {len(targets)} mapping(s); dry_run={args.dry_run}")
    for m in targets:
        try:
            migrate_one(api, m, dry_run=args.dry_run, keep_local=args.keep_local)
        except Exception as exc:
            print(f"  ERROR on {m.source} -> {m.target}: {exc}", file=sys.stderr)
            if not args.all:
                raise

    return 0


if __name__ == "__main__":
    sys.exit(main())
