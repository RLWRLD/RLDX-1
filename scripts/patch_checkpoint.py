"""Patch RLDX checkpoint configs + state_dict for compatibility.

Each refactor phase adds an idempotent handler that migrates older
checkpoints to the current code's schema. Handlers are tracked in a
per-checkpoint `.rldx_patched` JSON sentinel so re-running on an
already-migrated checkpoint is a no-op and cannot corrupt state.

Usage:
    python scripts/patch_checkpoint.py /path/to/checkpoint
    python scripts/patch_checkpoint.py huggingface/model-id  # patches cached snapshot

Handlers (in execution order):
    1. field_renames_v1          — config.json model_type / architectures / processor_config.processor_class
    2. embodiment_rename         — NEW_EMBODIMENT ↔ SIMULATION_GR1 (non-idempotent; sentinel-gated)
    3. moss_enable_to_use_moss   — `moss_enable` config field → `use_moss`
    4. yaml_rename               — experiment_cfg/*.yaml mirrors of the above
    5. metaquery_to_cog_token    — Phase 2: n_meta_queries / memory_n_meta_queries /
                                   use_meta_queries / meta_queries_mode / freeze_meta_queries
                                   config fields, `"meta_only"` string value,
                                   `*.meta_emb` → `*.cog_emb` state_dict keys +
                                   safetensors weight_map.
    6. mmdit_to_msat             — Phase 3: use_mmditv0 / use_mmditv1 / mmditv0_config /
                                   mmditv1_config config fields. State_dict keys are
                                   unchanged (MMDiT lives under
                                   `action_head.model.*`, not `action_head.mmdit.*`).
    7. moss_to_motion            — Phase 4: every `moss_*` config field →
                                   `motion_*`, `use_moss → use_motion`, state_dict
                                   prefix `*.moss_block.* → *.motion_block.*` +
                                   safetensors weight_map. Atomic with the rename
                                   of the contextvla probe literal `"moss_block"`
                                   to `"motion_block"` in the same commit.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys


# ============================================================================
# Static patch tables
# ============================================================================

PATCHES = {
    "config.json": {
        # Legacy class names AND the older "RLDX" string (pre-0296e80) all
        # map to the current runtime value "RLDX-1".
        "model_type": {
            "AlinVLAv0": "RLDX-1",
            "Gr00tN1d6": "RLDX-1",
            "RLDX": "RLDX-1",
        },
        # `architectures` is a list in HF configs; each element is mapped
        # independently. `AlinVLAv0Memory` was collapsed into the base
        # `RLDX` class in the refactor — memory is now a runtime flag
        # (`use_memory`) on `RLDXConfig`, not a separate subclass.
        "architectures": {
            "AlinVLAv0": "RLDX",
            "AlinVLAv0Memory": "RLDX",
            "Gr00tN1d6": "RLDX",
        },
    },
    "processor_config.json": {
        "processor_class": {
            "AlinVLAv0Processor": "RLDXProcessor",
            "Gr00tN1d6Processor": "RLDXProcessor",
        },
    },
}


# Basenames of JSON files inside a checkpoint that may reference embodiment
# tags (either as dict keys, string values, or inside processor kwargs). The
# text swap below rewrites every occurrence in these files.
EMBODIMENT_RENAME_TARGETS = {
    "config.json",
    "processor_config.json",
    "final_processor_config.json",
    "final_model_config.json",
    "dataset_statistics.json",
    "statistics.json",
    "embodiment_id.json",
    "metadata.json",
}


YAML_RENAME_TARGETS = {"config.yaml", "conf.yaml"}


# Sentinel file name + schema version written after a successful patch run.
PATCH_SENTINEL = ".rldx_patched"
SENTINEL_SCHEMA_VERSION = 2


# ============================================================================
# Embodiment rename + moss_enable rename helpers
# ============================================================================

def embodiment_text_swap(text: str) -> str:
    """Rename swap: NEW_EMBODIMENT ↔ SIMULATION_GR1 (and string values).

    After the rename:
        - pre-swap NEW_EMBODIMENT / "new_embodiment"   → GENERAL_EMBODIMENT / "general_embodiment"
        - pre-swap SIMULATION_GR1 / "simulation_gr1"   → NEW_EMBODIMENT     / "new_embodiment"

    This also rewrites processor kwargs whose name contains the string
    (e.g. "new_embodiment_train_ratio" → "general_embodiment_train_ratio")
    because those match `new_embodiment` as a substring.

    Uses a three-pass temp-token scheme per case (upper / lower) so that the
    two substitutions do not clobber each other.
    """
    # Upper-case (enum names, rarely in JSON but safe)
    text = text.replace("NEW_EMBODIMENT", "__TMP_NEWEMB_U__")
    text = text.replace("SIMULATION_GR1", "NEW_EMBODIMENT")
    text = text.replace("__TMP_NEWEMB_U__", "GENERAL_EMBODIMENT")
    # Lower-case (enum values / dict keys / CLI strings)
    text = text.replace("new_embodiment", "__tmp_newemb__")
    text = text.replace("simulation_gr1", "new_embodiment")
    text = text.replace("__tmp_newemb__", "general_embodiment")
    return text


def patch_embodiment_rename(ckpt_path: Path) -> int:
    """Walk checkpoint dir and rewrite every known JSON file's embodiment refs.

    Returns number of files actually changed. Unknown basenames are skipped
    to avoid touching unrelated JSON artifacts (e.g. transformer tokenizer files).

    Idempotence guard: if a target file contains no pre-swap tokens
    (`SIMULATION_GR1` / `simulation_gr1`) the swap is skipped. Rerunning the
    3-step temp-token swap on an already-migrated file shifts the
    `new_embodiment` key to `general_embodiment`, producing a duplicate JSON
    key that silently shadows the real `general_embodiment` on load — which
    corrupts embodiment-conditioned normalization in downstream inference.
    See `feedback_patch_checkpoint_safety.md`.
    """
    import json as _json

    changed_count = 0
    for fp in ckpt_path.rglob("*.json"):
        if fp.name not in EMBODIMENT_RENAME_TARGETS:
            continue
        src = fp.read_text()
        if "SIMULATION_GR1" not in src and "simulation_gr1" not in src:
            # Already post-swap — skip to preserve idempotence.
            continue
        dst = embodiment_text_swap(src)
        if src == dst:
            continue
        # Validate that the rewritten text is still valid JSON before writing —
        # string-level swap can corrupt a file only if our token leaked in.
        try:
            _json.loads(dst)
        except _json.JSONDecodeError as e:
            print(f"  [error] {fp.relative_to(ckpt_path)}: rename produced invalid JSON ({e})")
            continue
        fp.write_text(dst)
        print(f"  [embodiment-rename] {fp.relative_to(ckpt_path)}")
        changed_count += 1
    return changed_count


def patch_embodiment_rename_v2(ckpt_path: Path) -> int:
    """Drift-safe NEW_EMBODIMENT → GENERAL_EMBODIMENT rename.

    Catches the case where a source checkpoint has the lowercase
    ``new_embodiment`` key in ``processor_config.json``'s
    ``modality_configs`` dict but no ``simulation_gr1`` token —
    ``patch_embodiment_rename`` skips it as "already post-swap" because of
    its older guard, leaving the alias under the legacy name and breaking
    GENERAL_EMBODIMENT inference.

    Idempotent: only swaps if the file has ``new_embodiment`` AND no
    ``general_embodiment`` (so re-running on a v1-patched ckpt is a no-op).
    Skips ``simulation_gr1`` entirely — that swap is done by the v1 handler.
    """
    import json as _json

    changed_count = 0
    for fp in ckpt_path.rglob("*.json"):
        if fp.name not in EMBODIMENT_RENAME_TARGETS:
            continue
        src = fp.read_text()
        if "new_embodiment" not in src:
            continue
        if "general_embodiment" in src:
            # Already has the post-swap key — leave alone to avoid duplicate keys.
            continue
        dst = src.replace("new_embodiment", "general_embodiment")
        try:
            _json.loads(dst)
        except _json.JSONDecodeError as e:
            print(f"  [error] {fp.relative_to(ckpt_path)}: rename produced invalid JSON ({e})")
            continue
        fp.write_text(dst)
        print(f"  [embodiment-rename-v2] {fp.relative_to(ckpt_path)}")
        changed_count += 1
    return changed_count


def patch_moss_enable_to_use_moss(ckpt_path: Path) -> int:
    """Rename the model-config flag `moss_enable` → `use_moss` in every known
    JSON file. The CLI flag was always `--use-moss`; the model-config field
    was the only outlier (named `moss_enable` for historical reasons). Now
    aligned with `use_video` / `use_memory` / `use_physics`.

    Substring-safe: `moss_enable` appears only as a JSON key/value in our
    config files; no other field name contains it. JSON validity is checked
    before writing, same as the embodiment swap.
    """
    import json as _json

    changed_count = 0
    for fp in ckpt_path.rglob("*.json"):
        if fp.name not in EMBODIMENT_RENAME_TARGETS:
            continue
        src = fp.read_text()
        if "moss_enable" not in src:
            continue
        dst = src.replace("moss_enable", "use_moss")
        try:
            _json.loads(dst)
        except _json.JSONDecodeError as e:
            print(f"  [error] {fp.relative_to(ckpt_path)}: rename produced invalid JSON ({e})")
            continue
        fp.write_text(dst)
        print(f"  [moss-rename] {fp.relative_to(ckpt_path)}")
        changed_count += 1
    return changed_count


def patch_moss_enable_to_use_motion(ckpt_path: Path) -> int:
    """One-step shortcut: legacy `moss_enable` → final `use_motion`.

    Equivalent to running ``patch_moss_enable_to_use_moss`` followed by
    ``patch_moss_to_motion``, but applied as a single substring rewrite so the
    intermediate ``use_moss`` form never lands on disk. Used by checkpoint-
    migration tests that pin the end-state name directly.
    """
    import json as _json

    changed_count = 0
    for fp in ckpt_path.rglob("*.json"):
        if fp.name not in EMBODIMENT_RENAME_TARGETS:
            continue
        src = fp.read_text()
        if "moss_enable" not in src:
            continue
        dst = src.replace("moss_enable", "use_motion")
        try:
            _json.loads(dst)
        except _json.JSONDecodeError as e:
            print(f"  [error] {fp.relative_to(ckpt_path)}: rename produced invalid JSON ({e})")
            continue
        fp.write_text(dst)
        print(f"  [moss-enable-to-use-motion] {fp.relative_to(ckpt_path)}")
        changed_count += 1
    return changed_count


def patch_yaml_files(ckpt_path: Path) -> int:
    """Rewrite embodiment + moss field names in `experiment_cfg/*.yaml`.

    `experiment.py:run` saves both `experiment_cfg/config.yaml`
    (`Config.save`) and `experiment_cfg/conf.yaml` (OmegaConf-resolved).
    `launch_train._load_yaml_config` loads `experiment_cfg/config.yaml`
    as the primary source whenever a user resumes / fine-tunes on top of
    a base checkpoint, so a YAML that still says
    `embodiment_tag: simulation_gr1` (or carries the old `moss_enable`
    field) silently overrides any migration we did on `config.json`.

    The walk is scoped to YAMLs whose parent directory is named
    `experiment_cfg/` so we never touch stray YAMLs that happen to live
    next to a checkpoint (e.g. user-saved logs).

    Returns number of files actually changed.
    """
    changed_count = 0
    for fp in ckpt_path.rglob("*.yaml"):
        if fp.name not in YAML_RENAME_TARGETS:
            continue
        if fp.parent.name != "experiment_cfg":
            continue
        src = fp.read_text()
        dst = embodiment_text_swap(src)
        dst = dst.replace("moss_enable", "use_moss")
        if src == dst:
            continue
        fp.write_text(dst)
        print(f"  [yaml-rename] {fp.relative_to(ckpt_path)}")
        changed_count += 1
    return changed_count


# ============================================================================
# Phase 2 — metaquery → cog_token
# ============================================================================

# Field-level and string-value renames for config.json / YAML. Order matters
# — the longest pattern comes first to avoid collisions between
# `memory_n_meta_queries` and `n_meta_queries`.
_METAQUERY_FIELD_MAP: list[tuple[str, str]] = [
    ("memory_n_meta_queries", "memory_n_cog_tokens"),
    ("n_meta_queries",        "n_cog_tokens"),
    ("use_meta_queries",      "use_cog_tokens"),
    ("meta_queries_mode",     "cog_mode"),
    ("freeze_meta_queries",   "freeze_cog_tokens"),
    # String value — only "meta_only" transitions; "full" is unchanged
    ('"meta_only"',           '"cog_only"'),
]

# safetensors state_dict key prefix/substring rename. We anchor on the `.meta_emb`
# parameter suffix (always under a module name like `backbone.`), so a plain
# substring match is safe — no other tensor key contains `meta_emb`.
_METAQUERY_TENSOR_MAP: list[tuple[str, str]] = [
    (".meta_emb", ".cog_emb"),
]


def rename_safetensors_keys(
    ckpt_path: Path,
    rename_rules: list[tuple[str, str]],
) -> int:
    """Rename tensor keys inside every `*.safetensors` shard + update the
    `model.safetensors.index.json::weight_map` in lockstep.

    Args:
        ckpt_path: directory containing one or more `model*.safetensors`
            shards and (for sharded checkpoints) a `model.safetensors.index.json`.
        rename_rules: ordered list of (old, new) substring pairs applied to
            each tensor key in order. First match wins per key, so put the
            longest / most specific pattern first.

    Returns number of unique tensor keys renamed (summed across shards).
    """
    try:
        from safetensors.torch import safe_open, save_file
    except ImportError:
        print("  [warn] safetensors not installed — skipping tensor-key rename")
        return 0

    shards = sorted(ckpt_path.glob("model*.safetensors"))
    if not shards:
        return 0

    def _map_key(k: str) -> str:
        for old, new in rename_rules:
            if old in k:
                return k.replace(old, new)
        return k

    rename_count = 0
    new_weight_map: dict[str, str] = {}

    for shard_path in shards:
        tensors: dict = {}
        with safe_open(shard_path, framework="pt") as f:
            metadata = f.metadata() or {}
            for k in f.keys():
                new_k = _map_key(k)
                if new_k != k:
                    rename_count += 1
                tensors[new_k] = f.get_tensor(k)
        # Re-serialize shard only if something changed in this shard, otherwise
        # leave the file alone (preserves mtime + shard bit-identity).
        if any(_map_key(k) != k for k in tensors.keys()) or rename_count:
            save_file(tensors, str(shard_path), metadata=metadata)
        for new_k in tensors.keys():
            new_weight_map[new_k] = shard_path.name

    index_path = ckpt_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        old_map = index.get("weight_map", {})
        # Rebuild weight_map in the exact ordering of the physical shards'
        # enumeration above so index and blobs stay consistent.
        if old_map != new_weight_map:
            index["weight_map"] = new_weight_map
            with open(index_path, "w") as f:
                json.dump(index, f, indent=2)
            print(f"  [safetensors-index] weight_map rewritten ({len(new_weight_map)} keys)")

    return rename_count


def patch_metaquery_to_cog_token(ckpt_path: Path) -> int:
    """Issue #29 Phase 2 rename:
        config.json / yaml field names:
            n_meta_queries        → n_cog_tokens
            memory_n_meta_queries → memory_n_cog_tokens
            use_meta_queries      → use_cog_tokens
            meta_queries_mode     → cog_mode
            freeze_meta_queries   → freeze_cog_tokens
            "meta_only"           → "cog_only"
        state_dict (safetensors shards + weight_map):
            *.meta_emb → *.cog_emb

    Substring-safe because every target pattern is unambiguous inside our
    JSON / YAML / tensor-key namespace. Longest pattern is applied first
    (see `_METAQUERY_FIELD_MAP`).
    """
    import json as _json

    total_changes = 0

    # 1. JSON config files
    for fp in ckpt_path.rglob("*.json"):
        if fp.name not in EMBODIMENT_RENAME_TARGETS:
            continue
        src = fp.read_text()
        dst = src
        for old, new in _METAQUERY_FIELD_MAP:
            dst = dst.replace(old, new)
        if src == dst:
            continue
        try:
            _json.loads(dst)
        except _json.JSONDecodeError as e:
            print(f"  [error] {fp.relative_to(ckpt_path)}: metaquery rename produced invalid JSON ({e})")
            continue
        fp.write_text(dst)
        print(f"  [metaquery→cog_token] {fp.relative_to(ckpt_path)}")
        total_changes += 1

    # 2. YAML files (experiment_cfg/*.yaml)
    for fp in ckpt_path.rglob("*.yaml"):
        if fp.name not in YAML_RENAME_TARGETS:
            continue
        if fp.parent.name != "experiment_cfg":
            continue
        src = fp.read_text()
        dst = src
        for old, new in _METAQUERY_FIELD_MAP:
            dst = dst.replace(old, new)
        # YAML may quote string values or not; the unquoted form needs its
        # own pass.
        dst = dst.replace(" meta_only", " cog_only")
        if src == dst:
            continue
        fp.write_text(dst)
        print(f"  [metaquery-yaml] {fp.relative_to(ckpt_path)}")
        total_changes += 1

    # 3. safetensors state_dict keys (+ weight_map)
    tensor_renames = rename_safetensors_keys(ckpt_path, _METAQUERY_TENSOR_MAP)
    if tensor_renames:
        print(f"  [metaquery-tensors] {tensor_renames} state_dict key(s) renamed")
        total_changes += 1

    return total_changes


# ============================================================================
# Phase 3 — MMDiT → MSAT
# ============================================================================

# config.json / YAML field + string-value renames. Order: longest first.
_MMDIT_FIELD_MAP: list[tuple[str, str]] = [
    # Config field names
    ("use_mmditv1",    "use_msat"),
    ("use_mmditv0",    "use_msat_v0"),
    ("mmditv1_config", "msat_config"),
    ("mmditv0_config", "msat_v0_config"),
    # `mmdit_v1` appears as a registry string in run_scripts/deploy/droid_deploy.sh
    # — not in checkpoint artifacts — so no JSON rewrite is needed for it.
]


def patch_mmdit_to_msat(ckpt_path: Path) -> int:
    """Issue #29 Phase 3 rename:
        config.json / yaml field names:
            use_mmditv0    → use_msat_v0
            use_mmditv1    → use_msat
            mmditv0_config → msat_v0_config
            mmditv1_config → msat_config

    State_dict keys are NOT renamed in this phase: the `MMDiT` class lives
    inside `RLDXActionHead` as the attribute `self.model`, not `self.mmdit`,
    so state_dict prefixes (`action_head.model.*`) are unaffected by the
    class rename. Config-only migration.
    """
    import json as _json

    total_changes = 0

    # 1. JSON config files (config.json / processor_config.json / etc.)
    for fp in ckpt_path.rglob("*.json"):
        if fp.name not in EMBODIMENT_RENAME_TARGETS:
            continue
        src = fp.read_text()
        dst = src
        for old, new in _MMDIT_FIELD_MAP:
            dst = dst.replace(old, new)
        if src == dst:
            continue
        try:
            _json.loads(dst)
        except _json.JSONDecodeError as e:
            print(f"  [error] {fp.relative_to(ckpt_path)}: mmdit rename produced invalid JSON ({e})")
            continue
        fp.write_text(dst)
        print(f"  [mmdit→msat] {fp.relative_to(ckpt_path)}")
        total_changes += 1

    # 2. YAML files (experiment_cfg/{config,conf}.yaml)
    for fp in ckpt_path.rglob("*.yaml"):
        if fp.name not in YAML_RENAME_TARGETS:
            continue
        if fp.parent.name != "experiment_cfg":
            continue
        src = fp.read_text()
        dst = src
        for old, new in _MMDIT_FIELD_MAP:
            dst = dst.replace(old, new)
        if src == dst:
            continue
        fp.write_text(dst)
        print(f"  [mmdit-yaml] {fp.relative_to(ckpt_path)}")
        total_changes += 1

    return total_changes


# ============================================================================
# Phase 4 — moss → motion
# ============================================================================

_MOSS_FIELD_MAP: list[tuple[str, str]] = [
    # Config fields — longest first
    ("moss_injection_point",   "motion_injection_point"),
    ("moss_gradient_check",    "motion_gradient_check"),
    ("moss_use_layerscale",    "motion_use_layerscale"),
    ("moss_layerscale_init",   "motion_layerscale_init"),
    ("moss_use_layernorm",     "motion_use_layernorm"),
    ("moss_use_syncbn",        "motion_use_syncbn"),
    ("moss_insert_layer",      "motion_insert_layer"),
    ("moss_n_encoders",        "motion_n_encoders"),
    ("moss_ext_chnls",         "motion_ext_chnls"),
    ("moss_int_chnls",         "motion_int_chnls"),
    ("moss_corr_func",         "motion_corr_func"),
    ("moss_pool_type",         "motion_pool_type"),
    ("moss_int_mode",          "motion_int_mode"),
    ("moss_window",            "motion_window"),
    ("moss_config",            "motion_config"),
    ("moss_d_hid",             "motion_d_hid"),
    ("moss_drop",              "motion_drop"),
    ("use_moss",               "use_motion"),
    # Avoid touching the deprecated `moss_enable` legacy field; handler #3
    # already migrated that to `use_moss` before this one runs, and this pass
    # will catch the `use_moss → use_motion` step above.
]

_MOSS_TENSOR_MAP: list[tuple[str, str]] = [
    # state_dict keys — contextvla backbone parameters live under `moss_block.*`
    (".moss_block.", ".motion_block."),
]


def patch_moss_to_motion(ckpt_path: Path) -> int:
    """Issue #29 Phase 4 rename:
        config.json / yaml field names: every `moss_*` → `motion_*`
        state_dict keys (safetensors shards + index.json::weight_map):
            *.moss_block.* → *.motion_block.*

    The contextvla probe literal `"moss_block"` in
    `rldx/model/modules/contextvla/modeling_contextvla.py` was renamed to
    `"motion_block"` in the same commit as the state_dict rename, so
    patching an old checkpoint must update the tensor keys + weight_map
    to match — otherwise the probe would fail and trigger a
    `MotionModule.initialize_weights()` call that silently overwrites
    trained motion-module weights.
    """
    import json as _json

    total_changes = 0

    # 1. JSON config files
    for fp in ckpt_path.rglob("*.json"):
        if fp.name not in EMBODIMENT_RENAME_TARGETS:
            continue
        src = fp.read_text()
        dst = src
        for old, new in _MOSS_FIELD_MAP:
            dst = dst.replace(old, new)
        if src == dst:
            continue
        try:
            _json.loads(dst)
        except _json.JSONDecodeError as e:
            print(f"  [error] {fp.relative_to(ckpt_path)}: moss rename produced invalid JSON ({e})")
            continue
        fp.write_text(dst)
        print(f"  [moss→motion] {fp.relative_to(ckpt_path)}")
        total_changes += 1

    # 2. YAML files (experiment_cfg/{config,conf}.yaml)
    for fp in ckpt_path.rglob("*.yaml"):
        if fp.name not in YAML_RENAME_TARGETS:
            continue
        if fp.parent.name != "experiment_cfg":
            continue
        src = fp.read_text()
        dst = src
        for old, new in _MOSS_FIELD_MAP:
            dst = dst.replace(old, new)
        if src == dst:
            continue
        fp.write_text(dst)
        print(f"  [moss-yaml] {fp.relative_to(ckpt_path)}")
        total_changes += 1

    # 3. safetensors state_dict keys + weight_map
    tensor_renames = rename_safetensors_keys(ckpt_path, _MOSS_TENSOR_MAP)
    if tensor_renames:
        print(f"  [moss-tensors] {tensor_renames} state_dict key(s) renamed")
        total_changes += 1

    return total_changes


# ============================================================================
# Phase 5 — action_head → action_model
# ============================================================================

_ACTION_HEAD_FIELD_MAP: list[tuple[str, str]] = [
    # Longest / most specific first so `action_head_max_seq_len` and
    # `action_head_config` swap into their full forms before the bare
    # `action_head` substring is hit.
    ("action_head_max_seq_len", "action_model_max_seq_len"),
    ("action_head_config",      "action_model_config"),
    ("action_head",             "action_model"),
]

_ACTION_HEAD_TENSOR_MAP: list[tuple[str, str]] = [
    # state_dict keys: top-level `action_head.*` prefix. No other tensor key
    # contains the substring `action_head` in any known checkpoint, so a
    # substring rewrite is safe.
    ("action_head.", "action_model."),
]


def patch_action_head_to_action_model(ckpt_path: Path) -> int:
    """Issue #29 Phase 5 rename (pure — no structural changes):
        config.json / yaml field names:
            action_head_max_seq_len → action_model_max_seq_len
                (appears inside the nested `diffusion_model_cfg` JSON dict,
                not at the top level; text-level swap still works because
                the key is unambiguous.)
            action_head_config      → action_model_config
            action_head_*           → action_model_*   (anything else)
        state_dict keys (safetensors shards + index.json::weight_map):
            action_head.* → action_model.*

    The `self.action_decoder` attribute is NOT renamed — it lives under
    `action_model.action_decoder.*` already and the guide carves it out
    explicitly.
    """
    import json as _json

    total_changes = 0

    # 1. JSON config files
    for fp in ckpt_path.rglob("*.json"):
        if fp.name not in EMBODIMENT_RENAME_TARGETS:
            continue
        src = fp.read_text()
        dst = src
        for old, new in _ACTION_HEAD_FIELD_MAP:
            dst = dst.replace(old, new)
        if src == dst:
            continue
        try:
            _json.loads(dst)
        except _json.JSONDecodeError as e:
            print(f"  [error] {fp.relative_to(ckpt_path)}: action_head rename produced invalid JSON ({e})")
            continue
        fp.write_text(dst)
        print(f"  [action_head→action_model] {fp.relative_to(ckpt_path)}")
        total_changes += 1

    # 2. YAML files (experiment_cfg/{config,conf}.yaml)
    for fp in ckpt_path.rglob("*.yaml"):
        if fp.name not in YAML_RENAME_TARGETS:
            continue
        if fp.parent.name != "experiment_cfg":
            continue
        src = fp.read_text()
        dst = src
        for old, new in _ACTION_HEAD_FIELD_MAP:
            dst = dst.replace(old, new)
        if src == dst:
            continue
        fp.write_text(dst)
        print(f"  [action_head-yaml] {fp.relative_to(ckpt_path)}")
        total_changes += 1

    # 3. safetensors state_dict keys + weight_map
    tensor_renames = rename_safetensors_keys(ckpt_path, _ACTION_HEAD_TENSOR_MAP)
    if tensor_renames:
        print(f"  [action_head-tensors] {tensor_renames} state_dict key(s) renamed")
        total_changes += 1

    return total_changes


# ============================================================================
# Phase 6 — contextvla → vtc (video token compression)
# ============================================================================

_CONTEXTVLA_FIELD_MAP: list[tuple[str, str]] = [
    # backbone_model_type string value in config.json
    ("contextvla_qwen3_vl", "vtc_qwen3_vl"),
    # Defensive: any bare mention in config.json / processor_config.json
    ("contextvla_qwen3",    "vtc_qwen3"),
    ("ContextVLA",          "VTC"),
    ("contextvla",          "vtc"),
]


def patch_contextvla_to_vtc(ckpt_path: Path) -> int:
    """Issue #29 Phase 6 rename:
        config.json / yaml field and value renames:
            backbone_model_type: "contextvla_qwen3_vl" → "vtc_qwen3_vl"
            "ContextVLA" / "contextvla" references in config files → "VTC" / "vtc"

    The contextvla classes are Python-level names that were never written
    into safetensors state_dict keys (tensors sit under `backbone.*`, not
    `contextvla.*`), so no state_dict migration is required for this phase.
    """
    import json as _json

    total_changes = 0

    for fp in ckpt_path.rglob("*.json"):
        if fp.name not in EMBODIMENT_RENAME_TARGETS:
            continue
        src = fp.read_text()
        dst = src
        for old, new in _CONTEXTVLA_FIELD_MAP:
            dst = dst.replace(old, new)
        if src == dst:
            continue
        try:
            _json.loads(dst)
        except _json.JSONDecodeError as e:
            print(f"  [error] {fp.relative_to(ckpt_path)}: contextvla rename produced invalid JSON ({e})")
            continue
        fp.write_text(dst)
        print(f"  [contextvla→vtc] {fp.relative_to(ckpt_path)}")
        total_changes += 1

    for fp in ckpt_path.rglob("*.yaml"):
        if fp.name not in YAML_RENAME_TARGETS:
            continue
        if fp.parent.name != "experiment_cfg":
            continue
        src = fp.read_text()
        dst = src
        for old, new in _CONTEXTVLA_FIELD_MAP:
            dst = dst.replace(old, new)
        if src == dst:
            continue
        fp.write_text(dst)
        print(f"  [contextvla-yaml] {fp.relative_to(ckpt_path)}")
        total_changes += 1

    return total_changes


def patch_legacy_qwen3_vl_backbone(ckpt_path: Path) -> int:
    """Rewrite `backbone_model_type: "qwen3_vl"` → `"vtc_qwen3_vl"` in
    config.json / processor_config.json.

    Context: Issue #36 removed the vanilla Qwen3 backbone path
    (`Qwen3Backbone` / `BACKBONE_REGISTRY`). Release-era code only
    constructs `VTCQwen3VLBackbone` and raises if
    `backbone_model_type != "vtc_qwen3_vl"`. Legacy checkpoints that
    still carry `"qwen3_vl"` would trip that guard at load time AND
    (more insidiously) would land as the processor's `model_type`,
    mis-routing the VTC-specific collator branches (`"vtc" in
    self.model_type`) back to the non-VTC path. Rewrite on disk so
    load + downstream routing stays correct.

    Exact key-value match: only the `backbone_model_type` key's value
    gets rewritten. `contextvla_qwen3_vl` is caught upstream by
    `patch_contextvla_to_vtc`; a substring replace here would incorrectly
    mangle that value (`contextvla_qwen3_vl` contains `qwen3_vl`).

    Idempotent: re-runs find nothing to change.
    """
    import json as _json

    def _walk(obj) -> bool:
        """Recursively rewrite `backbone_model_type="qwen3_vl"` entries.
        Returns True if any change was made."""
        changed = False
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "backbone_model_type" and v == "qwen3_vl":
                    obj[k] = "vtc_qwen3_vl"
                    changed = True
                elif isinstance(v, (dict, list)):
                    if _walk(v):
                        changed = True
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    if _walk(item):
                        changed = True
        return changed

    total = 0
    for fname in ("config.json", "processor/processor_config.json"):
        fp = ckpt_path / fname
        if not fp.exists():
            continue
        try:
            cfg = _json.loads(fp.read_text())
        except _json.JSONDecodeError:
            continue
        if _walk(cfg):
            fp.write_text(_json.dumps(cfg, indent=2))
            print(f"  [legacy-qwen3-vl] {fname}: backbone_model_type qwen3_vl → vtc_qwen3_vl")
            total += 1
    return total


def patch_vlm_backbone_reference(ckpt_path: Path) -> int:
    """Rewrite legacy personal-namespace VLM backbone references to the
    RLWRLD release namespace.

    Production checkpoints serialize the backbone VLM's source repo
    under `model_name` (both in top-level `config.json` and inside
    `processor/processor_config.json`). Several ckpts carry
    `huiwon/alinvlm_v1_3` — which becomes dead on release since only
    the `RLWRLD/RLDX-1-VLM` mirror stays readable.

    Mapping (ordered longest-first so no partial collision):

        huiwon/alinvlm_v1_3      → RLWRLD/RLDX-1-VLM
        huiwon/alinvla           → RLWRLD/RLDX-1-VLM

    Idempotent: rerunning finds nothing to change.
    """
    import json as _json

    rules = [
        ("huiwon/alinvlm_v1_3", "RLWRLD/RLDX-1-VLM"),
        ("huiwon/alinvla", "RLWRLD/RLDX-1-VLM"),
    ]

    targets = [
        ckpt_path / "config.json",
        ckpt_path / "processor" / "processor_config.json",
    ]

    total = 0
    for fp in targets:
        if not fp.exists():
            continue
        src = fp.read_text()
        dst = src
        for old, new in rules:
            dst = dst.replace(old, new)
        if src == dst:
            continue
        try:
            _json.loads(dst)
        except _json.JSONDecodeError as e:
            print(
                f"  [error] {fp.relative_to(ckpt_path)}: VLM rename produced "
                f"invalid JSON ({e})"
            )
            continue
        fp.write_text(dst)
        print(f"  [vlm-ref] {fp.relative_to(ckpt_path)}")
        total += 1

    return total


_LEGACY_IMAGE_DROP_KEYS_JSON = (
    "image_crop_size",
    "image_target_size",
    "shortest_image_edge",
    "use_albumentations",
)
_LEGACY_IMAGE_DROP_KEYS_YAML = _LEGACY_IMAGE_DROP_KEYS_JSON + (
    # YAML mirror of the same field happens to use the ``_transforms``
    # suffix variant (e.g. observed in ``experiment_cfg/conf.yaml``).
    # Both spellings need to go.
    "use_albumentations_transforms",
)
_LEGACY_IMAGE_RENAMES = {
    "crop_fraction": "random_crop_fraction",
    # Pre-#22 image-pipeline names (still on simpler-google/widowx ckpts).
    "resize_max_area": "image_max_area",
    "resize_align_multiple": "image_resize_m",
}


def _migrate_legacy_image_kwargs_yaml(yaml_text: str) -> tuple[str, bool]:
    """Drop / rename legacy image-pipeline keys in raw YAML text.

    Strict line-based: every ``<indent><key>: <anything>`` line is
    matched. Each legacy key may appear multiple times in conf.yaml
    (model + data sections both mirror the schema), so all occurrences
    are processed. ``crop_fraction`` is renamed in place; the value
    and indentation are preserved verbatim. If ``random_crop_fraction``
    already exists somewhere in the file, the rename degrades to a
    drop to avoid producing duplicate keys (which would break YAML
    re-parse later).
    """
    import re

    new = yaml_text
    changed = False

    for key in _LEGACY_IMAGE_DROP_KEYS_YAML:
        pattern = re.compile(rf"^\s*{re.escape(key)}:.*\n?", re.MULTILINE)
        replaced = pattern.sub("", new)
        if replaced != new:
            changed = True
            new = replaced

    for old, target in _LEGACY_IMAGE_RENAMES.items():
        if re.search(rf"^\s*{re.escape(target)}:", new, re.MULTILINE):
            # Target key already present — drop legacy form, don't duplicate.
            pattern = re.compile(rf"^\s*{re.escape(old)}:.*\n?", re.MULTILINE)
        else:
            # Rename in place, preserving indent + value
            pattern = re.compile(rf"^(\s*){re.escape(old)}(:.*)$", re.MULTILINE)
            replaced = pattern.sub(rf"\1{target}\2", new)
            if replaced != new:
                changed = True
                new = replaced
            continue
        replaced = pattern.sub("", new)
        if replaced != new:
            changed = True
            new = replaced

    return new, changed


def patch_drop_dead_action_head_flags(ckpt_path: Path) -> int:
    """Drop legacy action-head selector flags that no current RLDXConfig
    field reads. The original ``use_mmditv0`` / ``use_mmditv1`` were a
    branch that picked between two action-head variants; the v0 path was
    dropped in the refactor and v1 (= MSAT) is the only path. The
    earlier ``patch_mmdit_to_msat`` renamed them to ``use_msat`` /
    ``use_msat_v0`` which are equally dead — RLDXConfig has neither
    field, so they sit as silent unknown kwargs. This handler removes
    all four spellings from JSON configs.

    Idempotent: when none of the keys are present the handler reports
    zero changes.
    """
    import json as _json

    DEAD = ("use_mmditv0", "use_mmditv1", "use_msat", "use_msat_v0")
    targets = [
        ckpt_path / "config.json",
        ckpt_path / "experiment_cfg" / "final_model_config.json",
    ]
    total = 0
    for fp in targets:
        if not fp.exists():
            continue
        try:
            data = _json.loads(fp.read_text())
        except _json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        dropped = [k for k in DEAD if k in data]
        if not dropped:
            continue
        for k in dropped:
            data.pop(k, None)
        fp.write_text(_json.dumps(data, indent=2))
        print(f"  [drop-dead-action-head] {fp.relative_to(ckpt_path)} -> dropped {dropped}")
        total += 1
    return total


def patch_drop_legacy_image_pipeline_kwargs(ckpt_path: Path) -> int:
    """Migrate ``processor_config.json`` (and yaml mirrors) from
    pre-#22/#36 image-pipeline kwargs to the current schema.

    The image-transformation refactor (Issues #22 / #36) removed or
    renamed five kwargs that older checkpoints still carry inside
    ``processor_kwargs`` (and the matching ``experiment_cfg/*.yaml``
    mirrors)::

        image_crop_size                → (removed)
        image_target_size              → (removed)
        shortest_image_edge            → (removed)
        use_albumentations             → (removed; now always True)
        use_albumentations_transforms  → (removed; YAML-only spelling)
        crop_fraction                  → random_crop_fraction
                                          (renamed; value preserved)

    Without this migration, ``RLDXProcessor.from_pretrained`` crashes
    immediately with ``TypeError: got an unexpected keyword argument
    'image_crop_size'`` on every legacy checkpoint
    (e.g. ``RLWRLD/RLDX-1-MT-ALLEX``). See Issue #46.

    Targets:
        * JSON: ``processor/processor_config.json`` (current layout),
          ``processor_config.json`` at root (pre-#22 flat layout),
          ``experiment_cfg/final_processor_config.json`` (training mirror).
        * YAML: ``experiment_cfg/conf.yaml`` and
          ``experiment_cfg/config.yaml``. These are the run-config
          mirror that ``_load_yaml_config`` reads when finetune resumes
          from a pretrained ckpt — a mismatch with the runtime Config
          dataclass produces dead fields at best, schema errors at
          worst. The rest of patch_checkpoint already follows the
          "JSON + YAML mirror" pattern (yaml_rename, action_head,
          contextvla, ...) — keep that convention.

    Idempotent: legacy keys are gone after the first run; a re-run
    finds nothing to change and reports zero modifications.
    """
    import json as _json

    json_targets = [
        ckpt_path / "processor" / "processor_config.json",
        ckpt_path / "processor_config.json",  # pre-#22 flat layout
        ckpt_path / "experiment_cfg" / "final_processor_config.json",
    ]
    yaml_targets = [
        ckpt_path / "experiment_cfg" / "conf.yaml",
        ckpt_path / "experiment_cfg" / "config.yaml",
    ]

    total = 0

    # ---- JSON ---------------------------------------------------------
    for fp in json_targets:
        if not fp.exists():
            continue
        try:
            data = _json.loads(fp.read_text())
        except _json.JSONDecodeError as e:
            print(f"  [error] {fp.relative_to(ckpt_path)}: invalid JSON ({e})")
            continue

        pkw = data.get("processor_kwargs")
        if not isinstance(pkw, dict):
            continue

        changed = False
        for k in _LEGACY_IMAGE_DROP_KEYS_JSON:
            if k in pkw:
                pkw.pop(k)
                changed = True
        for old, new in _LEGACY_IMAGE_RENAMES.items():
            if old in pkw:
                value = pkw.pop(old)
                # ``setdefault`` rather than overwrite — if a re-saved
                # checkpoint already wrote the new key, keep it.
                pkw.setdefault(new, value)
                changed = True

        if not changed:
            continue
        fp.write_text(_json.dumps(data, indent=2))
        print(f"  [drop-legacy-image-kwargs] {fp.relative_to(ckpt_path)}")
        total += 1

    # ---- YAML ---------------------------------------------------------
    for fp in yaml_targets:
        if not fp.exists():
            continue
        try:
            src = fp.read_text()
        except OSError as e:
            print(f"  [error] {fp.relative_to(ckpt_path)}: read failed ({e})")
            continue

        dst, changed = _migrate_legacy_image_kwargs_yaml(src)
        if not changed:
            continue
        fp.write_text(dst)
        print(f"  [drop-legacy-image-kwargs-yaml] {fp.relative_to(ckpt_path)}")
        total += 1

    return total


_EMBODIMENT_TRAIN_RATIO_RENAME = (
    "new_embodiment_train_ratio",
    "general_embodiment_train_ratio",
)


def patch_new_embodiment_train_ratio(ckpt_path: Path) -> int:
    """Rename ``new_embodiment_train_ratio`` → ``general_embodiment_train_ratio``.

    The training mixer field that controls how often a non-target
    embodiment is sampled was renamed (``rldx/configs/train_config.py``,
    ``rldx/configs/model/rldx.py``). The accompanying
    ``RLDXProcessor.__init__`` only accepts the new spelling and raises
    ``TypeError: got an unexpected keyword argument
    'new_embodiment_train_ratio'`` on every legacy checkpoint that still
    carries the old key inside ``processor_kwargs``.

    Targets:
        * JSON: ``config.json`` (top-level field on the model config),
          ``processor/processor_config.json::processor_kwargs``,
          ``processor_config.json::processor_kwargs`` (pre-#22 layout),
          ``experiment_cfg/final_model_config.json``,
          ``experiment_cfg/final_processor_config.json::processor_kwargs``.
        * YAML: ``experiment_cfg/conf.yaml`` and
          ``experiment_cfg/config.yaml`` — line-level rename matching
          the ``yaml_rename`` convention used elsewhere in this script.

    Idempotent: a re-run finds no remaining occurrences and reports
    zero changes.
    """
    import json as _json

    old, new = _EMBODIMENT_TRAIN_RATIO_RENAME

    json_top_level_targets = [
        ckpt_path / "config.json",
        ckpt_path / "experiment_cfg" / "final_model_config.json",
    ]
    json_processor_kwargs_targets = [
        ckpt_path / "processor" / "processor_config.json",
        ckpt_path / "processor_config.json",
        ckpt_path / "experiment_cfg" / "final_processor_config.json",
    ]
    yaml_targets = [
        ckpt_path / "experiment_cfg" / "conf.yaml",
        ckpt_path / "experiment_cfg" / "config.yaml",
    ]

    total = 0

    for fp in json_top_level_targets:
        if not fp.exists():
            continue
        try:
            data = _json.loads(fp.read_text())
        except _json.JSONDecodeError as e:
            print(f"  [error] {fp.relative_to(ckpt_path)}: invalid JSON ({e})")
            continue
        if old not in data:
            continue
        data.setdefault(new, data.pop(old))
        fp.write_text(_json.dumps(data, indent=2))
        print(f"  [embodiment-ratio] {fp.relative_to(ckpt_path)}")
        total += 1

    for fp in json_processor_kwargs_targets:
        if not fp.exists():
            continue
        try:
            data = _json.loads(fp.read_text())
        except _json.JSONDecodeError as e:
            print(f"  [error] {fp.relative_to(ckpt_path)}: invalid JSON ({e})")
            continue
        pkw = data.get("processor_kwargs")
        if not isinstance(pkw, dict) or old not in pkw:
            continue
        pkw.setdefault(new, pkw.pop(old))
        fp.write_text(_json.dumps(data, indent=2))
        print(f"  [embodiment-ratio] {fp.relative_to(ckpt_path)}")
        total += 1

    for fp in yaml_targets:
        if not fp.exists():
            continue
        src = fp.read_text()
        if old not in src:
            continue
        dst = src.replace(old, new)
        fp.write_text(dst)
        print(f"  [embodiment-ratio-yaml] {fp.relative_to(ckpt_path)}")
        total += 1

    return total


def patch_physics_flat_to_container(ckpt_path: Path) -> int:
    """Rewrite flat `action_model.physics_*` state_dict keys into the
    `action_model.physics.*` PhysicsHead container layout.

    Background: Gerald's commit `1c2c841e` extracted `PhysicsHead` as a
    container module under `RLDXActionModel.physics`, which shifted the
    serialized key names. The runtime helper `remap_physics_keys`
    (`rldx/model/core/physics_head.py:35`) was added to the **fallback**
    path of `PolicyLoader`; the primary `AutoModel.from_pretrained`
    path at `rldx/policy/policy_loader.py:183` loads the state_dict
    verbatim. For legacy checkpoints to survive the primary path, the
    rename has to happen on disk — which is this handler's job.

    Also covers `physics_mask_token`, which the runtime remap table
    misses.

    Idempotence: each rule's `old` prefix embeds the `action_model.`
    parent, and the new layout has `action_model.physics.physics_*`
    which does NOT contain `action_model.physics_` as a substring
    (the dot between `physics` and `physics_*` breaks the match), so
    `rename_safetensors_keys`' substring replace is safe to re-run.
    """
    rename_rules = [
        ("action_model.physics_encoder.",       "action_model.physics.physics_cond_encoder."),
        ("action_model.physics_cond_encoder.",  "action_model.physics.physics_cond_encoder."),
        ("action_model.physics_fut_encoder.",   "action_model.physics.physics_fut_encoder."),
        ("action_model.physics_decoder.",       "action_model.physics.physics_decoder."),
        ("action_model.physics_mask_token",     "action_model.physics.physics_mask_token"),
    ]
    n = rename_safetensors_keys(ckpt_path, rename_rules)
    if n > 0:
        print(
            f"  [physics-tensors] {n} state_dict key(s) remapped to "
            f"`action_model.physics.*` container layout"
        )
    return n


def patch_drop_rope_complex_buffers(ckpt_path: Path) -> int:
    """Drop complex64 RoPE `freqs_cis_*` buffers from safetensors shards.

    Background: `RoPEEmbedder1D.register_buffer("freqs_cis_i", ...)` used
    the default `persistent=True`, so the complex64 frequency table was
    serialized into each checkpoint's safetensors. The `transformers`
    loader's newer safetensors backend rejects complex dtypes with
    `ValueError: Cannot load safetensors of unknown dtype C64`.

    The runtime code has been updated to register these buffers with
    `persistent=False` — they are a deterministic function of
    (axis_dim, max_seq_len, theta) and cheap to rebuild at init time —
    so no weights are lost by stripping them from old checkpoints.

    Idempotent: re-runs find nothing to drop and return 0.
    """
    import json as _json

    from safetensors import safe_open
    from safetensors.torch import save_file

    def _is_rope_freqs(key: str) -> bool:
        return ".rope_embedder.freqs_cis_" in key

    idx_path = ckpt_path / "model.safetensors.index.json"
    single_path = ckpt_path / "model.safetensors"

    total_dropped = 0

    if idx_path.exists():
        idx = _json.loads(idx_path.read_text())
        weight_map = idx.get("weight_map", {})
        to_drop = [k for k in weight_map if _is_rope_freqs(k)]
        if not to_drop:
            return 0

        # Group drops by the shard file they belong to.
        by_shard: dict[str, list[str]] = {}
        for k in to_drop:
            by_shard.setdefault(weight_map[k], []).append(k)

        for shard_name, dropped_keys in by_shard.items():
            shard_fp = ckpt_path / shard_name
            kept: dict = {}
            with safe_open(str(shard_fp), framework="pt") as f:
                for k in f.keys():
                    if k in dropped_keys:
                        continue
                    kept[k] = f.get_tensor(k)
            save_file(kept, str(shard_fp), metadata={"format": "pt"})
            print(
                f"  [drop-rope] {shard_name}: removed {len(dropped_keys)} "
                f"complex RoPE buffer(s); shard now holds {len(kept)} tensor(s)"
            )

        # Update the weight_map and total_size metadata.
        for k in to_drop:
            del weight_map[k]
        if "metadata" in idx and "total_size" in idx["metadata"]:
            new_total = 0
            for shard_name in set(weight_map.values()):
                new_total += (ckpt_path / shard_name).stat().st_size
            idx["metadata"]["total_size"] = new_total
        idx_path.write_text(_json.dumps(idx, indent=2))
        print(
            f"  [safetensors-index] weight_map: dropped "
            f"{len(to_drop)} complex RoPE buffer entry(ies)"
        )
        total_dropped = len(to_drop)

    elif single_path.exists():
        kept: dict = {}
        dropped_keys: list[str] = []
        with safe_open(str(single_path), framework="pt") as f:
            for k in f.keys():
                if _is_rope_freqs(k):
                    dropped_keys.append(k)
                else:
                    kept[k] = f.get_tensor(k)
        if not dropped_keys:
            return 0
        save_file(kept, str(single_path), metadata={"format": "pt"})
        print(
            f"  [drop-rope] {single_path.name}: removed "
            f"{len(dropped_keys)} complex RoPE buffer(s)"
        )
        total_dropped = len(dropped_keys)

    return total_dropped


def patch_drop_block_causal_masking(ckpt_path: Path) -> int:
    """Drop `diffusion_model_cfg.use_block_causal_masking` from config.json.

    Background: an upstream feature branch added a `use_block_causal_masking`
    flag on MSAT/MMDiT that never landed on the release code path. Some
    production checkpoints (e.g. mt-allex) carry the field serialized under
    `diffusion_model_cfg`, which crashes `AutoModel.from_pretrained` with
    `TypeError: MSAT.__init__() got an unexpected keyword argument
    'use_block_causal_masking'`. Per release owner: the key is legacy and
    must not be present in release configs.

    The handler drops the field unconditionally. Idempotent.
    """
    import json as _json

    cfg_path = ckpt_path / "config.json"
    if not cfg_path.exists():
        return 0
    cfg = _json.loads(cfg_path.read_text())
    dmc = cfg.get("diffusion_model_cfg")
    if not isinstance(dmc, dict):
        return 0
    if "use_block_causal_masking" not in dmc:
        return 0
    value = dmc.pop("use_block_causal_masking")
    cfg_path.write_text(_json.dumps(cfg, indent=2))
    print(
        f"  [drop] diffusion_model_cfg.use_block_causal_masking "
        f"(was {value!r}, field not supported by release MSAT)"
    )
    return 1


def patch_model_type_underscore_to_hyphen(ckpt_path: Path) -> int:
    """Rename `model_type` value from ``"RLDX_1"`` (underscore) to
    ``"RLDX-1"`` (hyphen) in all checkpoint JSON + YAML artifacts.

    Targets:
        config.json, final_model_config.json,
        experiment_cfg/conf.yaml, experiment_cfg/config.yaml

    Idempotent: files already containing ``"RLDX-1"`` are skipped.
    """
    import json as _json

    json_targets = [
        ckpt_path / "config.json",
        ckpt_path / "final_model_config.json",
        ckpt_path / "experiment_cfg" / "final_model_config.json",
    ]
    yaml_targets = [
        ckpt_path / "experiment_cfg" / "conf.yaml",
        ckpt_path / "experiment_cfg" / "config.yaml",
    ]

    total = 0

    for fp in json_targets:
        if not fp.exists():
            continue
        src = fp.read_text()
        if "RLDX_1" not in src:
            continue
        dst = src.replace('"RLDX_1"', '"RLDX-1"')
        if src == dst:
            continue
        try:
            _json.loads(dst)
        except _json.JSONDecodeError as e:
            print(f"  [error] {fp.relative_to(ckpt_path)}: model_type hyphen rename produced invalid JSON ({e})")
            continue
        fp.write_text(dst)
        print(f"  [model_type RLDX_1→RLDX-1] {fp.relative_to(ckpt_path)}")
        total += 1

    for fp in yaml_targets:
        if not fp.exists():
            continue
        src = fp.read_text()
        if "RLDX_1" not in src:
            continue
        dst = src.replace("RLDX_1", "RLDX-1")
        if src == dst:
            continue
        fp.write_text(dst)
        print(f"  [model_type RLDX_1→RLDX-1 yaml] {fp.relative_to(ckpt_path)}")
        total += 1

    return total


def patch_vlm_backbone_reference_uppercase(ckpt_path: Path) -> int:
    """Force-uppercase the VLM backbone reference from ``RLWRLD/RLDX-1-vlm``
    to ``RLWRLD/RLDX-1-VLM`` in config.json, processor_config.json, and
    processor/processor_config.json.

    Context: ``patch_vlm_backbone_reference`` rewrites legacy personal-
    namespace refs to ``RLWRLD/RLDX-1-VLM``, but some checkpoints may have
    landed with a lowercase ``vlm`` suffix. This handler normalises to the
    canonical uppercase form.

    Idempotent: files already containing ``RLWRLD/RLDX-1-VLM`` (uppercase)
    are skipped.
    """
    import json as _json

    targets = [
        ckpt_path / "config.json",
        ckpt_path / "processor_config.json",
        ckpt_path / "processor" / "processor_config.json",
        ckpt_path / "final_processor_config.json",
        ckpt_path / "experiment_cfg" / "final_processor_config.json",
    ]

    total = 0
    for fp in targets:
        if not fp.exists():
            continue
        src = fp.read_text()
        if "RLWRLD/RLDX-1-vlm" not in src:
            continue
        dst = src.replace("RLWRLD/RLDX-1-vlm", "RLWRLD/RLDX-1-VLM")
        if src == dst:
            continue
        try:
            _json.loads(dst)
        except _json.JSONDecodeError as e:
            print(
                f"  [error] {fp.relative_to(ckpt_path)}: VLM uppercase rename produced "
                f"invalid JSON ({e})"
            )
            continue
        fp.write_text(dst)
        print(f"  [vlm-ref-uppercase] {fp.relative_to(ckpt_path)}")
        total += 1

    return total


# ============================================================================
# Sentinel — tracks which handlers have already run on this checkpoint
# ============================================================================

def _load_sentinel(ckpt_path: Path) -> dict | None:
    """Load the `.rldx_patched` sentinel from a checkpoint directory.

    Returns:
        None                 — sentinel does not exist (fresh checkpoint)
        {version, applied, ...} — parsed JSON (schema v2+)
        {version:1, applied:[...], legacy_text:"..."}
                             — legacy v1 text sentinel migrated in-memory; the
                             caller should re-save in v2 JSON on next write.
    """
    p = ckpt_path / PATCH_SENTINEL
    if not p.exists():
        return None
    text = p.read_text().strip()
    if text.startswith("{"):
        try:
            data = json.loads(text)
            if "applied" not in data or "version" not in data:
                raise ValueError("sentinel missing required fields")
            return data
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[warn] {PATCH_SENTINEL} unparseable ({e}); treating as missing")
            return None
    # Legacy v1 text sentinel (freeform). By construction it was only written
    # after the old `main()` ran the embodiment-swap + moss-rename + yaml
    # handlers in sequence, so those three are the ones we know have
    # executed.
    print("[migrate] legacy v1 text sentinel detected — converting to v2 JSON schema on next save")
    return {
        "version": 1,
        "applied": ["field_renames_v1", "embodiment_rename", "moss_enable_to_use_moss", "yaml_rename"],
        "legacy_text": text,
    }


def _save_sentinel(ckpt_path: Path, applied: set[str]) -> None:
    """Persist the `.rldx_patched` sentinel as v2 JSON."""
    p = ckpt_path / PATCH_SENTINEL
    data = {
        "version": SENTINEL_SCHEMA_VERSION,
        "applied": sorted(applied),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    p.write_text(json.dumps(data, indent=2) + "\n")


# ============================================================================
# Generic helpers
# ============================================================================

def resolve_path(path_or_repo_id: str) -> Path:
    """Resolve a local path or HuggingFace repo ID to a local directory."""
    p = Path(path_or_repo_id)
    if p.exists():
        return p
    try:
        from huggingface_hub import snapshot_download
        local = snapshot_download(path_or_repo_id)
        return Path(local)
    except Exception as e:
        print(f"Error: could not resolve '{path_or_repo_id}' as local path or HF repo: {e}")
        sys.exit(1)


def patch_file(filepath: Path, field_patches: dict[str, dict[str, str]]) -> bool:
    """Patch a JSON file by swapping top-level string (or list-of-string)
    field values through the provided mapping. Returns True if anything
    changed.
    """
    if not filepath.exists():
        print(f"  [skip] {filepath.name} not found")
        return False

    with open(filepath) as f:
        data = json.load(f)

    changed = False
    for field, mapping in field_patches.items():
        old_val = data.get(field)
        if isinstance(old_val, list):
            new_list = [mapping.get(v, v) for v in old_val]
            if new_list != old_val:
                data[field] = new_list
                print(f"  [patch] {filepath.name}: {field} = {old_val!r} → {new_list!r}")
                changed = True
            else:
                print(f"  [ok] {filepath.name}: {field} = {old_val!r} (no patch needed)")
        elif old_val in mapping:
            new_val = mapping[old_val]
            data[field] = new_val
            print(f"  [patch] {filepath.name}: {field} = {old_val!r} → {new_val!r}")
            changed = True
        elif old_val is not None:
            print(f"  [ok] {filepath.name}: {field} = {old_val!r} (no patch needed)")

    if changed:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    return changed


# ============================================================================
# Main
# ============================================================================

def apply_all_patches(ckpt_path: Path) -> int:
    """Run all patch handlers in order, sentinel-gated. Returns total changed file count.

    Single source of truth for the patch sequence. The CLI ``main()`` is a thin
    wrapper around this; ``scripts/migrate_hf_naming.py`` also calls it directly so
    the two callers cannot drift out of sync.
    """
    sentinel = _load_sentinel(ckpt_path) or {"version": SENTINEL_SCHEMA_VERSION, "applied": []}
    applied: set[str] = set(sentinel.get("applied", []))

    total_changes = 0

    def _run(name: str, fn) -> None:
        nonlocal total_changes
        if name in applied:
            print(f"[skip] {name} (sentinel records it already ran)")
            return
        print(f"[run] {name}")
        n = fn()
        applied.add(name)
        total_changes += n

    # 1. Static field renames in config.json + processor_config.json
    def _field_renames() -> int:
        total = 0
        for filename, field_patches in PATCHES.items():
            if patch_file(ckpt_path / filename, field_patches):
                total += 1
        return total
    _run("field_renames_v1", _field_renames)

    # 2. Embodiment rename swap — non-idempotent; gate via sentinel
    print()
    _run("embodiment_rename", lambda: patch_embodiment_rename(ckpt_path))
    _run("embodiment_rename_v2", lambda: patch_embodiment_rename_v2(ckpt_path))

    # 3. moss_enable → use_moss
    print()
    _run("moss_enable_to_use_moss", lambda: patch_moss_enable_to_use_moss(ckpt_path))

    # 4. experiment_cfg/*.yaml mirrors of the above
    print()
    _run("yaml_rename", lambda: patch_yaml_files(ckpt_path))

    # 5. Phase 2: metaquery → cog_token (config + yaml + state_dict + weight_map)
    print()
    _run("metaquery_to_cog_token", lambda: patch_metaquery_to_cog_token(ckpt_path))

    # 6. Phase 3: MMDiT → MSAT (config + yaml — state_dict unchanged)
    print()
    _run("mmdit_to_msat", lambda: patch_mmdit_to_msat(ckpt_path))

    # 7. Phase 4: moss → motion (config + yaml + state_dict + weight_map)
    print()
    _run("moss_to_motion", lambda: patch_moss_to_motion(ckpt_path))

    # 8. Phase 5: action_head → action_model (config + yaml + state_dict + weight_map)
    print()
    _run("action_head_to_action_model", lambda: patch_action_head_to_action_model(ckpt_path))

    # 9. Phase 6: contextvla → vtc (config + yaml only — no state_dict impact)
    print()
    _run("contextvla_to_vtc", lambda: patch_contextvla_to_vtc(ckpt_path))

    # 10. Legacy cleanup — strip `use_block_causal_masking` from config.json.
    #     Not part of Issue #29 rename scope, but required for release-era
    #     AutoModel.from_pretrained to accept the older config schema.
    print()
    _run("drop_block_causal_masking", lambda: patch_drop_block_causal_masking(ckpt_path))

    # 11. Legacy cleanup — drop complex64 RoPE `freqs_cis_*` buffers from
    #     safetensors shards. The newer transformers/safetensors backend
    #     can't load complex dtypes; runtime code now registers those
    #     buffers with persistent=False so they're rebuilt at init time.
    print()
    _run("drop_rope_complex_buffers", lambda: patch_drop_rope_complex_buffers(ckpt_path))

    # 11.5. Legacy cleanup — drop pre-#22/#36 image-pipeline kwargs from
    #       processor_config.json (Issue #46). Old ckpts carry
    #       image_crop_size / image_target_size / shortest_image_edge /
    #       use_albumentations / crop_fraction inside processor_kwargs;
    #       new RLDXProcessor.__init__ rejects them as unexpected kwargs.
    print()
    _run("drop_legacy_image_pipeline_kwargs", lambda: patch_drop_legacy_image_pipeline_kwargs(ckpt_path))
    _run("drop_dead_action_head_flags", lambda: patch_drop_dead_action_head_flags(ckpt_path))

    # 12. Legacy migration — flat `action_model.physics_*` keys to the
    #     `action_model.physics.*` PhysicsHead container layout. Runtime
    #     remap exists in the PolicyLoader fallback only, so the primary
    #     AutoModel.from_pretrained path needs on-disk rewriting.
    #     Must run after `action_head_to_action_model` (Phase 5) so the
    #     `action_model.` parent prefix is already settled.
    print()
    _run("physics_flat_to_container", lambda: patch_physics_flat_to_container(ckpt_path))

    # 12.5. Field rename — `new_embodiment_train_ratio`
    #       → `general_embodiment_train_ratio` (model config + processor
    #       kwargs + YAML mirror). Required for `RLDXProcessor.__init__`
    #       to accept the saved processor_kwargs without raising
    #       ``TypeError: got an unexpected keyword argument``.
    print()
    _run(
        "rename_new_embodiment_train_ratio",
        lambda: patch_new_embodiment_train_ratio(ckpt_path),
    )

    # 13. VLM backbone reference — rewrite `huiwon/alinvlm_v1_3` →
    #     `RLWRLD/RLDX-1-VLM` in config.json and processor_config.json so
    #     released ckpts don't carry dead personal-namespace pointers.
    print()
    _run("vlm_backbone_reference", lambda: patch_vlm_backbone_reference(ckpt_path))

    # 13.5. Force-uppercase any residual `RLWRLD/RLDX-1-vlm` → `RLWRLD/RLDX-1-VLM`.
    #       Runs immediately after vlm_backbone_reference so the canonical
    #       uppercase form is always the on-disk value.
    print()
    _run("vlm_backbone_reference_uppercase", lambda: patch_vlm_backbone_reference_uppercase(ckpt_path))

    # 14. Legacy `backbone_model_type: "qwen3_vl"` → `"vtc_qwen3_vl"` —
    #     Issue #36 removed the vanilla Qwen3 backbone path; checkpoints
    #     that still carry "qwen3_vl" would fail the single-backbone guard
    #     in RLDX.__init__ AND mis-route the processor model_type.
    print()
    _run("legacy_qwen3_vl_backbone", lambda: patch_legacy_qwen3_vl_backbone(ckpt_path))

    # 15. model_type underscore → hyphen: "RLDX_1" → "RLDX-1" (release
    #     naming convention). Runs last so it catches any residual value
    #     written by earlier handlers (e.g. field_renames_v1 mapped to
    #     "RLDX_1"; this pass corrects the final form).
    print()
    _run("model_type_underscore_to_hyphen", lambda: patch_model_type_underscore_to_hyphen(ckpt_path))

    _save_sentinel(ckpt_path, applied)
    return total_changes


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    ckpt_path = resolve_path(sys.argv[1])
    print(f"Patching checkpoint: {ckpt_path}\n")
    n = apply_all_patches(ckpt_path)
    if n:
        print(f"\nDone. Patched {n} file(s) or blob(s).")
    else:
        print("\nNo changes needed. Checkpoint is already compatible.")


if __name__ == "__main__":
    main()
