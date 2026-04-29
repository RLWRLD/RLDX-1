"""Model loading for the inference benchmarks."""

from __future__ import annotations

import contextlib
import gc
import io
import json
from pathlib import Path

import torch
from transformers import AutoModel
from transformers.modeling_utils import str_to_torch_dtype

import rldx.model.core.processing_rldx  # noqa: F401

# Importing the RLDX model + processor modules has the side effect of
# registering ``"RLDX_1"`` with ``AutoConfig`` / ``AutoModel`` /
# ``AutoProcessor``.  Without these the bare ``AutoModel.from_pretrained``
# call below cannot resolve a patched checkpoint whose ``config.json``
# carries ``model_type="RLDX_1"``.
import rldx.model.core.rldx  # noqa: F401
from rldx.utils.dist import rank_zero_print as _print

from .registry import MODEL_REGISTRY


# safetensors needs an explicit complex64 entry to round-trip RoPE inv_freq.
if "C64" not in str_to_torch_dtype:
    str_to_torch_dtype["C64"] = torch.complex64


# Model-type strings that ``scripts/patch_checkpoint.py`` migrates — a local
# checkpoint dir carrying any of these without the ``.rldx_patched`` sentinel
# triggers a loud warning from ``_ensure_patched`` below.
_UNMIGRATED_MODEL_TYPES = {"AlinVLAv0", "AlinVLAv0Memory", "Gr00tN1d6", "RLDX"}


def _ensure_patched(model_path) -> None:
    """Warn when a local ckpt dir looks unpatched.

    Skips HF repo IDs and invalid paths silently — only fires when a local
    directory exists with a ``config.json`` whose ``model_type`` is one of
    the pre-RLDX_1 strings and there's no ``.rldx_patched`` sentinel.
    """
    if not isinstance(model_path, (str, Path)):
        return
    p = Path(str(model_path))
    if not p.exists() or not p.is_dir() or (p / ".rldx_patched").exists():
        return
    config_json = p / "config.json"
    if not config_json.exists():
        return
    try:
        cfg = json.loads(config_json.read_text())
    except Exception:
        return
    if cfg.get("model_type") in _UNMIGRATED_MODEL_TYPES:
        _print(
            f"[rldx.inference.loader] WARNING: {p} has model_type="
            f"{cfg.get('model_type')!r} but no ``.rldx_patched`` sentinel. "
            f"Run ``python scripts/patch_checkpoint.py {p}`` first."
        )


def _load_full_model(
    model_type, model_path=None, device=0, rtc_inference_mode=None, rtc_inference_delay=None
):
    """Load the full RLDX model from a checkpoint.

    ``rtc_inference_mode`` / ``rtc_inference_delay`` mirror the same
    flags on ``run_rldx_server.py`` — when non-None they're forwarded
    to ``AutoModel.from_pretrained`` and override the on-disk
    ``config.rtc_inference_*`` fields before ``RLDX.__init__`` runs
    its ``RTCConfig.validate`` check.  This is the only RTC config
    path the benchmark / server consumers can affect from the
    outside; everything downstream reads ``model.config`` so a single
    override here propagates to the dispatcher's ``bake_prefix_len``
    and the GraphSafe forward.
    """
    model_cfg = MODEL_REGISTRY[model_type]
    hf_path = model_path or model_cfg["hf_path"]

    torch.cuda.set_device(device)
    cuda_device = torch.device(f"cuda:{device}")

    overrides = {}
    if rtc_inference_mode is not None:
        overrides["rtc_inference_mode"] = rtc_inference_mode
    if rtc_inference_delay is not None:
        overrides["rtc_inference_delay"] = int(rtc_inference_delay)

    _print(f"  Loading full RLDX model: {hf_path}")
    if overrides:
        _print(f"  RTC override: {overrides}")
    with contextlib.redirect_stdout(io.StringIO()):
        full_model = AutoModel.from_pretrained(
            hf_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            **overrides,
        )
    return full_model, cuda_device


def load_backbone(args):
    """Load backbone model and collect metadata.

    Returns:
        backbone: nn.Module
        meta: dict with keys: select_layer, hidden_size,
              num_llm_layers, num_deepstack, model_cfg, device
    """
    model_cfg = MODEL_REGISTRY[args.model_type]
    model_path = args.model_path or model_cfg["hf_path"]
    # The VTC backbone variant (Issue #29 Phase 6 rename of contextvla → vtc)
    # is selected when the model_type carries either spelling. The legacy
    # ``contextvla`` / ``alinvla`` substrings stay matched for back-compat
    # with un-migrated registry entries.
    is_vtc = (
        "vtc" in args.model_type or "contextvla" in args.model_type or "alinvla" in args.model_type
    )

    # Apply model-specific default args
    for k, v in model_cfg.get("default_args", {}).items():
        cli_key = k.replace("-", "_")
        if getattr(args, cli_key, None) is None or not getattr(args, cli_key):
            setattr(args, cli_key, v)

    _print(f"Loading backbone: {args.model_type}")
    _print(f"  HF path: {model_path}")

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")

    if model_cfg.get("load_mode") == "extract_backbone":
        full_model, device = _load_full_model(args.model_type, model_path, device=args.device)
        backbone = full_model.backbone
        del full_model.action_model
        del full_model
        gc.collect()
        torch.cuda.empty_cache()
        backbone.requires_grad_(False)
        backbone = backbone.to(device=device, dtype=torch.bfloat16).eval()
        select_layer = backbone.select_layers[-1]
        _print(f"  Extracted backbone (action_model freed, select_layer={select_layer})")
    else:
        import importlib

        module_path, class_name = model_cfg["backbone_cls"].rsplit(".", 1)
        mod = importlib.import_module(module_path)
        BackboneCls = getattr(mod, class_name)

        select_layer = 18
        backbone_kwargs = dict(
            model_name=model_path,
            tune_llm=False,
            tune_visual=False,
            select_layer=select_layer,
            load_bf16=True,
        )
        if not is_vtc:
            backbone_kwargs["use_flash_attention"] = True
        if getattr(args, "use_cog_tokens", False):
            backbone_kwargs["use_cog_tokens"] = True
            backbone_kwargs["cog_mode"] = "cog_only"
            backbone_kwargs["n_cog_tokens"] = args.n_cog_tokens

        backbone = BackboneCls(**backbone_kwargs)
        backbone.qwen_linear = backbone.qwen_linear.to(torch.bfloat16)
        backbone = backbone.to(device).eval()

    backbone.qwen_model.config.use_cache = False
    backbone.qwen_model.model.language_model.config.use_cache = False

    hidden_size = backbone.qwen_model.model.language_model.config.hidden_size
    num_llm_layers = len(backbone.qwen_model.model.language_model.layers)
    num_deepstack = len(getattr(backbone.qwen_model.model.visual, "deepstack_visual_indexes", []))

    _print(f"  Hidden size: {hidden_size}")
    _print(f"  LLM layers (after pruning): {num_llm_layers}")
    _print(f"  Select layers: {backbone.select_layers}")

    meta = {
        "select_layer": select_layer,
        "hidden_size": hidden_size,
        "num_llm_layers": num_llm_layers,
        "num_deepstack": num_deepstack,
        "model_cfg": model_cfg,
        "device": device,
    }
    return backbone, meta


def load_action_model(model_type="rldx_1_pretrain", model_path=None, device=0):
    """Load action-model components from checkpoint.

    Loads the full RLDX model and extracts the action_model module,
    then frees the backbone to save memory.

    Returns:
        components: dict with keys: msat, state_encoder, action_encoder,
                    action_decoder, position_embedding, dims, plus physics
                    sub-modules when ``use_physics=True``.
    """
    full_model, cuda_device = _load_full_model(model_type, model_path, device=device)

    action_model = full_model.action_model
    action_model.requires_grad_(False)
    action_model = action_model.to(device=cuda_device, dtype=torch.bfloat16).eval()

    # Extract sub-modules. The MSAT diffusion model is registered as
    # ``action_model.model`` so its state-dict keys live under that prefix.
    msat = action_model.model
    state_encoder = action_model.state_encoder
    action_encoder = action_model.action_encoder
    action_decoder = action_model.action_decoder
    position_embedding = action_model.position_embedding

    # Dimensions from the loaded model
    sa_dim = msat.inner_dim  # 1536
    vl_dim = msat.vl_proj_to_sa.in_features  # 4096
    hidden_size = msat.proj_out_2.out_features  # 1024
    action_dim = action_decoder.layer2.W.shape[2]  # output_dim
    max_state_dim = state_encoder.layer1.W.shape[1]  # input_dim

    # Physics sub-modules (RLDX-1 midtrain variants). ``PhysicsHead`` nests
    # them under ``action_model.physics.*``; ``use_physics`` is a flag on
    # ``action_model`` itself.
    use_physics = getattr(action_model, "use_physics", False)
    physics_info = {}
    if use_physics:
        physics = action_model.physics
        physics_info = {
            "physics_cond_encoder": physics.physics_cond_encoder,
            "physics_fut_encoder": physics.physics_fut_encoder,
            "physics_decoder": physics.physics_decoder,
            "physics_dim": physics.physics_dim,
            "physics_hist_len": physics.physics_hist_len,
            "physics_fut_len": physics.physics_fut_len,
            "physics_use_flow_matching": physics.physics_use_flow_matching,
        }
        _print(
            f"    Physics: dim={physics_info['physics_dim']}, "
            f"hist={physics_info['physics_hist_len']}, "
            f"fut={physics_info['physics_fut_len']}, "
            f"flow_matching={physics_info['physics_use_flow_matching']}"
        )

    # Free backbone
    del full_model.backbone
    del full_model
    gc.collect()
    torch.cuda.empty_cache()

    _print("  Extracted action model (backbone freed)")
    _print(f"    MSAT: {sum(p.numel() for p in msat.parameters()) / 1e6:.1f}M params")
    _print(f"    sa_dim={sa_dim}, vl_dim={vl_dim}, hidden_size={hidden_size}")

    return {
        "msat": msat,
        "state_encoder": state_encoder,
        "action_encoder": action_encoder,
        "action_decoder": action_decoder,
        "position_embedding": position_embedding,
        "action_model": action_model,  # keep full ref for GraphSafeActionModel
        "use_physics": use_physics,
        **physics_info,
        "dims": {
            "sa_dim": sa_dim,
            "vl_dim": vl_dim,
            "hidden_size": hidden_size,
            "action_dim": action_dim,
            "max_state_dim": max_state_dim,
        },
    }


def load_memory(model_type="rldx_1_midtrain_allex", model_path=None, device=0):
    """Load memory module from checkpoint.

    The checkpoint config stores `use_memory=True` and `memory_cfg`, but
    AutoModel may load a non-memory variant of the RLDX class which
    doesn't create a ``.memory`` attribute. We construct TransformerMemory
    from the config and load the ``memory.*`` weights from the state dict
    directly.

    Returns:
        memory_module: TransformerMemory (on device, eval, bf16)
        memory_config: dict with memory_length, memory_n_cog_tokens,
                       concat_memory, hidden_size, etc.
    """
    from rldx.model.modules.memory import TransformerMemory

    model_cfg = MODEL_REGISTRY[model_type]
    hf_path = model_path or model_cfg["hf_path"]

    torch.cuda.set_device(device)
    cuda_device = torch.device(f"cuda:{device}")

    _print(f"  Loading memory from: {hf_path}")

    # --- Read config.json for memory parameters ---
    from huggingface_hub import hf_hub_download

    cfg_path = hf_hub_download(hf_path, "config.json")
    with open(cfg_path) as f:
        ckpt_cfg = json.load(f)

    if not ckpt_cfg.get("use_memory", False):
        raise ValueError(
            f"Checkpoint {hf_path} has use_memory=False. "
            f"Use a checkpoint trained with --use-memory."
        )

    mem_cfg_raw = ckpt_cfg["memory_cfg"]
    memory_length = ckpt_cfg.get("memory_length", 4)
    # Pre-RLDX_1 checkpoints (not yet run through patch_checkpoint.py) still
    # store the value under the old key — fall back to it so this loader
    # can warn the user instead of silently defaulting.
    memory_n_cog_tokens = ckpt_cfg.get("memory_n_cog_tokens") or ckpt_cfg.get(
        "memory_n_meta_queries", 16
    )

    # The pre-RLDX_1 memory class overrode hidden_size to backbone dim and
    # set block_attn_size = memory_n_cog_tokens. Checkpoint configs store
    # the *pre-override* values, so we infer the real hidden_size from
    # the saved weights below.
    mem_cfg_raw["block_attn_size"] = memory_n_cog_tokens
    mem_cfg_raw["max_position_embeddings"] = memory_length * memory_n_cog_tokens

    # --- Load memory weights from state dict ---

    from safetensors import safe_open

    index_path = hf_hub_download(hf_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    # Find which shards contain memory.* keys
    memory_shards = set()
    for key, shard in index["weight_map"].items():
        if key.startswith("memory."):
            memory_shards.add(shard)

    memory_state_dict = {}
    for shard_name in memory_shards:
        shard_path = hf_hub_download(hf_path, shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("memory."):
                    # Strip "memory." prefix for TransformerMemory.load_state_dict()
                    clean_key = key[len("memory.") :]
                    memory_state_dict[clean_key] = f.get_tensor(key)

    # Infer real hidden_size from weight shape
    if "layers.0.self_attn.q_proj.weight" in memory_state_dict:
        real_hidden = memory_state_dict["layers.0.self_attn.q_proj.weight"].shape[1]
        mem_cfg_raw["hidden_size"] = real_hidden
        mem_cfg_raw["intermediate_size"] = real_hidden * 4

    # --- Construct TransformerMemory and load weights ---
    memory_module = TransformerMemory(**mem_cfg_raw)
    missing, unexpected = memory_module.load_state_dict(memory_state_dict, strict=False)
    if missing:
        _print(f"    [warn] Missing keys: {missing}")
    if unexpected:
        _print(f"    [warn] Unexpected keys: {unexpected}")

    memory_module.requires_grad_(False)
    memory_module = memory_module.to(device=cuda_device, dtype=torch.bfloat16).eval()

    n_cog_tokens = ckpt_cfg.get("n_cog_tokens") or ckpt_cfg.get("n_meta_queries", 64)
    memory_config = {
        "memory_length": memory_length,
        "memory_n_cog_tokens": memory_n_cog_tokens,
        "concat_memory": ckpt_cfg.get("concat_memory", False),
        "n_cog_tokens": n_cog_tokens,
        "hidden_size": memory_module.hidden_size,
        "num_layers": len(memory_module.layers),
        "num_heads": memory_module.config.num_attention_heads,
        "block_attn_size": memory_module.block_attn_size,
        "use_causal_attn": memory_module.use_causal_attn,
        "use_rope": memory_module.use_rope,
    }

    n_params = sum(p.numel() for p in memory_module.parameters()) / 1e6

    _print(f"  Loaded memory module ({len(memory_state_dict)} tensors)")
    _print(f"    Params: {n_params:.1f}M")
    _print(
        f"    Config: K={memory_config['memory_length']}, "
        f"n_cog_mem={memory_config['memory_n_cog_tokens']}, "
        f"concat={memory_config['concat_memory']}"
    )
    _print(
        f"    hidden_size={memory_config['hidden_size']}, "
        f"layers={memory_config['num_layers']}, "
        f"heads={memory_config['num_heads']}"
    )

    return memory_module, memory_config


def load_vla(
    model_type="rldx_1_pretrain",
    model_path=None,
    device=0,
    rtc_inference_mode=None,
    rtc_inference_delay=None,
):
    """Load full VLA model: backbone + action_model (intact).

    Unlike load_backbone() / load_action_model() which free the other half,
    this keeps everything for the full VLA pipeline.

    Returns:
        backbone: nn.Module (backbone, on device)
        action_model: RLDXActionModel (intact, on device)
        meta: dict with select_layer, hidden_size, etc.
    """
    full_model, cuda_device = _load_full_model(
        model_type,
        model_path,
        device=device,
        rtc_inference_mode=rtc_inference_mode,
        rtc_inference_delay=rtc_inference_delay,
    )

    # Move the *whole* model to the target device first.  Sub-modules
    # carved out below (``backbone``, ``action_model``) share parameter
    # objects with ``full_model``, so a per-submodule ``.to(...)`` would
    # cover them — but ``full_model.memory`` (when use_memory=True) is a
    # peer of those two and would otherwise stay on CPU and crash inside
    # ``RLDX.get_action`` → ``self.memory(...)`` with the classic device
    # mismatch (`weight cpu, input cuda:0`).  One whole-model move is
    # cheaper to reason about than tracking which peer modules need it.
    full_model = full_model.to(device=cuda_device, dtype=torch.bfloat16)

    # --- Extract backbone ---
    backbone = full_model.backbone
    backbone.requires_grad_(False)
    backbone = backbone.eval()

    select_layer = backbone.select_layers[-1]
    backbone.qwen_model.config.use_cache = False
    backbone.qwen_model.model.language_model.config.use_cache = False

    hidden_size = backbone.qwen_model.model.language_model.config.hidden_size
    num_llm_layers = len(backbone.qwen_model.model.language_model.layers)
    num_deepstack = len(getattr(backbone.qwen_model.model.visual, "deepstack_visual_indexes", []))

    # --- Extract action model (intact, including vlln) ---
    action_model = full_model.action_model
    action_model.requires_grad_(False)
    action_model = action_model.eval()

    # Keep full_model alive (backbone/action_model are same Python objects, no extra GPU memory)
    full_model.requires_grad_(False)
    full_model.eval()

    _print("  Loaded full VLA (backbone + action_model)")
    _print(f"    Backbone: select_layer={select_layer}, hidden_size={hidden_size}")
    _print(f"    MSAT: {sum(p.numel() for p in action_model.model.parameters()) / 1e6:.1f}M params")

    # Physics info — fields live under action_model.physics (PhysicsHead).
    use_physics = getattr(action_model, "use_physics", False)
    if use_physics:
        physics = action_model.physics
        _print(
            f"    Physics: dim={physics.physics_dim}, "
            f"hist={physics.physics_hist_len}, fut={physics.physics_fut_len}"
        )

    # Motion module — lives on the underlying Qwen3VL visual encoder,
    # not the wrapper backbone class.
    visual = backbone.qwen_model.model.visual
    motion_block = getattr(visual, "motion_block", None)
    motion_insert_layer = getattr(visual, "motion_insert_layer", None)
    if motion_block is not None:
        _print(f"    Motion: insert_layer={motion_insert_layer}")

    # Memory info — when ``use_memory=True`` the base RLDX class doesn't
    # always create a ``.memory`` attribute, so re-load ``memory.*``
    # weights from the state dict directly.
    memory_module = None
    memory_config = None
    hf_path = model_path or MODEL_REGISTRY[model_type]["hf_path"]
    try:
        from huggingface_hub import hf_hub_download

        _cfg = json.load(open(hf_hub_download(hf_path, "config.json")))
        if _cfg.get("use_memory", False):
            _print("    Memory: detected in checkpoint, loading...")
            memory_module, memory_config = load_memory(model_type, model_path, device=device)
    except Exception as e:
        _print(f"    Memory: detection failed ({e})")

    meta = {
        "full_model": full_model,
        "select_layer": select_layer,
        "hidden_size": hidden_size,
        "num_llm_layers": num_llm_layers,
        "num_deepstack": num_deepstack,
        "model_cfg": MODEL_REGISTRY[model_type],
        "device": cuda_device,
        "use_physics": use_physics,
        "motion_block": motion_block,
        "motion_insert_layer": motion_insert_layer,
        "memory_module": memory_module,
        "memory_config": memory_config,
    }

    return backbone, action_model, meta
