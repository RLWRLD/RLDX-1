"""Serve-time wrappers applying inference optimization paths to an RLDXPolicy.

Paths (see ``rldx/inference/README.md``):
    A — vanilla (no change)
    B — Torch Inductor: ``torch.compile`` per learnable sub-module of the
        inference path (vision tower stays eager — flash-attn varlen op is
        Dynamo-incompatible)
    C — GraphSafe + CUDAGraph: capture once on first real call, replay forever
    D — Custom Chain: GraphSafe substrate + Triton-fused kernel chain

Usage:
    from rldx.inference.serve_optimization import apply_optimization
    apply_optimization(policy, path="D")
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Any

import torch
from transformers.modeling_utils import str_to_torch_dtype

from rldx.utils.dist import rank_zero_print as _print


_INF_DIR = os.path.dirname(os.path.abspath(__file__))
for _sub in (
    "",
    "action_model",
    "backbone",
    "memory",
    "engine",
    os.path.join("action_model", "engine"),
    os.path.join("backbone", "engine"),
    os.path.join("memory", "engine"),
):
    _d = os.path.join(_INF_DIR, _sub) if _sub else _INF_DIR
    if _d not in sys.path:
        sys.path.insert(0, _d)


# safetensors needs an explicit complex64 entry to round-trip RoPE inv_freq.
if "C64" not in str_to_torch_dtype:
    str_to_torch_dtype["C64"] = torch.complex64


def _find_full_model(policy: Any) -> Any:
    """Return the RLDX full model from an RLDXPolicy (or the model itself)."""
    return getattr(policy, "model", policy)


# Per-leaf compile produces one CUDA Graph per leaf, and cudagraph_trees
# cannot track buffer ownership across the eager Python glue between them.
# ``-no-cudagraphs`` keeps Triton autotune while disabling the CUDA Graph wrap.
_PATH_B_COMPILE_MODE = "max-autotune-no-cudagraphs"


def _apply_path_b(full_model: Any) -> dict[str, Any]:
    """Compile every learnable sub-module of the inference path.

    The vision tower is excluded: ``flash_attn._flash_attn_varlen_forward``
    declares ``max_seqlen_*`` as ``SymInt`` but rejects FakeTensors at
    Dynamo trace time, so compilation aborts.
    """
    try:
        llm = full_model.backbone.qwen_model.model.language_model
        action_model = full_model.action_model
    except AttributeError as e:
        raise RuntimeError(f"Model does not expose expected structure for Path B: {e}")

    mode = _PATH_B_COMPILE_MODE
    t0 = time.time()
    compiled_modules: list[str] = []

    for i, layer in enumerate(llm.layers):
        inner = layer.layer if hasattr(layer, "layer") else layer
        compiled = torch.compile(inner, mode=mode)
        if hasattr(layer, "layer"):
            layer.layer = compiled
        else:
            llm.layers[i] = compiled
    compiled_modules.append(f"LLM×{len(llm.layers)}")

    action_model.action_encoder = torch.compile(action_model.action_encoder, mode=mode)
    compiled_modules.append("action_encoder")

    action_model.state_encoder = torch.compile(action_model.state_encoder, mode=mode)
    compiled_modules.append("state_encoder")

    action_model.model = torch.compile(action_model.model, mode=mode)
    compiled_modules.append("MSAT")

    if getattr(action_model, "action_decoder", None) is not None:
        action_model.action_decoder = torch.compile(action_model.action_decoder, mode=mode)
        compiled_modules.append("action_decoder")

    if getattr(action_model, "physics", None) is not None:
        action_model.physics = torch.compile(action_model.physics, mode=mode)
        compiled_modules.append("physics")

    memory_attr = "memory" if hasattr(full_model, "memory") else "_memory_module"
    memory_module = getattr(full_model, memory_attr, None)
    if memory_module is not None:
        setattr(full_model, memory_attr, torch.compile(memory_module, mode=mode))
        compiled_modules.append("memory")

    return {
        "path": "B",
        "compile_mode": mode,
        "setup_s": time.time() - t0,
        "compiled_llm_layers": len(llm.layers),
        "compiled_modules": compiled_modules,
    }


def _extract_memory_config(full_model: Any) -> tuple[Any | None, dict[str, Any] | None]:
    """Return ``(memory_module, memory_config)``, or ``(None, None)`` when memory is off."""
    cfg = full_model.config
    if not getattr(cfg, "use_memory", False):
        return None, None

    memory_module = getattr(full_model, "memory", None) or getattr(
        full_model, "_memory_module", None
    )
    if memory_module is None:
        return None, None

    n_cog = getattr(cfg, "n_cog_tokens", 64)
    mem_n_cog = getattr(cfg, "memory_n_cog_tokens", None) or n_cog

    memory_config = {
        "memory_length": getattr(cfg, "memory_length", 1),
        "memory_n_cog_tokens": mem_n_cog,
        "concat_memory": getattr(cfg, "concat_memory", False),
        "n_cog_tokens": n_cog,
        "hidden_size": memory_module.hidden_size,
    }
    return memory_module, memory_config


def _build_graph_safe_vla_from_real_inputs(
    full_model,
    backbone_inputs,
    action_inputs,
    dtype,
    device,
    num_inference_timesteps=4,
    action_horizon=16,
    prefix_len=0,
):
    """Construct GraphSafeVLA from a loaded model + a real sample.

    ``backbone_inputs`` must be the dict returned by
    ``full_model.prepare_input(...)[0]`` on a real sample. ``prefix_len``
    bakes the RTC trained-mode prefix length into ``GraphSafeActionModel``.
    """
    import importlib.util as _ilu

    def _load(modname, relpath):
        spec = _ilu.spec_from_file_location(modname, os.path.join(_INF_DIR, relpath))
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    from action_model.model import GraphSafeActionModel
    from backbone.model import GraphSafeQwen3VLBackbone

    GraphSafeVLA = _load("_gs_vla", "model/graph_safe_vla.py").GraphSafeVLA

    backbone = full_model.backbone
    action_model = full_model.action_model

    gs_backbone = GraphSafeQwen3VLBackbone(backbone, backbone_inputs).eval()

    with torch.no_grad():
        vl_out = gs_backbone(backbone_inputs)
    n_vl_raw = vl_out.shape[1]
    _print(f"  [PathD] VLM output shape: {list(vl_out.shape)}  (n_vl_raw={n_vl_raw})")

    # Memory support
    memory_module, memory_config = _extract_memory_config(full_model)
    gs_memory = None
    if memory_module is not None and memory_config is not None:
        from memory.model import GraphSafeMemory

        gs_memory = GraphSafeMemory(
            memory_module=memory_module,
            memory_length=memory_config["memory_length"],
            memory_n_cog_tokens=memory_config["memory_n_cog_tokens"],
            device=device,
            dtype=dtype,
        ).eval()
        n_cog_mem = memory_config["memory_n_cog_tokens"]
        n_vl = n_vl_raw + n_cog_mem if memory_config["concat_memory"] else n_vl_raw
    else:
        n_vl = n_vl_raw

    # Infer state horizon and action dims from the real action_inputs
    state = action_inputs["state"] if "state" in action_inputs else action_inputs.state
    n_state = state.shape[1]
    N_sa = n_state + action_horizon

    gs_action_model = GraphSafeActionModel(
        action_model=action_model,
        n_vl=n_vl,
        n_sa_pure=N_sa,
        action_horizon=action_horizon,
        action_dim=action_model.action_decoder.layer2.W.shape[2],
        num_inference_timesteps=num_inference_timesteps,
        device=device,
        dtype=dtype,
        prefix_len=prefix_len,
    ).eval()

    gs_vla = GraphSafeVLA(
        gs_backbone, gs_action_model, gs_memory=gs_memory, memory_config=memory_config
    ).eval()
    return gs_vla, n_state, action_horizon


def _flatten_collated(collated_inputs: dict) -> dict:
    """Mirror ``RLDX.get_action``'s flatten: unwrap nested ``inputs`` and merge
    top-level kwargs (e.g. ``action_prefix``, ``rtc_prefix_len``) into one dict.
    Without this, RTC-injected top-level keys leave modality keys nested under
    ``"inputs"`` and downstream lookups (``backbone_inputs["pixel_values"]``)
    raise KeyError. Matches the eager path's ``inputs is None`` tolerance and
    its loud-fail on collision (see ``RLDX.get_action``).
    """
    inner = collated_inputs.get("inputs")
    if inner is None:
        return {k: v for k, v in collated_inputs.items() if k != "inputs"}
    if not isinstance(inner, dict):
        raise TypeError(
            f"_flatten_collated: 'inputs' must be a dict (got {type(inner).__name__}); "
            "PolicyRuntime always wraps modality dicts under 'inputs'."
        )
    extras = {k: v for k, v in collated_inputs.items() if k != "inputs"}
    collision = set(inner).intersection(extras)
    if collision:
        raise ValueError(
            "_flatten_collated: keys collide between nested 'inputs' and top-level "
            f"kwargs: {sorted(collision)}. Pass each key in exactly one place."
        )
    return {**inner, **extras}


class _CompiledDispatcher:
    """Wraps ``full_model.get_action`` to route through a compiled chain.

    First call serves the action via the vanilla path while building +
    compiling the GraphSafe / Triton chain in the same window. Subsequent
    calls copy real tensors into persistent static buffers and run the
    compiled chain, returning ``{"action_pred": action_tensor}``.
    """

    def __init__(
        self,
        policy,
        path: str,
        compile_mode: str,
        num_inference_timesteps: int = 4,
        action_horizon: int = 16,
        bake_prefix_len: int = 0,
    ):
        """
        Args:
            bake_prefix_len: RTC ``trained`` prefix length baked into
                ``GraphSafeActionModel`` (honoured by both Path C and D —
                the slice writes flow into the CUDAGraph capture and the
                Triton chain through the same wrapper). 0 disables it.
        """
        self.policy = policy
        self.path = path
        self.compile_mode = compile_mode
        self.num_inference_timesteps = num_inference_timesteps
        self.action_horizon = action_horizon
        self.bake_prefix_len = int(bake_prefix_len)

        self.full_model = _find_full_model(policy)
        self._orig_get_action = self.full_model.get_action

        self._ready = False
        self._failed = False
        self._compiled = None
        self._gs_vla = None
        self._buffers = None
        self._action_dim = None
        self._has_physics = False

    def _first_time_build(self, collated_inputs):
        _print(
            f"[Path{self.path}] First call: capturing shapes + building compiled chain...",
            flush=True,
        )
        t0 = time.time()

        dtype = torch.bfloat16
        device = self.full_model.device

        # PolicyRuntime calls ``get_action(**collated)`` — when it wraps the
        # modality dict under ``inputs`` we unwrap it here.
        real_inputs = _flatten_collated(collated_inputs)
        backbone_inputs, action_inputs = self.full_model.prepare_input(real_inputs)

        bi = dict(backbone_inputs)
        action_horizon = self.action_horizon
        n_steps = self.num_inference_timesteps

        gs_vla, n_state, _ = _build_graph_safe_vla_from_real_inputs(
            self.full_model,
            bi,
            action_inputs,
            dtype=dtype,
            device=device,
            num_inference_timesteps=n_steps,
            action_horizon=action_horizon,
            prefix_len=self.bake_prefix_len,
        )
        self._gs_vla = gs_vla

        action_model = self.full_model.action_model
        self._action_dim = action_model.action_decoder.layer2.W.shape[2]
        state = action_inputs["state"] if "state" in action_inputs else action_inputs.state
        state = state.to(device=device, dtype=dtype)
        embodiment_id = (
            action_inputs["embodiment_id"]
            if "embodiment_id" in action_inputs
            else action_inputs.embodiment_id
        )
        embodiment_id = embodiment_id.to(device=device, dtype=torch.long)

        pv = bi["pixel_values"].to(device=device, dtype=dtype)
        if pv.ndim == 3:
            pv = pv.reshape(-1, pv.shape[-1]).contiguous()

        B = state.shape[0]

        # Persistent buffers — addresses fixed for CUDA Graph
        pv_buf = pv.clone().contiguous()
        state_buf = state.clone().contiguous()
        emb_buf = embodiment_id.clone().contiguous()
        init_noise_buf = torch.randn(
            (B, action_horizon, self._action_dim), device=device, dtype=dtype
        )

        self._buffers = {
            "pixel_values": pv_buf,
            "state": state_buf,
            "embodiment_id": emb_buf,
            "init_noise": init_noise_buf,
        }
        if self.bake_prefix_len > 0:
            self._buffers["prefix_actions"] = torch.zeros(
                (B, self.bake_prefix_len, self._action_dim),
                device=device,
                dtype=dtype,
            )

        if self.path == "D":
            import importlib.util as _ilu

            _sp = _ilu.spec_from_file_location(
                "_cvc", os.path.join(_INF_DIR, "engine", "custom_vla_chain.py")
            )
            _cvc_mod = _ilu.module_from_spec(_sp)
            _sp.loader.exec_module(_cvc_mod)
            build_custom_vla_chain = _cvc_mod.build_custom_vla_chain
            compile_custom_vla_chain = _cvc_mod.compile_custom_vla_chain
            chain = build_custom_vla_chain(
                gs_vla,
                device,
                dtype=dtype,
                bake_prefix_len=self.bake_prefix_len,
            )
            self._has_physics = chain.__class__.__name__.startswith("CustomExpanded")
            prefix_buf = self._buffers.get("prefix_actions")
            if prefix_buf is not None:
                sample = (pv_buf, state_buf, emb_buf, init_noise_buf, prefix_buf)
            else:
                sample = (pv_buf, state_buf, emb_buf, init_noise_buf)
            compiled, compile_time = compile_custom_vla_chain(
                chain, sample, compile_mode=self.compile_mode
            )
            self._compiled = compiled
            _print(
                f"[PathD] Custom chain built + compiled in {compile_time:.1f}s "
                f"(bake_prefix_len={self.bake_prefix_len})",
                flush=True,
            )

        else:  # "C"
            import importlib.util as _ilu

            _sp = _ilu.spec_from_file_location(
                "_cg", os.path.join(_INF_DIR, "engine", "cuda_graph.py")
            )
            _cg_mod = _ilu.module_from_spec(_sp)
            _sp.loader.exec_module(_cg_mod)
            setup_vla_cuda_graph = _cg_mod.setup_vla_cuda_graph
            prefix_buf = self._buffers.get("prefix_actions")
            replay_fn, _ = setup_vla_cuda_graph(
                gs_vla,
                bi,
                state_buf,
                emb_buf,
                init_noise=init_noise_buf,
                physics_init_noise=None,
                prefix_actions=prefix_buf,
            )

            def _replay(pv, st, emb, init_noise=None, prefix_actions=None):
                return replay_fn(
                    bi,
                    st,
                    emb,
                    init_noise=init_noise,
                    physics_init_noise=None,
                    prefix_actions=prefix_actions,
                )

            self._compiled = _replay
            self._has_physics = False
            _print("[PathC] CUDA Graph captured", flush=True)

        _print(f"[Path{self.path}] Build+compile total: {time.time() - t0:.1f}s", flush=True)
        self._ready = True

    def __call__(self, **collated_inputs):
        if self._failed:
            return self._orig_get_action(**collated_inputs)

        if not self._ready:
            # First call: serve via vanilla and build the compiled chain.
            first_result = self._orig_get_action(**collated_inputs)
            try:
                self._first_time_build(collated_inputs)
            except Exception as e:
                self._failed = True
                _print(f"[Path{self.path}] Build failed: {e}. Falling back to vanilla.", flush=True)
                traceback.print_exc()
            return first_result

        try:
            real_inputs = _flatten_collated(collated_inputs)
            backbone_inputs, action_inputs = self.full_model.prepare_input(real_inputs)
            pv = backbone_inputs["pixel_values"]
            if pv.ndim == 3:
                pv = pv.reshape(-1, pv.shape[-1])
            pv = pv.to(self._buffers["pixel_values"].dtype).contiguous()

            st = action_inputs["state"] if "state" in action_inputs else action_inputs.state
            st = st.to(self._buffers["state"].dtype).contiguous()
            emb = (
                action_inputs["embodiment_id"]
                if "embodiment_id" in action_inputs
                else action_inputs.embodiment_id
            )
            emb = emb.to(torch.long).contiguous()

            # Shape drift would corrupt the captured graph — fall back to vanilla.
            for k, new in [("pixel_values", pv), ("state", st), ("embodiment_id", emb)]:
                if new.shape != self._buffers[k].shape:
                    _print(
                        f"[Path{self.path}] Shape drift on '{k}' "
                        f"({new.shape} vs {self._buffers[k].shape}); vanilla fallback.",
                        flush=True,
                    )
                    return self._orig_get_action(**collated_inputs)

            self._buffers["pixel_values"].copy_(pv)
            self._buffers["state"].copy_(st)
            self._buffers["embodiment_id"].copy_(emb)
            self._buffers["init_noise"].normal_()

            # Trained-mode prefix — copy frozen actions from the previous
            # chunk into the persistent buffer; PolicyRuntime injects them
            # under ``action_prefix`` (or top-level kwarg).
            prefix_buf = self._buffers.get("prefix_actions")
            if prefix_buf is not None:
                src = real_inputs.get("action_prefix")
                if src is None:
                    raise RuntimeError(
                        "trained-mode chain requires ``action_prefix`` in the "
                        "request inputs but none was provided"
                    )
                src = src.to(device=prefix_buf.device, dtype=prefix_buf.dtype)
                if src.shape[1] >= self.bake_prefix_len:
                    prefix_buf.copy_(src[:, : self.bake_prefix_len])
                else:
                    raise RuntimeError(
                        f"action_prefix has {src.shape[1]} timesteps but the "
                        f"chain was baked for prefix_len={self.bake_prefix_len}"
                    )

            extra = {"prefix_actions": prefix_buf} if prefix_buf is not None else {}
            with torch.no_grad():
                action = self._compiled(
                    self._buffers["pixel_values"],
                    self._buffers["state"],
                    self._buffers["embodiment_id"],
                    init_noise=self._buffers["init_noise"],
                    **extra,
                )

            from transformers import BatchFeature

            return BatchFeature({"action_pred": action})
        except Exception as e:
            _print(f"[Path{self.path}] Hot-path failure: {e}. Falling back to vanilla.", flush=True)
            traceback.print_exc()
            self._failed = True
            return self._orig_get_action(**collated_inputs)


def _apply_path_cd(policy, path: str, compile_mode: str, bake_prefix_len: int = 0) -> dict:
    """Install a first-call-capture dispatcher on ``policy.model.get_action``.

    The wrapper sits outside the model — ``RLDX`` itself is unchanged, and
    the captured graph is the same forward the dispatcher otherwise runs.
    """
    full_model = _find_full_model(policy)
    cfg = getattr(full_model, "config", None)
    action_horizon = int(getattr(cfg, "action_horizon", 16)) if cfg is not None else 16
    dispatcher = _CompiledDispatcher(
        policy,
        path=path,
        compile_mode=compile_mode,
        action_horizon=action_horizon,
        bake_prefix_len=bake_prefix_len,
    )
    dispatcher.full_model.get_action = dispatcher  # type: ignore[assignment]
    return {
        "path": path,
        "installed": "_CompiledDispatcher",
        "compile_mode": compile_mode,
        "bake_prefix_len": dispatcher.bake_prefix_len,
    }


from rldx.inference._rtc_dispatch import resolve_rtc_for_bake as _resolve_rtc_for_bake  # noqa: E402


def apply_optimization(policy, path: str = "A", compile_mode: str = "max-autotune") -> dict:
    """Apply an inference optimization path to ``policy``.

    See module docstring for the path matrix. ``compile_mode`` is forwarded
    to paths C/D; path A is eager and path B fixes its own mode internally.
    """
    path = path.upper()
    if path not in {"A", "B", "C", "D"}:
        raise ValueError(f"Unknown optimization path: {path!r}")

    full_model = _find_full_model(policy)

    if path == "A":
        _print("[serve_optimization] Path A (vanilla) — no model modification.")
        return {"path": "A"}

    if path == "B":
        try:
            info = _apply_path_b(full_model)
            _print(
                f"[serve_optimization] Path B applied in {info['setup_s']:.2f}s "
                f"(compiled: {', '.join(info['compiled_modules'])}, "
                f"mode={info['compile_mode']})."
            )
            return info
        except Exception as e:
            _print(f"[serve_optimization] Path B failed: {e}. Falling back to A.")
            traceback.print_exc()
            return {"path": "B", "fallback": "A", "error": str(e)}

    bake_prefix_len = _resolve_rtc_for_bake(full_model, path)
    cfg = getattr(full_model, "config", None)
    rtc_mode = getattr(cfg, "rtc_inference_mode", "none") if cfg is not None else "none"
    if rtc_mode == "guided":
        raise ValueError(
            f"path={path!r} cannot serve rtc_inference_mode='guided' — the compiled "
            "fullgraph cannot route VJP through ``action_prefix``. Use rtc_inference_mode "
            "in {'none', 'trained'} for paths C/D, or fall back to path A/B."
        )
    if rtc_mode == "trained" and bake_prefix_len == 0:
        _print(
            "[serve_optimization] WARNING: rtc_inference_mode='trained' but "
            "rtc_inference_delay is 0 — captured graph will run as mode='none'."
        )

    _print(
        f"[serve_optimization] Path {path}: installing first-call-capture dispatcher "
        f"(compile_mode={compile_mode}, bake_prefix_len={bake_prefix_len})."
    )
    return _apply_path_cd(
        policy,
        path=path,
        compile_mode=compile_mode,
        bake_prefix_len=bake_prefix_len,
    )
