import glob
import json
import os
from typing import Optional

# from transformers.trainer_utils import load_sharded_checkpoint
from accelerate import load_checkpoint_in_model
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoProcessor

from rldx.utils.dist import rank_zero_print as _print

from .layer_wrapper import LayerWrapper
from .modeling_qwen3_vl import Qwen3VLForConditionalGeneration


def _checkpoint_has_motion_weights(
    path_or_name: str, revision: Optional[str] = None
) -> Optional[bool]:
    """Detect whether a checkpoint carries `motion_block.*` parameter tensors.

    Returns:
        True:  definitively contains motion module weights (skip re-init)
        False: definitively does not contain motion module weights (safe to re-init)
        None:  could not determine (conservative: preserve loaded weights)

    Works for both sharded (model.safetensors.index.json + model-*-of-*.safetensors)
    and non-sharded (single model.safetensors) layouts. Handles local directories
    and HF Hub identifiers. Any probe failure returns None rather than a
    false-negative, because a false-negative here silently overwrites trained
    motion module weights via `motion_block.initialize_weights()`.
    """
    from safetensors import safe_open

    if os.path.isdir(path_or_name):
        shards = sorted(glob.glob(os.path.join(path_or_name, "*.safetensors")))
        if not shards:
            _print(f"[w] motion module probe: no *.safetensors under {path_or_name}")
            return None
        try:
            for shard in shards:
                with safe_open(shard, framework="pt") as f:
                    if any("motion_block" in k for k in f.keys()):
                        return True
            return False
        except Exception as exc:
            _print(f"[w] motion module probe: scan failed on {path_or_name}: {exc!r}")
            return None

    # HF Hub path: try sharded index first (cheap), then fall back to the
    # single-shard layout. EntryNotFoundError from the Hub means "file
    # doesn't exist in the repo" — that's information, not an error.
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
    except Exception as exc:
        _print(f"[w] motion module probe: huggingface_hub unavailable: {exc!r}")
        return None

    try:
        index_path = hf_hub_download(
            path_or_name, "model.safetensors.index.json", revision=revision
        )
    except EntryNotFoundError:
        index_path = None
    except Exception as exc:
        _print(f"[w] motion module probe: index.json download failed for {path_or_name}: {exc!r}")
        return None

    if index_path is not None:
        try:
            with open(index_path) as f:
                index = json.load(f)
            return any("motion_block" in k for k in index.get("weight_map", {}))
        except Exception as exc:
            _print(f"[w] motion module probe: index.json parse failed: {exc!r}")
            return None

    # Non-sharded HF Hub checkpoint: don't download the full model.safetensors
    # (potentially 10s of GB) just to read key names. Return indeterminate and
    # let the caller preserve whatever weights from_pretrained already loaded.
    _print(
        f"[w] motion module probe: {path_or_name} has no sharded index.json. Not "
        f"downloading full model.safetensors for key scan; preserving "
        f"loaded motion module weights (skipping re-init)."
    )
    return None


class VTC_Qwen3VL(Qwen3VLForConditionalGeneration):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, motion_config=None, **kwargs):
        # Issue #22 — pop HF download kwargs out so they reach every
        # snapshot_download / *.from_pretrained call below that actually hits
        # the Hub. If we left them in ``kwargs`` they would flow only into
        # ``Qwen3VLForConditionalGeneration.from_pretrained`` and break
        # ``_from_config`` (which takes model-init kwargs, not download
        # kwargs) on the VTC branch. Pinning ``revision`` here is the whole
        # point of the fix — previously HEAD was pulled regardless of
        # ``--model-revision``.
        download_kwargs = {
            k: kwargs.pop(k) for k in ("revision", "cache_dir", "token") if k in kwargs
        }
        revision = download_kwargs.get("revision")

        if "vtc" in pretrained_model_name_or_path.lower():
            _print(
                f"[i] VTC loading pretrained VTC + Qwe3nVL weights from {pretrained_model_name_or_path}"
            )
            # Reference architecture config — always the upstream Qwen3-VL.
            # revision pins the RLDX repo, not this reference, so don't thread
            # download_kwargs here.
            base_config = AutoConfig.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        else:
            _print(f"[i] VTC loading Qwen3-VL weights from {pretrained_model_name_or_path}")
            base_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, **download_kwargs
            )

        # Inject motion module config into vision_config before model construction
        if motion_config is not None:
            for k, v in motion_config.items():
                setattr(base_config.vision_config, k, v)
            _print(f"[i] motion module config injected into vision_config: {motion_config}")

        if "vtc" in pretrained_model_name_or_path.lower():
            model = Qwen3VLForConditionalGeneration._from_config(base_config, **kwargs)
        else:
            # Only pass explicit config when motion module modifies it; otherwise use default loading
            extra = {"config": base_config} if motion_config is not None else {}
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path, **extra, **download_kwargs, **kwargs
            )

        # Re-apply motion module init only when motion module is newly added (not in checkpoint).
        # from_pretrained's _init_weights overwrites kaiming Conv3d init and any
        # custom init applied in MotionModule.initialize_weights().
        if motion_config is not None and hasattr(model.model.visual, "motion_block"):
            probe = _checkpoint_has_motion_weights(pretrained_model_name_or_path, revision=revision)
            if probe is True:
                _print(
                    "[i] motion module weights loaded from checkpoint, skipping re-initialization"
                )
            elif probe is False:
                model.model.visual.motion_block.initialize_weights()
                _print("[i] motion module weights re-initialized (not found in checkpoint)")
            else:
                # Indeterminate: preserve whatever from_pretrained already
                # loaded. Re-initializing here risked overwriting trained
                # motion module weights when the probe couldn't reach the checkpoint
                # (missing index.json for non-sharded ckpts, HF download
                # failure, malformed json, etc.).
                _print(
                    "[w] motion module ckpt probe indeterminate — skipping re-init to "
                    "avoid overwriting loaded weights. If fresh motion module init is "
                    "intended, verify the checkpoint layout."
                )

        for layer_idx in range(len(model.model.language_model.layers)):
            model.model.language_model.layers[layer_idx] = LayerWrapper(
                model.model.language_model.layers[layer_idx],
                layer_idx=layer_idx,
                internal_projection=4,
                img_pattern=[151652],
                motion_token=1,
            )
        if "vtc" in pretrained_model_name_or_path.lower():
            if os.path.isdir(pretrained_model_name_or_path):
                local_dir = pretrained_model_name_or_path
            else:
                # Issue #22 — thread revision/cache_dir/token so the weight
                # blobs we load below match the pinned commit, not HEAD.
                local_dir = snapshot_download(pretrained_model_name_or_path, **download_kwargs)

            processor = AutoProcessor.from_pretrained(
                pretrained_model_name_or_path, **download_kwargs
            )

            model.resize_token_embeddings(len(processor.tokenizer))

            load_checkpoint_in_model(model, local_dir, device_map={"": "cpu"})
            _print(f"[VTC] weights loaded from {local_dir}")

        return model
