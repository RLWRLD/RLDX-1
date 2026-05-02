import json
from pathlib import Path

import numpy as np
from rldx.configs.base_config import Config
from rldx.configs.model.rldx import RLDXConfig
from rldx.data.dataset.factory import DatasetFactory
from rldx.experiment.dist_utils import get_rank
from rldx.model.core.processing_rldx import RLDXProcessor
from rldx.model.core.rldx import RLDX
from rldx.model.pipeline import ModelPipeline
from rldx.model.registry import register_model
from rldx.utils.dist import rank_zero_print as _print
from safetensors import safe_open
from termcolor import colored
import torch
from transformers import AutoProcessor
import transformers.modeling_utils as modeling_utils


# Complex tensors that safetensors cannot load (re-computed at model init)
_COMPLEX_TENSOR_SKIP_KEYS = {
    "action_model.model.rope_embedder.freqs_cis_0",
    "action_model.model.rope_embedder.freqs_cis_1",
}


def _load_checkpoint_state_dict(ckpt_path: Path, _patched_loader, skip_keys: set) -> dict:
    """Load checkpoint state dict from sharded or single safetensors file."""
    if (ckpt_path / "model.safetensors.index.json").exists():
        import json as _json_loader

        with open(ckpt_path / "model.safetensors.index.json") as _f:
            index = _json_loader.load(_f)
        shard_files = set(index["weight_map"].values())
        full_state_dict = {}
        for shard_file in sorted(shard_files):
            shard_dict = _patched_loader(str(ckpt_path / shard_file))
            for k in skip_keys:
                shard_dict.pop(k, None)
            full_state_dict.update(shard_dict)
        return full_state_dict
    elif (ckpt_path / "model.safetensors").exists():
        state_dict = _patched_loader(str(ckpt_path / "model.safetensors"))
        for k in skip_keys:
            state_dict.pop(k, None)
        return state_dict
    else:
        raise FileNotFoundError(f"No model weights found in {ckpt_path}")


def _load_state_dict_with_shape_filter(model, state_dict: dict, skip_keys: set):
    """Load state dict into model, gracefully handling shape mismatches for architecture changes."""
    model_state = model.state_dict()
    filtered = {}
    shape_mismatched = []

    for key, value in state_dict.items():
        if key in skip_keys:
            continue
        if key in model_state:
            if value.shape == model_state[key].shape:
                filtered[key] = value
            else:
                shape_mismatched.append(
                    f"  {key}: checkpoint={list(value.shape)} vs model={list(model_state[key].shape)}"
                )

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    missing = [k for k in missing if k not in skip_keys]
    return missing, unexpected, shape_mismatched


def _reload_base_layer_weights_for_lora(model, ckpt_path: Path, skip_keys: set) -> int:
    """Restore the base-layer weights of PEFT-wrapped Linear modules — Issue #54.

    PEFT's ``inject_adapter_in_model`` wraps ``foo.q_proj`` as a ``LoraLinear``
    and moves the original Linear to ``foo.q_proj.base_layer``. A pre-LoRA
    checkpoint stores the original weight at ``foo.q_proj.weight``, but the
    HF ``from_pretrained`` / ``load_state_dict`` key matcher now expects
    ``foo.q_proj.base_layer.weight`` — so those tensors are silently NOT
    loaded and the wrapped Linear ends up at its random PyTorch init.

    Symptoms (from upstream PR #23 / #15):
      * ``train_flow_matching_loss`` either NaN at step 1 or stuck at ~1.35
        instead of the expected ~0.40 — silently wrong, no exception.

    This function walks the safetensors shards of the checkpoint and for
    every ``X.weight`` / ``X.bias`` key whose ``X.base_layer.<suffix>``
    exists in the model (= a LoRA-wrapped Linear), copies the checkpoint
    value into that base_layer tensor.

    Returns:
        Number of base-layer tensors fixed up. ``0`` is the no-op case
        (no LoRA wrapping in the model, or no matching keys in the ckpt).
    """
    try:
        from safetensors.torch import load_file as _safe_load_file
    except ImportError:
        return 0

    files = []
    if (ckpt_path / "model.safetensors.index.json").exists():
        with open(ckpt_path / "model.safetensors.index.json") as _f:
            _index = json.load(_f)
        files = [ckpt_path / s for s in sorted(set(_index["weight_map"].values()))]
    elif (ckpt_path / "model.safetensors").exists():
        files = [ckpt_path / "model.safetensors"]
    else:
        return 0

    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())
    fixed = 0
    with torch.no_grad():
        for f in files:
            shard = _safe_load_file(str(f), device="cpu")
            for ckpt_key, tensor in shard.items():
                if ckpt_key in skip_keys:
                    continue
                if ckpt_key in model_keys:
                    continue  # already loaded by the main load
                # Try the base_layer-inserted variant: "X.weight" → "X.base_layer.weight"
                for suffix in (".weight", ".bias"):
                    if ckpt_key.endswith(suffix):
                        remapped = ckpt_key[: -len(suffix)] + ".base_layer" + suffix
                        if remapped in model_keys:
                            target = model_sd[remapped]
                            if tensor.shape == target.shape:
                                target.data.copy_(tensor.to(target.device, dtype=target.dtype))
                                fixed += 1
                        break
    return fixed


def convert_tensors_to_lists(obj):
    """Recursively convert tensors to lists in nested dictionaries/lists."""
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_tensors_to_lists(item) for item in obj]
    else:
        return obj


class RLDXPipeline(ModelPipeline):
    model_class = RLDX
    processor_class = RLDXProcessor

    def __init__(self, config: Config, save_cfg_dir: Path):
        super().__init__(config)
        self.save_cfg_dir = save_cfg_dir

        # Build transformers loading kwargs from training config
        transformers_loading_kwargs = {
            "trust_remote_code": self.config.training.transformers_trust_remote_code,
            "local_files_only": self.config.training.transformers_local_files_only,
        }
        if self.model_config.model_revision is not None:
            transformers_loading_kwargs["revision"] = self.model_config.model_revision
        if self.config.training.transformers_cache_dir is not None:
            transformers_loading_kwargs["cache_dir"] = self.config.training.transformers_cache_dir
        if self.config.training.transformers_access_token is not None:
            transformers_loading_kwargs["token"] = self.config.training.transformers_access_token

        self.transformers_loading_kwargs = transformers_loading_kwargs

    @property
    def model_config(self):
        return self.config.model

    def setup(self):
        self.model = self._create_model()
        self.train_dataset, self.eval_dataset = self._create_dataset(self.save_cfg_dir)
        self.data_collator = self._create_collator()

    def _create_model(self):
        """Setup model with proper vocabulary expansion."""
        # Inject physics_delta_indices from data config into model config (before model creation)
        if getattr(self.config.model, "use_physics", False):
            physics_keys = getattr(self.config.model, "physics_keys", [])
            for emb_tag, modalities in self.config.data.modality_configs.items():
                if not isinstance(modalities, dict):
                    continue
                for pk in physics_keys:
                    if pk in modalities:
                        mc = modalities[pk]
                        delta = getattr(mc, "delta_indices", None)
                        if delta is None and isinstance(mc, dict):
                            delta = mc.get("delta_indices", [])
                        if delta:
                            self.config.model.physics_delta_indices = delta
                            _print(
                                f"[Physics] Injected physics_delta_indices from '{pk}': {len(delta)} frames"
                            )
                            break
                else:
                    continue
                break

        if self.config.training.start_from_checkpoint is not None:
            ckpt_path = Path(self.config.training.start_from_checkpoint)
            _print(f"\n[i] Loading checkpoint from {ckpt_path}")

            # Patched loader: skip complex tensors that safetensors can't handle
            _orig = modeling_utils.load_state_dict

            def _patched(checkpoint_file, *args, **kwargs):
                if str(checkpoint_file).endswith(".safetensors"):
                    tensors = {}
                    with safe_open(checkpoint_file, framework="pt") as f:
                        for key in f.keys():
                            t = f.get_tensor(key)
                            if t.is_complex():
                                continue
                            tensors[key] = t.contiguous()
                    return tensors
                return _orig(checkpoint_file, *args, **kwargs)

            modeling_utils.load_state_dict = _patched

            # 1. Create model from runtime config (skip backbone pretrained weights)
            loading_kwargs = {**self.transformers_loading_kwargs, "skip_pretrained_weights": True}
            model_cls = self.model_class
            model = model_cls(self.config.model, transformers_loading_kwargs=loading_kwargs)

            # 2. Load checkpoint weights once
            state_dict = _load_checkpoint_state_dict(ckpt_path, _patched, _COMPLEX_TENSOR_SKIP_KEYS)

            # Remap legacy physics keys (physics_encoder→physics_cond_encoder→physics.physics_cond_encoder)
            from rldx.model.modules.action_model.physics_head import remap_physics_keys

            state_dict = remap_physics_keys(state_dict)

            # 3. Apply weights with shape filtering
            missing_keys, unexpected_keys, shape_mismatched = _load_state_dict_with_shape_filter(
                model, state_dict, _COMPLEX_TENSOR_SKIP_KEYS
            )

            if shape_mismatched:
                _print(
                    colored(
                        f"[i] Shape-mismatched keys (re-initialized, {len(shape_mismatched)}):",
                        "yellow",
                    )
                )
                for msg in shape_mismatched:
                    _print(colored(msg, "yellow"))

            if missing_keys:
                _print(
                    colored(
                        f"[i] Missing keys (newly added modules, {len(missing_keys)}):", "yellow"
                    )
                )
                for k in missing_keys:
                    _print(colored(f"  {k}", "yellow"))

            if unexpected_keys:
                _print(
                    colored(
                        f"[w] Unexpected keys from checkpoint ({len(unexpected_keys)}):", "yellow"
                    )
                )
                for k in unexpected_keys[:20]:
                    _print(colored(f"  {k}", "yellow"))
                if len(unexpected_keys) > 20:
                    _print(colored(f"  ... and {len(unexpected_keys) - 20} more", "yellow"))

            # Initialize mask_token if missing
            mask_token_missing = any("mask_token" in key for key in missing_keys)
            if mask_token_missing and model.action_model.mask_token is not None:
                with torch.no_grad():
                    model.action_model.mask_token.data.copy_(
                        0.02 * torch.randn_like(model.action_model.mask_token)
                    )
                _print("mask_token not in checkpoint - initialized")

            # Issue #54/#55 — fix up the base-layer weights of PEFT-wrapped
            # Linear modules so the ``X.weight`` keys in a pre-LoRA checkpoint
            # actually land on ``X.base_layer.weight``. Without this the
            # wrapped Linear silently keeps its random PyTorch init and
            # training loss is wrong (NaN for backbone LoRA, ~1.35 for
            # action-model LoRA).
            uses_lora = getattr(self.config.model, "action_model_use_lora", False) or getattr(
                self.config.model, "backbone_use_lora", False
            )
            if uses_lora:
                n_fixed = _reload_base_layer_weights_for_lora(
                    model, ckpt_path, _COMPLEX_TENSOR_SKIP_KEYS
                )
                _print(
                    colored(
                        f"[LoRA] Re-loaded {n_fixed} base-layer weight tensors "
                        f"from checkpoint (would otherwise be random-init).",
                        "green" if n_fixed > 0 else "yellow",
                    )
                )
                # Drop remapped base_layer keys from missing_keys so the
                # new-param-warmup callback doesn't treat them as new.
                missing_keys = [k for k in missing_keys if ".base_layer." not in k]

            _print(
                colored(
                    f"\n[i] Checkpoint loaded: {len(missing_keys)} missing, "
                    f"{len(unexpected_keys)} unexpected, {len(shape_mismatched)} shape-mismatched",
                    "green",
                )
            )

            model._new_param_names = set(missing_keys)
            modeling_utils.load_state_dict = _orig

        else:
            # From scratch: backbone loads pretrained weights from HuggingFace
            model_cls = self.model_class
            model = model_cls(
                self.config.model, transformers_loading_kwargs=self.transformers_loading_kwargs
            )

        # Override old arguments for post-training
        model.config.general_embodiment_train_ratio = (
            self.config.model.general_embodiment_train_ratio
        )

        # Sync physics config for correct serialization (config.json)
        for _key in ("use_physics", "physics_delta_indices", "physics_use_flow_matching"):
            if hasattr(self.config.model, _key):
                setattr(model.config, _key, getattr(self.config.model, _key))

        _print(colored(f"\nModel Config: {model.config}", "yellow"))
        if get_rank() == 0:
            with open(self.save_cfg_dir / "final_model_config.json", "w") as f:
                f.write(model.config.to_filtered_json())
        # Print parameter statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _print(f"\n[i] Total parameters: {total_params:,}")
        _print(
            f"[i] Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )
        _print("Model: ", model)

        return model

    def _get_statistics(self) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None:
        return None

    def _get_embodiment_id_mapping(self) -> dict[str, int]:
        return None

    def _create_dataset(self, save_cfg_dir: Path):
        """Create appropriate dataset based on task and mode."""

        _use_pretrained_processor = self.config.training.start_from_checkpoint is not None

        if _use_pretrained_processor:
            processor = AutoProcessor.from_pretrained(
                Path(self.config.training.start_from_checkpoint) / "processor",
                # Overrides
                modality_configs=self.config.data.modality_configs,
                image_max_area=self.model_config.image_max_area,
                image_resize_m=self.model_config.image_resize_m,
                random_crop_fraction=self.model_config.random_crop_fraction,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                use_relative_action=self.model_config.use_relative_action,
                memory_length=getattr(self.model_config, "memory_length", 1)
                if getattr(self.model_config, "use_memory", False)
                else 1,
                general_embodiment_train_ratio=getattr(
                    self.model_config, "general_embodiment_train_ratio", 0
                ),
                conversation_image_first=getattr(
                    self.model_config, "conversation_image_first", False
                ),
                physics_keys=getattr(self.model_config, "physics_keys", None),
                physics_dims=getattr(self.model_config, "physics_dims", None),
                allow_missing_physics=getattr(self.model_config, "allow_missing_physics", False),
                **self.transformers_loading_kwargs,
            )
        else:
            processor = self.processor_class(
                modality_configs=self.config.data.modality_configs,
                statistics=self._get_statistics(),  # By default is None, so this will be computed and set later.
                use_percentiles=self.model_config.use_percentiles,
                embodiment_id_mapping=self._get_embodiment_id_mapping(),  # By default is None, so this will be set later.
                image_max_area=self.model_config.image_max_area,
                image_resize_m=self.model_config.image_resize_m,
                random_crop_fraction=self.model_config.random_crop_fraction,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                max_state_dim=self.model_config.max_state_dim,
                max_action_dim=self.model_config.max_action_dim,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                use_relative_action=self.model_config.use_relative_action,
                memory_length=getattr(self.model_config, "memory_length", 1)
                if getattr(self.model_config, "use_memory", False)
                else 1,
                general_embodiment_train_ratio=self.model_config.general_embodiment_train_ratio,
                conversation_image_first=getattr(
                    self.model_config, "conversation_image_first", False
                ),
                physics_keys=getattr(self.model_config, "physics_keys", None),
                physics_dims=getattr(self.model_config, "physics_dims", None),
                allow_missing_physics=getattr(self.model_config, "allow_missing_physics", False),
            )

        if get_rank() == 0:
            with open(self.save_cfg_dir / "final_processor_config.json", "w") as f:
                json.dump({k: str(v) for k, v in vars(processor).items()}, f, indent=2)

        self.processor = processor
        dataset_factory = DatasetFactory(config=self.config)
        train_dataset, eval_dataset = dataset_factory.build(processor=self.processor)

        # Save dataset statistics for inference
        stats = train_dataset.get_dataset_statistics()
        stats_dict = convert_tensors_to_lists(stats)
        # Save statistics
        with open(save_cfg_dir / "dataset_statistics.json", "w") as f:
            json.dump(stats_dict, f, indent=2)
        _print("Saved dataset statistics for inference")

        return train_dataset, eval_dataset

    def _create_collator(self):
        data_collator = self.processor.collator
        return data_collator


register_model(RLDXConfig, RLDXPipeline)
