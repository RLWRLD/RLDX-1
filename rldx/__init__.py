"""RLDX — RLWRLD RLDX-1 Vision-Language-Action model.

The package's public symbols are exposed via lazy attribute access so that
``import rldx`` (or ``from rldx.data.embodiment_tags import EmbodimentTag``)
does not eagerly pull in the heavy training-side dependencies (torch,
albumentations, transformers, ...). Each attribute is loaded on first
access. Loading a model attribute (``rldx.RLDX``, ``rldx.RLDXConfig``,
``rldx.RLDXProcessor``) is what registers the model with HuggingFace's
``AutoConfig`` / ``AutoModel`` / ``AutoProcessor`` registries, so::

    import rldx

    rldx.RLDXConfig  # triggers AutoConfig.register
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained("RLWRLD/RLDX-1-PT")

works out of the box.
"""

from importlib import import_module
from typing import TYPE_CHECKING


__version__ = "0.1.0"

_LAZY_ATTRS = {
    "Config": ("rldx.configs.base_config", "Config"),
    "RLDXConfig": ("rldx.configs.model.rldx", "RLDXConfig"),
    "TrainConfig": ("rldx.configs.train_config", "TrainConfig"),
    "EmbodimentTag": ("rldx.data.embodiment_tags", "EmbodimentTag"),
    "RLDXProcessor": ("rldx.model.core.processing_rldx", "RLDXProcessor"),
    "RLDX": ("rldx.model.core.rldx", "RLDX"),
    "RLDXPipeline": ("rldx.model.core.setup", "RLDXPipeline"),
    "RLDXPolicy": ("rldx.policy.rldx_policy", "RLDXPolicy"),
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_path, attr = _LAZY_ATTRS[name]
        value = getattr(import_module(module_path), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'rldx' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))


__all__ = ["__version__", *sorted(_LAZY_ATTRS)]


if TYPE_CHECKING:
    from rldx.configs.base_config import Config
    from rldx.configs.model.rldx import RLDXConfig
    from rldx.configs.train_config import TrainConfig
    from rldx.data.embodiment_tags import EmbodimentTag
    from rldx.model.core.processing_rldx import RLDXProcessor
    from rldx.model.core.rldx import RLDX
    from rldx.model.core.setup import RLDXPipeline
    from rldx.policy.rldx_policy import RLDXPolicy
