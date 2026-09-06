"""RLDX — RLWRLD RLDX-1 Vision-Language-Action model.

Importing ``rldx.model`` registers RLDX-1 with HuggingFace's ``AutoConfig``
/ ``AutoModel`` / ``AutoProcessor`` registries::

    import rldx.model
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained("RLWRLD/RLDX-1-PT")

The public symbols are also exposed as lazy attributes (``rldx.RLDX``,
``rldx.RLDXConfig``, ``rldx.RLDXProcessor``, ...) so that callers can
reach the underlying classes without sub-module knowledge.

Keep package initialization model-free: importing a policy client in a
simulation process must not require the training/model dependency stack.
"""

from importlib import import_module
from typing import TYPE_CHECKING


__version__ = "1.0.1"

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
