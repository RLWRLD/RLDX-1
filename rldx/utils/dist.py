import builtins
import contextlib
import os


try:
    import torch

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _env_rank(default=0):
    return int(os.environ.get("RANK", str(default)))


def is_dist_initialized() -> bool:
    return _HAS_TORCH and torch.distributed.is_available() and torch.distributed.is_initialized()


def get_global_rank() -> int:
    if is_dist_initialized():
        with contextlib.suppress(Exception):
            return torch.distributed.get_rank()
    return _env_rank(0)


def is_global_zero() -> bool:
    return get_global_rank() == 0


def rank_zero_print(*args, force: bool = False, **kwargs):
    """Print only on global rank 0 (or if force=True)."""
    if force or is_global_zero():
        return builtins.print(*args, **kwargs)
