"""The native transport must remain usable in a model-free simulator."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("statement", [
    "import rldx",
    "from rldx.policy.server_client import PolicyClient, PolicyServer",
])
def test_import_does_not_load_model_stack(statement):
    result = subprocess.run(
        [sys.executable, "-c", f"""
import importlib.abc
import sys

class BlockModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'rldx.model' or fullname.split('.')[0] in (
            'transformers', 'albumentations', 'flash_attn', 'tyro'
        ):
            raise AssertionError('Client imported model dependency: ' + fullname)

sys.meta_path.insert(0, BlockModelImports())
{statement}
assert 'rldx.model' not in sys.modules
"""],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
