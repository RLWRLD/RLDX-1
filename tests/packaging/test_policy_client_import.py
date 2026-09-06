"""The native transport must remain usable in a model-free simulator."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "statement",
    [
        "import rldx",
        "from rldx.policy.server_client import PolicyClient, PolicyServer",
        "from rldx import EmbodimentTag; assert EmbodimentTag is not None",
    ],
)
def test_import_does_not_load_model_stack(statement):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
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
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_unknown_public_attribute_raises():
    import rldx

    with pytest.raises(AttributeError, match="no attribute"):
        getattr(rldx, "not_a_public_rldx_symbol")


@pytest.mark.parametrize(
    "path,function",
    [
        ("rldx/policy/policy_loader.py", "_load_model"),
        ("rldx/policy/policy_loader.py", "_load_processor"),
        ("rldx/inference/utils/input_generator.py", "generate_synthetic_input"),
    ],
)
def test_direct_loading_helpers_register_before_hf_loading(path, function):
    """Static order guard complements the real full-model registration gate."""
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / path
    tree = ast.parse(source.read_text())
    body = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function
    )
    registration = [
        node.lineno
        for node in ast.walk(body)
        if isinstance(node, ast.Import) and any(alias.name == "rldx.model" for alias in node.names)
    ]
    loads = [
        node.lineno
        for node in ast.walk(body)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
    ]
    assert registration and loads and min(registration) < min(loads)


def test_model_registration_in_full_environment():
    """Opt-in model-stack gate; never treat lean-client tests as server coverage."""
    import os

    if os.environ.get("RLDX_TEST_MODEL_IMPORT") != "1":
        pytest.skip("Set RLDX_TEST_MODEL_IMPORT=1 in the full model environment")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import rldx.model
from transformers import AutoConfig
from transformers.models.auto.modeling_auto import MODEL_MAPPING
from transformers.models.auto.processing_auto import PROCESSOR_MAPPING
from rldx.model.core.rldx import RLDX
from rldx.model.core.processing_rldx import RLDXProcessor
config = AutoConfig.for_model('RLDX-1')
assert MODEL_MAPPING[type(config)] is RLDX
assert PROCESSOR_MAPPING[type(config)] is RLDXProcessor
""",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
