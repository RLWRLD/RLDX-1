#!/usr/bin/env bash
set -euxo pipefail

# Where this script lives (put it inside your repo)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_REPO="$SCRIPT_DIR/../../../.."
ROBOCASA365_REPO="$PROJECT_REPO/external_dependencies/robocasa365"
UV_ENV="$SCRIPT_DIR/robocasa365_uv"

git submodule update --init "$ROBOCASA365_REPO"

# Build helpers
# python -m pip install cmake==3.18.4
rm -rf "$UV_ENV"
mkdir -p "$UV_ENV"
uv venv "$UV_ENV/.venv" --python 3.10
source "$UV_ENV/.venv/bin/activate"
uv pip install setuptools wheel

# Core deps
uv pip install torch==2.5.1 torchvision==0.20.1
# Linux-only: preinstall flash-attn to avoid compiling inside other wheels
INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN:-1}
if [[ "$(uname -s)" == "Linux" && "$INSTALL_FLASH_ATTN" == "1" ]]; then
  uv pip install --no-build-isolation flash-attn==2.7.4.post1 || echo "flash-attn install skipped/failed; continuing"
fi

# Sim stack
uv pip install "git+https://github.com/ARISE-Initiative/robosuite.git@master"
uv pip install -e "$ROBOCASA365_REPO" --config-settings editable_mode=compat

# Make your project importable in this venv without re-resolving deps
uv pip install --editable "$PROJECT_REPO" --no-deps

# Assets and env/macros setup for RoboCasa365
printf 'y\n' | python "$ROBOCASA365_REPO/robocasa/scripts/download_kitchen_assets.py"
python "$ROBOCASA365_REPO/robocasa/scripts/setup_macros.py"

# Sanity import & env construction
python - <<'PY'
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import gymnasium as gym, robocasa, robosuite
print("Imports OK:", robosuite.__version__)
env = gym.make("robocasa/PickPlaceCounterToCabinet", split="pretrain", seed=0)
_ = env.reset()
env.close()
print("Env OK:", type(env))
PY
