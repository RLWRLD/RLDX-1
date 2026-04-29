#!/usr/bin/env bash
set -euxo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_REPO="$SCRIPT_DIR/../../../.."
SIMPLER_REPO="$PROJECT_REPO/external_dependencies/SimplerEnv"
UV_ENV="$SCRIPT_DIR/simpler_uv"

# SimplerEnv may already be populated (manual clone or external setup) even if
# the submodule pointer is not committed in HEAD of the parent project. The
# canonical layout requires SimplerEnv + its inner ManiSkill2_real2sim
# submodule to both be present and populated.
if [ -f "$SIMPLER_REPO/setup.py" ] || [ -f "$SIMPLER_REPO/pyproject.toml" ]; then
    echo "[setup_SimplerEnv] $SIMPLER_REPO already populated — checking inner submodules."
    # Ensure SimplerEnv's own submodules (ManiSkill2_real2sim) are initialized
    # even when SimplerEnv itself was a manual clone rather than a real
    # submodule of the RLDX repo.
    if [ ! -f "$SIMPLER_REPO/ManiSkill2_real2sim/setup.py" ] && \
       [ ! -f "$SIMPLER_REPO/ManiSkill2_real2sim/pyproject.toml" ]; then
        if [ -d "$SIMPLER_REPO/.git" ] || [ -f "$SIMPLER_REPO/.git" ]; then
            echo "[setup_SimplerEnv] initializing ManiSkill2_real2sim submodule inside $SIMPLER_REPO ..."
            (cd "$SIMPLER_REPO" && git submodule update --init --recursive)
        else
            echo "[setup_SimplerEnv] $SIMPLER_REPO is not a git repo and ManiSkill2_real2sim is empty —"
            echo "  cloning ManiSkill2_real2sim directly."
            git clone https://github.com/allenzren/ManiSkill2_real2sim \
                "$SIMPLER_REPO/ManiSkill2_real2sim"
        fi
    fi
else
    git submodule update --init --recursive "$SIMPLER_REPO"
fi

# Numpy pin: cluster uses 1.26.4; SimplerEnv README mentions 1.24.4 for pinocchio IK.
# Override by exporting SIMPLER_NUMPY=1.24.4 if needed.
SIMPLER_NUMPY="${SIMPLER_NUMPY:-1.26.4}"

# python -m pip install -U uv
rm -rf "$UV_ENV"
mkdir -p "$UV_ENV"
uv venv "$UV_ENV/.venv" --python 3.10
source "$UV_ENV/.venv/bin/activate"
# Pin setuptools <80: setuptools 80+ removed `pkg_resources`, which sapien's
# `renderer_config.py` still imports at module load time. The lower bound
# keeps modern build hooks; the upper bound restores `pkg_resources`.
uv pip install "setuptools>=68,<80"

# Core deps (match cluster’s pyproject pattern)
uv pip install \
  gymnasium==0.29.1 \
  json-numpy>=2.1.1 \
  numpy=="$SIMPLER_NUMPY" \
  opencv-python-headless==4.10.0.84 \
  ray==2.48.0

# Install SimplerEnv sources (editable)
# uv pip install -e "$SIMPLER_REPO/ManiSkill2_real2sim" --config-settings editable_mode=compat
# uv pip install -e "$SIMPLER_REPO" --config-settings editable_mode=compat

uv pip install -e "$SIMPLER_REPO/ManiSkill2_real2sim"
uv pip install -e "$SIMPLER_REPO"

# Make your OSS project importable
uv pip install --editable "$PROJECT_REPO" --no-deps

uv pip install tianshou==0.5.1 pydantic av zmq torchvision==0.22.0 transformers==4.51.3

# Sanity check
python - <<'PY'
from rldx.eval.sim.SimplerEnv.simpler_env import register_simpler_envs
register_simpler_envs()
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
print("SimplerEnv import OK")
import gymnasium as gym
env = gym.make("simpler_env_google/google_robot_pick_object")
env.reset()
env.close()
print("Env OK:", type(env))
PY

echo "SimplerEnv ready at: $UV_ENV/.venv/bin/python3"