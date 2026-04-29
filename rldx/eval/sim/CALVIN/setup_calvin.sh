#!/usr/bin/env bash
set -euxo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set paths relative to script location
CALVIN_REPO="$SCRIPT_DIR/../../../../external_dependencies/calvin"
PROJECT_REPO="$SCRIPT_DIR/../../../.."
CALVIN_UV_ENV="$SCRIPT_DIR/calvin_uv"

# init submodule only if not already populated
if [ ! -f "$CALVIN_REPO/install.sh" ]; then
    git submodule update --init $CALVIN_REPO
fi

rm -rf $CALVIN_UV_ENV
mkdir -p $CALVIN_UV_ENV
uv venv $CALVIN_UV_ENV/.venv --python 3.8
source $CALVIN_UV_ENV/.venv/bin/activate

uv pip install wheel cmake==3.18.4
# pyhash==0.9.3은 setuptools>=58에서 use_2to3 제거로 빌드 실패
# --no-build-isolation으로 venv의 setuptools<58 사용
uv pip install "setuptools<58"
uv pip install pyhash==0.9.3 --no-build-isolation
uv pip install tyro==1.0.5

uv pip install "numpy<1.24"  # np.int was removed in numpy 1.24; tacto/networkx use it
uv pip install -e $CALVIN_REPO/calvin_env/tacto
uv pip install -e $CALVIN_REPO/calvin_env
uv pip install -e $CALVIN_REPO/calvin_models
uv pip install -e $PROJECT_REPO/external_dependencies/openpi-client
