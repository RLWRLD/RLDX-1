#!/usr/bin/env bash
# LIBERO-Plus canonical setup: clones the LIBERO-plus repo and strips
# the task-asset zip into rldx-side paths.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_REPO="$SCRIPT_DIR/../../../.."
LIBERO_PLUS_REPO="$PROJECT_REPO/external_dependencies/LIBERO-plus"
LIBERO_PLUS_UV_ENV="$SCRIPT_DIR/libero_plus_uv"

# ImageMagick shared libs live under miniconda; wand (imported transitively by
# libero.libero.envs) fails to import without this on our cluster.
export LD_LIBRARY_PATH="$HOME/miniconda3/lib:${LD_LIBRARY_PATH:-}"

# =========================================
# 1. Clone LIBERO-Plus repository
# =========================================
if [ ! -d "$LIBERO_PLUS_REPO" ]; then
    echo "[1/4] Cloning LIBERO-Plus repository..."
    mkdir -p "$PROJECT_REPO/external_dependencies"
    git clone https://github.com/sylvestf/LIBERO-plus.git "$LIBERO_PLUS_REPO"
else
    echo "[1/4] LIBERO-Plus repository already exists (skipped)"
fi

# =========================================
# 2. Create virtual environment
# =========================================
if [ -d "$LIBERO_PLUS_UV_ENV/.venv" ] && [ -f "$LIBERO_PLUS_UV_ENV/.venv/bin/python" ]; then
    echo "[2/4] LIBERO-Plus venv already exists (skipped)"
    echo "      To recreate, delete $LIBERO_PLUS_UV_ENV and re-run."
else
    echo "[2/4] Setting up LIBERO-Plus virtual environment..."
    rm -rf "$LIBERO_PLUS_UV_ENV"
    mkdir -p "$LIBERO_PLUS_UV_ENV"
    uv venv "$LIBERO_PLUS_UV_ENV/.venv" --python 3.10
    source "$LIBERO_PLUS_UV_ENV/.venv/bin/activate"

    # Install LIBERO-Plus requirements + package
    uv pip install --requirements "$LIBERO_PLUS_REPO/requirements.txt"
    uv pip install -e "$LIBERO_PLUS_REPO" --config-settings editable_mode=compat

    # Extra requirements for perturbation features
    if [ -f "$LIBERO_PLUS_REPO/extra_requirements.txt" ]; then
        uv pip install --requirements "$LIBERO_PLUS_REPO/extra_requirements.txt" || true
    fi

    # Install RLDX project (no-deps to avoid conflicts)
    uv pip install --editable "$PROJECT_REPO" --no-deps

    # Align key dependencies with rest of pipeline
    uv pip install \
        torch==2.5.1 torchvision==0.20.1 \
        pydantic av tianshou==0.5.1 tyro pandas dm_tree \
        einops==0.8.1 albumentations==1.4.18 zmq \
        transformers==4.51.3 \
        msgpack==1.1.0 msgpack-numpy==0.4.8 \
        gymnasium==0.29.1 \
        numpy==1.26.4

    uv pip install --editable "$PROJECT_REPO" --no-deps

    echo "  Venv setup complete."
fi

# Create LIBERO config pointing to LIBERO-Plus paths
PLUS_CONFIG_DIR="$LIBERO_PLUS_REPO/.libero_config"
mkdir -p "$PLUS_CONFIG_DIR"
cat > "$PLUS_CONFIG_DIR/config.yaml" << EOF
assets: $LIBERO_PLUS_REPO/libero/libero/./assets
bddl_files: $LIBERO_PLUS_REPO/libero/libero/./bddl_files
benchmark_root: $LIBERO_PLUS_REPO/libero/libero
datasets: $LIBERO_PLUS_REPO/libero/libero/../datasets
init_states: $LIBERO_PLUS_REPO/libero/libero/./init_files
EOF
export LIBERO_CONFIG_PATH="$PLUS_CONFIG_DIR"

# =========================================
# 3. Download + extract assets from HuggingFace
#
# NOTE: the upstream zip ships with an internal prefix of the form
#   inspire/hdd/project/.../LIBERO-plus-0/assets/...
# so we cannot just ``unzip -d .`` into ``libero/libero/`` — we extract while
# stripping the prefix so files land directly under ``libero/libero/assets/``.
#
# The LIBERO-plus git repo itself ships with a subset of ``assets/`` (e.g.
# articulated_objects, scenes) but is missing ``stable_scanned_objects/``,
# ``textures/``, ``turbosquid_objects/``, ``wall.xml`` — checking for one of
# those zip-only dirs tells us whether we still need to extract.
# =========================================
ASSETS_DIR="$LIBERO_PLUS_REPO/libero/libero/assets"
ASSETS_ZIP="$LIBERO_PLUS_REPO/libero_plus_data/assets.zip"
if [ -d "$ASSETS_DIR/stable_scanned_objects" ] && [ -d "$ASSETS_DIR/turbosquid_objects" ] && [ -d "$ASSETS_DIR/textures" ] && [ -f "$ASSETS_DIR/wall.xml" ]; then
    echo "[3/4] LIBERO-Plus assets (zip contents) already present (skipped)"
else
    echo "[3/4] Preparing LIBERO-Plus assets..."
    source "$LIBERO_PLUS_UV_ENV/.venv/bin/activate"
    uv pip install huggingface_hub 2>/dev/null || true

    if [ ! -f "$ASSETS_ZIP" ]; then
        echo "  Downloading assets.zip from HuggingFace (~6.4 GB)..."
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Sylvest/LIBERO-plus',
    repo_type='dataset',
    local_dir='$LIBERO_PLUS_REPO/libero_plus_data',
    allow_patterns=['assets.zip'],
)
print('Download complete.')
" || { echo "[ERROR] HuggingFace download failed."; exit 1; }
    else
        echo "  Reusing existing assets.zip at $ASSETS_ZIP"
    fi

    echo "  Extracting assets.zip into $ASSETS_DIR (stripping internal prefix)..."
    mkdir -p "$ASSETS_DIR"
    python - "$ASSETS_ZIP" "$ASSETS_DIR" <<'PYEOF'
import os, sys, zipfile

zip_path = sys.argv[1]
dest_dir = sys.argv[2]

with zipfile.ZipFile(zip_path) as zf:
    # Auto-detect the internal prefix. The zip is expected to contain entries
    # like '<prefix>/assets/<subdir>/...'; we strip everything up to and
    # including 'assets/' so contents land directly under dest_dir.
    prefix = None
    for name in zf.namelist():
        idx = name.find("/assets/")
        if idx != -1:
            prefix = name[: idx + len("/assets/")]
            break
    if prefix is None:
        # Fallback: assume files are already rooted under 'assets/'
        prefix = "assets/"
    print(f"  Stripping zip prefix: {prefix!r}")

    total = sum(1 for n in zf.namelist() if n.startswith(prefix))
    done = 0
    for member in zf.infolist():
        if not member.filename.startswith(prefix):
            continue
        rel = member.filename[len(prefix):]
        if not rel:
            continue
        target = os.path.join(dest_dir, rel)
        if member.is_dir():
            os.makedirs(target, exist_ok=True)
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with zf.open(member) as src, open(target, "wb") as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                dst.write(chunk)
        done += 1
        if done % 20000 == 0:
            print(f"  ... {done}/{total} files")
    print(f"  Extraction complete ({done} files).")
PYEOF

    # Basic sanity: the four zip-only items must now exist.
    for req in stable_scanned_objects turbosquid_objects textures wall.xml; do
        if [ ! -e "$ASSETS_DIR/$req" ]; then
            echo "[ERROR] Expected $ASSETS_DIR/$req after extraction — check zip integrity."
            exit 1
        fi
    done
fi

# =========================================
# 4. Sanity check
# =========================================
echo "[4/4] Running sanity check..."
source "$LIBERO_PLUS_UV_ENV/.venv/bin/activate"
rm -rf "$HOME/.libero"
echo "y" | python -c "from libero.libero import get_libero_path; get_libero_path('bddl_files')" 2>/dev/null || true
python - <<'PY'
from libero.libero import benchmark

benchmark_dict = benchmark.get_benchmark_dict()
print("  Available suites:", list(benchmark_dict.keys()))
for suite_name in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    if suite_name in benchmark_dict:
        suite = benchmark_dict[suite_name]()
        print(f"  [OK] {suite_name}: {suite.get_num_tasks()} tasks")
    else:
        print(f"  [SKIP] {suite_name}: not found")
print("")
print("  Setup complete! Ready for evaluation.")
PY

echo ""
echo "================================================"
echo "LIBERO-Plus evaluation setup complete!"
echo ""
echo "Usage:"
echo "  sbatch run_scripts/eval/libero_plus/eval_libero_plus.sh <ckpt_name>"
echo "================================================"
