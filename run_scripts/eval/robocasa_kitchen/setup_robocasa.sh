#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../../.."

bash rldx/eval/sim/robocasa/setup_RoboCasa.sh

#NOTE(MK): Disable flash_attn -> TODO: Resolve flash_attn compilation error
SITE=rldx/eval/sim/robocasa/robocasa_uv/.venv/lib/python3.10/site-packages
mv $SITE/flash_attn $SITE/flash_attn__disabled 2>/dev/null || true
mv $SITE/flash_attn_2_cuda*.so $SITE/flash_attn_2_cuda__disabled.so 2>/dev/null || true
mv $SITE/flash_attn_cuda*.so $SITE/flash_attn_cuda__disabled.so 2>/dev/null || true

# Apply seed-clamp patch to robocasa submodule. Necessary because gymnasium
# 0.29's SyncVectorEnv forwards 64-bit seeds that numpy's legacy seeding
# rejects with ``Seed must be between 0 and 2**32 - 1``.
PATCH_DIR="$SCRIPT_DIR/patches"
ROBOCASA_DIR="$PROJECT_ROOT/external_dependencies/robocasa"

if [ -d "$PATCH_DIR" ]; then
    echo "Applying patches to robocasa..."
    for patch in "$PATCH_DIR"/*.patch; do
        [ -f "$patch" ] || continue
        if git -C "$ROBOCASA_DIR" apply --check "$patch" 2>/dev/null; then
            git -C "$ROBOCASA_DIR" apply "$patch"
            echo "  Applied: $(basename "$patch")"
        else
            echo "  Skipped (already applied or conflict): $(basename "$patch")"
        fi
    done
fi
