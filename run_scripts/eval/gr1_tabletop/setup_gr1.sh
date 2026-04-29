#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../../.."

bash rldx/eval/sim/robocasa-gr1-tabletop-tasks/setup_RoboCasaGR1TabletopTasks.sh

#NOTE(MK): Disable flash_attn -> TODO: Resolve flash_attn compilation error
SITE=rldx/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/lib/python3.10/site-packages
mv $SITE/flash_attn $SITE/flash_attn__disabled
mv $SITE/flash_attn_2_cuda*.so $SITE/flash_attn_2_cuda__disabled.so 2>/dev/null || true
mv $SITE/flash_attn_cuda*.so $SITE/flash_attn_cuda__disabled.so 2>/dev/null || true

# Apply determinism patches to robocasa submodule
PATCH_DIR="$SCRIPT_DIR/patches"
ROBOCASA_DIR="$PROJECT_ROOT/external_dependencies/robocasa-gr1-tabletop-tasks"

echo "Applying patches to robocasa-gr1-tabletop-tasks..."
for patch in "$PATCH_DIR"/*.patch; do
    [ -f "$patch" ] || continue
    if git -C "$ROBOCASA_DIR" apply --check "$patch" 2>/dev/null; then
        git -C "$ROBOCASA_DIR" apply "$patch"
        echo "  Applied: $(basename "$patch")"
    else
        echo "  Skipped (already applied or conflict): $(basename "$patch")"
    fi
done