#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"

bash "$PROJECT_REPO/rldx/eval/sim/robocasa365/setup_RoboCasa365.sh"

#NOTE(MK): Disable flash_attn -> TODO: Resolve flash_attn compilation error
SITE="$PROJECT_REPO/rldx/eval/sim/robocasa365/robocasa365_uv/.venv/lib/python3.10/site-packages"
mv "$SITE/flash_attn" "$SITE/flash_attn__disabled" 2>/dev/null || true
mv "$SITE"/flash_attn_2_cuda*.so "$SITE/flash_attn_2_cuda__disabled.so" 2>/dev/null || true
mv "$SITE"/flash_attn_cuda*.so "$SITE/flash_attn_cuda__disabled.so" 2>/dev/null || true
