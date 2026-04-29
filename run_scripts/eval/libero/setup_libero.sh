#!/bin/bash
set -euo pipefail
export NO_ALBUMENTATIONS_UPDATE=1
BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
bash "$BASE_DIR/rldx/eval/sim/LIBERO/setup_libero.sh"
