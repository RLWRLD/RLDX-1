#!/bin/bash
set -euo pipefail
export NO_ALBUMENTATIONS_UPDATE=1

# Set up the SimplerEnv venv for RLDX (Google robot tasks).
# This delegates to the canonical setup script under rldx/eval/sim/SimplerEnv/,
# which handles Google-VM and Google-VA environments identically.
bash rldx/eval/sim/SimplerEnv/setup_SimplerEnv.sh
