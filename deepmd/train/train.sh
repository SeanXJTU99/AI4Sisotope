#!/usr/bin/env bash
# Train DPA-2 force field with DP-LONG long-range electrostatic correction.
# Uses multi-GPU data-parallel training for the BF3 isotope effect system.
#
# Usage:
#   bash train.sh [num_gpus]
#
# Expected GPU-hours: ~48 on 4x V100 for 1M steps.

set -euo pipefail

NUM_GPUS="${1:-4}"
export OMP_NUM_THREADS=4
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0,1,2,3

CONFIG_FILE="input_dpa2.json"
LOG_FILE="train_dpa2.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

# ---- Validate config ----
if ! python -c "import json; json.load(open('${CONFIG_FILE}'))" 2>/dev/null; then
    log "ERROR: Invalid JSON in ${CONFIG_FILE}"
    exit 1
fi

# ---- Train ----
log "Starting DPA-2 training with ${NUM_GPUS} GPU(s)..."
log "Config: ${CONFIG_FILE}"

dp train "${CONFIG_FILE}" --mpi-log=master 2>&1 | tee -a "${LOG_FILE}"

# Check exit code
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "ERROR: Training failed. Check ${LOG_FILE} for details."
    exit 1
fi

log "Training completed successfully."
log "Learning curve data: lcurve.out"
log "Latest checkpoint: $(ls -t model.ckpt-*.pt 2>/dev/null | head -1 || echo 'checkpoint not found')"

# ---- Freeze model ----
log "Freezing model..."
dp freeze -o frozen_model.pb 2>&1 | tee -a "${LOG_FILE}"

if [ -f "frozen_model.pb" ]; then
    log "Model frozen: frozen_model.pb ($(du -h frozen_model.pb | cut -f1))"
else
    log "ERROR: Model freezing failed."
    exit 1
fi

log "All done. Ready for PIMD inference."
