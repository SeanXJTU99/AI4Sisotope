#!/usr/bin/env bash
# DP-GEN active learning pipeline for isotope effect force field training.
# Supports Slurm, PBS/Torque, and local execution.
#
# Usage:
#   bash run_dpgen.sh [slurm|pbs|local]
#
# Stages:
#   1. init_bulk  -- prepare initial training data
#   2. run         -- active learning loop (exploration -> labeling -> training)
#   3. auto_test   -- final validation

set -euo pipefail

SCHEDULER="${1:-local}"
PARAM_FILE="param.json"
MACHINE_FILE="machine.json"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ---- Stage 1: Initialize ----
log "Stage 1/3: Initializing training data..."
if [ -d "data/init" ]; then
    log "Initial data directory exists, skipping init_bulk."
else
    dpgen init_bulk ${PARAM_FILE} ${MACHINE_FILE}
    log "Initial data prepared."
fi

# ---- Stage 2: Active Learning ----
log "Stage 2/3: Running DP-GEN active learning loop..."
dpgen run ${PARAM_FILE} ${MACHINE_FILE}
log "Active learning loop completed."

# ---- Stage 3: Validation ----
log "Stage 3/3: Running final validation..."
dpgen auto_test ${PARAM_FILE} ${MACHINE_FILE}
log "DP-GEN pipeline finished successfully."
log "Trained models saved in: model_deviation/"
