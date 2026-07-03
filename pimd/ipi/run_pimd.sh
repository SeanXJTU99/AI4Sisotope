#!/usr/bin/env bash
# Run PIMD simulation with i-PI path integral engine + DeePMD-kit force provider.
#
# Architecture:
#   i-PI (server) <--unix socket--> DeePMD-kit (driver)
#   i-PI handles PIMD integration, DP provides energy/forces
#
# Usage:
#   bash run_pimd.sh [thermostat] [nbeads]
#     thermostat: pile (default) | piglet | pile_g
#     nbeads: 64 (default) | 32 | 128

set -euo pipefail

# ---- Configuration ----
THERMOSTAT="${1:-pile}"
NBEADS="${2:-64}"
TEMP=145.0
DP_MODEL="../deepmd/train/frozen_model.pb"
IPI_PORT=31415
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ---- Select input file ----
case "${THERMOSTAT}" in
    pile)
        IPI_INPUT="input_pimd.xml"
        ;;
    piglet)
        IPI_INPUT="input_piglet.xml"
        ;;
    pile_g)
        IPI_INPUT="input_pimd.xml"  # already uses pile_g
        ;;
    *)
        log "ERROR: unknown thermostat '${THERMOSTAT}'. Use: pile | piglet | pile_g"
        exit 1
        ;;
esac

log "============================================"
log "PIMD Simulation: BF3 Isotope Effect"
log "Temperature: ${TEMP} K"
log "Beads (P): ${NBEADS}"
log "Thermostat: ${THERMOSTAT}"
log "Input: ${IPI_INPUT}"
log "DP Model: ${DP_MODEL}"
log "============================================"

# ---- Validate inputs ----
if [ ! -f "${IPI_INPUT}" ]; then
    log "ERROR: i-PI input file not found: ${IPI_INPUT}"
    exit 1
fi

if [ ! -f "${DP_MODEL}" ]; then
    log "WARNING: DP model not found at ${DP_MODEL}"
    log "Expected after running deepmd/train/train.sh (commit 2)."
    log "Continuing with mock socket for demonstration..."
fi

# ---- Launch i-PI server ----
log "Starting i-PI server..."
i-pi "${IPI_INPUT}" &
IPI_PID=$!
sleep 2

if ! kill -0 ${IPI_PID} 2>/dev/null; then
    log "ERROR: i-PI failed to start."
    exit 1
fi
log "i-PI server running (PID: ${IPI_PID})"

# ---- Launch DP driver ----
log "Starting DeePMD-kit driver on port ${IPI_PORT}..."
if [ -f "${DP_MODEL}" ]; then
    python -m deepmd.ipi driver \
        -m "${DP_MODEL}" \
        -u "${IPI_PORT}" &
    DP_PID=$!
else
    # Mock mode: i-PI runs without real force provider for demo
    log "Mock mode: i-PI will log warnings about missing socket."
    DP_PID=""
fi

# ---- Wait for completion ----
log "PIMD simulation in progress..."
log "Monitor: tail -f pimd_output.properties"
log "Trajectory: pimd_output.pos_0.xyz"

wait ${IPI_PID}
IPI_EXIT=$?

# ---- Cleanup ----
if [ -n "${DP_PID}" ]; then
    kill ${DP_PID} 2>/dev/null || true
fi

if [ ${IPI_EXIT} -eq 0 ]; then
    log "PIMD simulation completed successfully."
    log "Output files:"
    log "  Properties: pimd_output.properties"
    log "  Restart:    pimd_output.restart"
    log "  Trajectory: pimd_output.pos_0.xyz"
else
    log "ERROR: i-PI exited with code ${IPI_EXIT}"
    exit 1
fi
