#!/bin/bash
# Run trace-based load simulation

# Default values
TRACE_FILE="${1:-data/BurstGPT_1.csv}"
DURATION_HOURS="${2:-24}"
TIME_SCALE="${3:-1.0}"
LOG_NAME="${4:-$(basename $TRACE_FILE .csv)_$(date +%Y%m%d_%H%M%S)}"

# Ensure results directory exists
mkdir -p results

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hybrid_inference

# Run simulation
echo "Starting simulation:"
echo "  Trace: $TRACE_FILE"
echo "  Duration: ${DURATION_HOURS}h"
echo "  Time scale: ${TIME_SCALE}x"
echo "  Log: results/${LOG_NAME}.log"

cd /root/hybridInference
PYTHONPATH=/root/hybridInference:$PYTHONPATH nohup python -u simulator/run_trace.py "$TRACE_FILE" "$DURATION_HOURS" "$TIME_SCALE" \
    > "results/${LOG_NAME}.log" 2>&1 &

PID=$!
echo "Started simulation with PID: $PID"
echo "Monitor with: tail -f results/${LOG_NAME}.log"