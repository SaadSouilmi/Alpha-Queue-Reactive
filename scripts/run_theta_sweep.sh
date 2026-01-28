#!/bin/bash

# Theta sweep for alpha consumption calibration
# Runs simulations in parallel with different theta values

# Defaults
TICKER="PFE"
MAX_JOBS=20

# Parse flags
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--ticker)
            TICKER="$2"
            shift 2
            ;;
        -j|--jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "  -t, --ticker TICKER   Ticker symbol (default: PFE)"
            echo "  -j, --jobs N          Max parallel jobs (default: 20)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

BINARY="./cmake-build-debug/sample_alpha"

# Theta values to test (fine grid 0 to 0.15)
THETAS=(0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.11 0.12 0.13 0.14 0.15)

echo "Running theta sweep for $TICKER"
echo "Max parallel jobs: $MAX_JOBS"
echo "Theta values: ${THETAS[*]}"
echo ""

# Run baseline without race first
echo "Starting baseline (no race)"
$BINARY "$TICKER" &

# Run in parallel with job control
job_count=1
for theta in "${THETAS[@]}"; do
    echo "Starting theta=$theta"
    $BINARY "$TICKER" --race --theta "$theta" &

    ((job_count++))
    if ((job_count >= MAX_JOBS)); then
        wait -n  # Wait for any job to finish
        ((job_count--))
    fi
done

# Wait for remaining jobs
wait

echo ""
echo "All simulations complete."
