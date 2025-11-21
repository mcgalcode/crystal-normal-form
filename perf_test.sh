#!/bin/bash

PARTITION_DIR=$1
NUM_PROCS=$2
MAX_ITERS=$3

# Array to store PIDs of worker processes
PIDS=()

uv run python scripts/setup_search_db.py --xi 1.5 --delta 5 --partitions-dir $PARTITION_DIR --num-partitions=16 --start-cif Zr_HCP.cif --end-cif Zr_BCC.cif --supercell-index 1
# Trap Ctrl-C and kill all background processes
trap 'echo ""; echo "Caught Ctrl-C, killing all workers..."; kill ${PIDS[@]} 2>/dev/null; exit 1' INT

echo "Running pathfind with $PARTITION_DIR DB and NUM_PROCS=$NUM_PROCS and MAX_ITERS=$MAX_ITERS"
echo "Press Ctrl-C to stop all workers"

for i in $(seq 1 $NUM_PROCS); do
    uv run python scripts/pathfind_flood.py $PARTITION_DIR $MAX_ITERS &
    PIDS+=($!)
done

wait