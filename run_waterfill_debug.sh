#!/bin/bash
#SBATCH --job-name=cnf-waterfill-debug
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=cpu
#SBATCH --output=logs/waterfill_%j.out
#SBATCH --error=logs/waterfill_%j.err

# =============================================================================
# CNF Water-Filling Search - Debug Job
# =============================================================================
#
# BEFORE RUNNING:
# 1. Edit PARTITION_DB_DIR to point to your partition database directory
# 2. Edit CONDA_ENV if using a different environment name
# 3. Adjust resources above if needed (nodes, tasks, time)
# 4. Create logs directory: mkdir -p $SCRATCH/cnf_project/logs
#
# TO SUBMIT:
#   sbatch run_waterfill_debug.sh
#
# TO MONITOR:
#   squeue -u $USER
#   tail -f logs/waterfill_<jobid>.out
# =============================================================================

# --- CONFIGURATION (EDIT THESE) ---
# Path to your partition database directory
PARTITION_DB_DIR="$SCRATCH/cnf_project/zr_search_waterfill_32"

# Conda environment name
CONDA_ENV="cnf"

# Search parameters
MAX_ITERS=1000           # Maximum iterations for debug run
LOG_LEVEL=2              # 0=FATAL, 1=SEVERE, 2=WARN, 3=INFO, 4=DEBUG
# --- END CONFIGURATION ---

echo "========================================"
echo "CNF Water-Filling Search - Debug Job"
echo "========================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Start time:   $(date)"
echo "Partition DB: $PARTITION_DB_DIR"
echo "========================================"
echo

# Load conda (NERSC-specific)
module load python
source activate $CONDA_ENV

# Verify environment
echo "Python:       $(which python)"
echo "Environment:  $CONDA_NAME"
echo

# Check that partition database exists
if [ ! -d "$PARTITION_DB_DIR" ]; then
    echo "ERROR: Partition database directory not found: $PARTITION_DB_DIR"
    exit 1
fi

# Count partition files
NUM_PARTITIONS=$(ls -1 "$PARTITION_DB_DIR"/graph_partition_*.db 2>/dev/null | wc -l)
echo "Found $NUM_PARTITIONS partition databases"
echo

# Run the water-filling search
echo "Starting water-filling search..."
echo "Command: cnf-pathfind-waterfill $PARTITION_DB_DIR --max-iters $MAX_ITERS --log-lvl $LOG_LEVEL"
echo

cnf-pathfind-waterfill "$PARTITION_DB_DIR" \
    --max-iters $MAX_ITERS \
    --log-lvl $LOG_LEVEL

EXIT_CODE=$?

echo
echo "========================================"
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================"

exit $EXIT_CODE
