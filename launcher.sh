#!/bin/bash

# Preferred GPUs in order of priority
GPU_LIST=("A6000" "A100" "A5000" "A4000" "A2")
SELECTED_CONSTRAINT=""

# Query SLURM for each GPU type
for gpu in "${GPU_LIST[@]}"; do
    if sinfo -N -o "%G %t" | grep -q "$gpu.*idle"; then
        SELECTED_CONSTRAINT=$gpu
        break
    fi
done

# Exit if no GPU found
if [ -z "$SELECTED_CONSTRAINT" ]; then
  echo "No available GPUs found from: ${GPU_LIST[*]}"
  exit 1
fi

echo "Selected GPU: $SELECTED_CONSTRAINT"

# Submit job with selected constraint
sbatch --constraint=$SELECTED_CONSTRAINT slurm_job.sh