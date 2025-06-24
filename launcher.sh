#!/usr/bin/env bash

LOCAL="$(pwd)"
REMOTE="snellius:~/master_ai_thesis/"
SLURM_SCRIPT="run_job.sh"
REMOTE_OUT="snellius:~/master_ai_thesis/outputs/"
REMOTE_LOGS="snellius:~/master_ai_thesis/slurm-*.out"

LOCAL_OUT="./outputs/"

# Sync code to Snellius
rsync -azP --delete \
  --exclude='build' \
  --exclude='.git/' \
  --exclude='.idea/' \
  --exclude='lgcn.egg-info/' \
  "$LOCAL/" "$REMOTE"
echo "Synced code to Snellius"

# Submit job (non-blocking) and capture job ID
JOB_ID=$(ssh snellius "cd ~/master_ai_thesis && sbatch $SLURM_SCRIPT" | awk '{print $4}')
echo "Submitted job $JOB_ID"

# Monitor job status
while true; do
  STATE=$(ssh snellius "squeue -j $JOB_ID -h -o %T")
  if [[ -z "$STATE" ]]; then
    echo "Job $JOB_ID completed"
    break
  else
    echo "Job $JOB_ID status: $STATE"
    sleep 1
  fi
done

# Fetch output results
echo "Fetching job outputs..."
mkdir -p "$LOCAL_OUT"
rsync -azP --delete "$REMOTE_OUT" "$LOCAL_OUT"
rsync -azP --delete "$REMOTE_LOGS" "$LOCAL"

echo "Done. Outputs are in $LOCAL_OUT"