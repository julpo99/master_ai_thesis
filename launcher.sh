#!/usr/bin/env bash

LOCAL="$(pwd)"
REMOTE="snellius:~/master_ai_thesis/"
SLURM_SCRIPT="run_job.sh"
REMOTE_OUT="snellius:~/master_ai_thesis/outputs/"
REMOTE_LOGS="snellius:~/master_ai_thesis/slurm-*.out"

LOCAL_OUT="./outputs/"

# Sync code to Snellius
rsync -az -q -e 'ssh -q' --delete \
  --exclude='build' \
  --exclude='.git/' \
  --exclude='.idea/' \
  --exclude='lgcn.egg-info/' \
  --exclude='outputs/' \
  --exclude='slurm-*.out' \
  --exclude='__pycache__/' \
  --exclude='**/__pycache__/' \
  --exclude='*.pyc' \
  "$LOCAL/" "$REMOTE"
echo "Synced code to Snellius"

# Submit job (non-blocking) and capture job ID
JOB_ID=$(ssh snellius "cd ~/master_ai_thesis && sbatch $SLURM_SCRIPT" | awk '{print $4}')
echo "➡️Job $JOB_ID submitted"

# Monitor job status
while true; do
  STATE=$(ssh snellius "squeue -j $JOB_ID -h -o %T")
  if [[ -z "$STATE" ]]; then
    echo "✅ Job $JOB_ID completed"
    break
  else
    echo "⏳ Job $JOB_ID status: $STATE"
    sleep 1
  fi
done

# Fetch output results
mkdir -p "$LOCAL_OUT"
rsync -az -q -e 'ssh -q' "$REMOTE_OUT" "$LOCAL_OUT"
rsync -az -q -e 'ssh -q' --delete "$REMOTE_LOGS" "$LOCAL"
echo "📥 Outputs synced"

# Clean up remote outputs and logs
ssh -q snellius << 'EOF' &> /dev/null
  rm -rf ~/master_ai_thesis/outputs/*
  rm -f ~/master_ai_thesis/slurm-*.out
EOF

echo -e "🎉 Done\n"