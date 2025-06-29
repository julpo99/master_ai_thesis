#!/usr/bin/env bash

echo "⛔ Cancelling all your jobs on Snellius..."

ssh -q snellius "bash -s" << 'EOF'
  JOB_IDS=$(squeue -u $USER -h -o '%A')
  if [[ -z "$JOB_IDS" ]]; then
    echo "✅ No jobs to cancel."
  else
    echo "Found jobs: $JOB_IDS"
    scancel $JOB_IDS
    echo "🗑️  All jobs cancelled."
  fi
EOF