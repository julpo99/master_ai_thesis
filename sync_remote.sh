#!/usr/bin/env bash

LOCAL="$(pwd)"
REMOTE="snellius:~/master_ai_thesis/"


# Sync code to Snellius
rsync -azP --delete \
  --exclude='build' \
  --exclude='.git/' \
  --exclude='.idea/' \
  --exclude='lgcn.egg-info/' \
  "$LOCAL/" "$REMOTE"
echo "Synced code to Snellius"