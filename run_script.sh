#!/bin/bash
#SBATCH --job-name=lgcn               # Job name
#SBATCH --time=00:15:00               # Time limit (hh:mm:ss)
#SBATCH -N 1                          # Number of nodes
#SBATCH --partition=defq              # Default partition
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --gres=gpu:1                  # Number of GPU cores per task
#SBATCH --output=output_%j.log    # Output log file (%j will be replaced with the job ID)


# Paths
SCRATCH=/var/scratch/jra223
SCRIPT_DIR=$SCRATCH/master_ai_thesis
TRAIN_PATH=$SCRIPT_DIR"experiments/train.py"


# Load required modules for DAS6
 . /etc/bashrc
 . /etc/profile.d/lmod.sh
module load cuda12.3/toolkit
module load cuDNN/cuda12.3


source ~/.bashrc

# This loads the virtual environment with our packages
source /var/scratch/jra223/master_ai_thesis/.venv/bin/activate

# Check which Python is active (important!)
which python
python --version

#echo "Running on node: $(hostname)"
#echo "Using GPU cores: $SLURM_NTASKS"

# Base directory for the experiment
mkdir $SCRIPT_DIR/experiments_output
cd $SCRIPT_DIR/experiments_output

# Simple trick to create a unique directory for each run of the script
echo $$
mkdir o`echo $$`
cd o`echo $$`

# Run the actual experiment
#python -u $TRAIN_PATH > 'output.out'

python <<EOF
import torch
print(torch.cuda.is_available())
EOF