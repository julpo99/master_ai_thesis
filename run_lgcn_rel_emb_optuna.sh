#!/bin/bash
#SBATCH --job-name=lgcn_rel_emb_optuna# Job name
#SBATCH --time=02:00:00               # Time limit (hh:mm:ss)
#SBATCH -N 1                          # Number of nodes
#SBATCH --partition=defq              # Default partition
#SBATCH --constraint=A6000            # GPU type
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --gres=gpu:1                  # Number of GPU cores per task
#SBATCH --output=output_%j.log        # Output log file (%j will be replaced with the job ID)


# Paths
SCRATCH=/var/scratch/jra223
SCRIPT_DIR=$SCRATCH/master_ai_thesis
TRAIN_PATH=$SCRIPT_DIR/"experiments/train.py"


# Load required modules for DAS6
 . /etc/bashrc
 . /etc/profile.d/lmod.sh
module load cuda12.6/toolkit
module load cuDNN/cuda12.6

# This loads the anaconda virtual environment with our packages
source $HOME/.bashrc
conda activate /var/scratch/jra223/master_ai_thesis/env


# Check which Python is active
#which python
#python --version

#echo "Running on node: $(hostname)"
#echo "Using GPU cores: $SLURM_NTASKS"

# Base directory for the experiment (only if not exists)
if [ ! -d $SCRIPT_DIR/outputs ]; then
  mkdir $SCRIPT_DIR/outputs
fi
# cd if the directory exists
if [ -d $SCRIPT_DIR/outputs ]; then
  cd $SCRIPT_DIR/outputs
else
  echo "Directory $SCRIPT_DIR/outputs does not exist."
  exit 1
fi

# Simple trick to create a unique directory for each run of the script
echo $$
mkdir o`echo $$`
cd o`echo $$`


#python <<EOF
#import torch
#print(torch.cuda.is_available())
#print(torch.cuda.get_device_name(0))
#EOF


# Run the actual experiment
python -u $TRAIN_PATH > 'output.out'