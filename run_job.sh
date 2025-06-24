#!/bin/bash
#SBATCH --job-name=lgcn               # Job name
#SBATCH --time=24:00:00               # Time limit (hh:mm:ss)
#SBATCH -N 1                          # Number of nodes
#SBATCH --partition=gpu_h100          # Default partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --gres=gpu:1                  # Number of GPU cores per task


# Paths
PROJECT=~/master_ai_thesis
TRAIN_SCRIPT=$PROJECT/"experiments/train.py"

# Load anaconda virtual environment
source $HOME/.bashrc
conda activate conda_env


# Check which Python is active
#which python
#python --version

#echo "Running on node: $(hostname)"
#echo "Using GPU cores: $SLURM_NTASKS"

# Base directory for the experiment (only if not exists)
if [ ! -d $PROJECT/outputs ]; then
  mkdir $PROJECT/outputs
fi
# cd if the directory exists
if [ -d $PROJECT/outputs ]; then
  cd $PROJECT/outputs
else
  echo "Directory $PROJECT/outputs does not exist."
  exit 1
fi

# Simple trick to create a unique directory for each run of the script
JOBID=${SLURM_JOB_ID}
echo $JOBID
mkdir -p o-$JOBID
cd o-$JOBID


#python <<EOF
#import torch
#print(torch.cuda.is_available())
#print(torch.cuda.get_device_name(0))
#EOF

# Ensure Python can find project modules
export PYTHONPATH="$HOME/master_ai_thesis:$PYTHONPATH"


# Run the actual experiment
python -u $TRAIN_SCRIPT > 'output.out'