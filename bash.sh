#!/bin/bash
#SBATCH --job-name=cctv_analysis    # Job name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1        # Number of tasks per node
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --mem=32G                  # Memory per node
#SBATCH --partition=gpu            # GPU partition

# Load required modules
module purge
module load cuda/11.8
module load cudnn/8.6.0
module load anaconda3/2023.03

# Create new environment from yml
log "Creating new conda environment from environment.yml..."
conda env create -f "$PROJECT_ROOT/environment.yml"

if [ $? -ne 0 ]; then
    log "ERROR: Failed to create conda environment"
    exit 1
fi

# Activate environment and verify installation
log "Verifying installation..."
# We need to source conda.sh to use conda activate in script
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cctv-analysis

# Run the job
python src/main.py