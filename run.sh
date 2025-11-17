#!/bin/bash
#SBATCH --job-name=bcic_array
#SBATCH --time=02:00:00             # Time limit (hh:mm:ss)
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --nodelist=cn7  
#SBATCH --output=output.log         # Output log file
#SBATCH --error=error.log  
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate myenv

export PYTHONWARNINGS="ignore::CryptographyDeprecationWarning"
export MNE_DATA="$HOME/mne_data"
cd "$SLURM_SUBMIT_DIR"

SUBJECT_ID="${SLURM_ARRAY_TASK_ID}"
srun python -W ignore::CryptographyDeprecationWarning run_bcic.py
