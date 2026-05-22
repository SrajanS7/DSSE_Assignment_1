#!/bin/bash
#SBATCH --job-name="Group13_Week4"
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu
#SBATCH --mem=240G
#SBATCH --output=week4_group13_%j.log

module purge
module load lang/Python/3.10.4-GCCcore-11.3.0
module load system/CUDA/12.4.0

source ~/Week4/venv/bin/activate

export HF_HOME=/scratch/hpc-prf-dssecs/group13
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="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"

echo "Starting Week 4 – $SCRIPT..."
python ~/Week4/$SCRIPT
echo "Done."
