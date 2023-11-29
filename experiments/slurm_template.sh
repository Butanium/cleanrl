#!/bin/sh
#SBATCH --job-name=${JOB_NAME}
#SBATCH --workdir=${PATH_TO_PROJECT_DIR}/experiments
#SBATCH --output=./logs/slurm_${JOB_NAME}.out
#SBATCH --error=./logs/errslurm_${JOB_NAME}.out
#SBATCH --time=15:00:00
#SBATCH --partition=single
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1


poetry run ${your command}