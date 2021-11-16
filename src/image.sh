#!/bin/bash

#SBATCH -J Proteus
#SBATCH --array=0-63

#SBATCH --job-name=Proteus_Image
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --time=1-00:00:00
#SBATCH --mem=10g

#SBATCH --output="%x.out%j"     # fichier de sortie ${SLURM_JOB_NAME}.out${SLURM_JOB_ID}
#SBATCH --comment="Running Proteus FDTD simulation for image generating"

# Load software
module load anaconda/3-2020.07
conda activate /home/brateaqu/PyAc/proteus

python3 src/Proteus_image.py 2 1000 120 0.001 1 1 2000 20 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT